use std::time::SystemTime;

use crate::KvCache;

#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    Token(i32),
    Embed(Vec<f32>),
    Multimodal(u64),
}

#[derive(Clone, Debug)]
pub struct InputCacheSlot {
    pub id: usize,
    pub inputs: Vec<Input>,
    pub in_use: bool,
    pub last_used: SystemTime,
}

impl InputCacheSlot {
    fn new(id: usize) -> Self {
        Self {
            id,
            inputs: Vec::new(),
            in_use: false,
            last_used: SystemTime::UNIX_EPOCH,
        }
    }
}

pub struct InputCache {
    pub num_ctx: usize,
    pub slots: Vec<InputCacheSlot>,
    pub multi_user_cache: bool,
    pub cache: Option<Box<dyn KvCache + Send>>, // optional backend
}

impl InputCache {
    pub fn new(num_ctx: usize, num_slots: usize) -> Self {
        let mut slots = Vec::new();
        for i in 0..num_slots {
            slots.push(InputCacheSlot::new(i));
        }
        Self {
            num_ctx,
            slots,
            multi_user_cache: false,
            cache: None,
        }
    }

    pub fn count_common_prefix(a: &[Input], b: &[Input]) -> usize {
        let mut count = 0;
        for (i, ai) in a.iter().enumerate() {
            if i >= b.len() {
                break;
            }
            if ai != &b[i] {
                break;
            }
            count += 1;
        }
        count
    }

    /// Locate a cache slot for the given prompt, returning the slot index and
    /// the remaining portion of the prompt that still needs processing.
    pub fn load_cache_slot(
        &mut self,
        prompt: Vec<Input>,
        cache_prompt: bool,
    ) -> Result<(usize, Vec<Input>), &'static str> {
        let (slot_idx, mut num_past) = if self.multi_user_cache {
            self.find_best_cache_slot(&prompt)?
        } else {
            self.find_longest_cache_slot(&prompt)?
        };

        if !cache_prompt {
            num_past = 0;
        }

        let slot = &mut self.slots[slot_idx];
        slot.in_use = true;
        slot.last_used = SystemTime::now();

        if num_past == prompt.len() && num_past > 0 {
            num_past -= 1; // ensure at least one token to generate
        }

        slot.inputs = prompt[..num_past].to_vec();

        let remaining = prompt[num_past..].to_vec();
        Ok((slot_idx, remaining))
    }

    pub fn find_longest_cache_slot(
        &self,
        prompt: &[Input],
    ) -> Result<(usize, usize), &'static str> {
        let mut longest = None; // (count, index)
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.in_use {
                continue;
            }
            let count = Self::count_common_prefix(&slot.inputs, prompt);
            if longest.map_or(true, |(c, _i)| count > c) {
                longest = Some((count, i));
            }
        }
        if let Some((count, idx)) = longest {
            Ok((idx, count))
        } else {
            Err("no available cache slots")
        }
    }

    pub fn find_best_cache_slot(
        &mut self,
        prompt: &[Input],
    ) -> Result<(usize, usize), &'static str> {
        let mut longest = (0usize, None); // (count, index)
        let mut oldest = (SystemTime::now(), None); // (time, index)
        for (i, slot) in self.slots.iter().enumerate() {
            let count = Self::count_common_prefix(&slot.inputs, prompt);
            if count > longest.0 {
                longest = (count, Some(i));
            }
            if !slot.in_use {
                if let Some(_current_oldest) = oldest.1 {
                    let t = slot.last_used;
                    if t < oldest.0 {
                        oldest = (t, Some(i));
                    }
                } else {
                    oldest = (slot.last_used, Some(i));
                }
            }
        }
        let longest_idx = longest.1.ok_or("no available cache slots")?;
        if longest.0 == self.slots[longest_idx].inputs.len() && !self.slots[longest_idx].in_use {
            return Ok((longest_idx, longest.0));
        }
        let oldest_idx = oldest.1.ok_or("no available cache slots")?;
        if self.slots[oldest_idx].in_use {
            return Err("no available cache slots");
        }
        if longest.0 > 0 && longest_idx != oldest_idx {
            let prefix = self.slots[longest_idx].inputs[..longest.0].to_vec();
            self.slots[oldest_idx].inputs = prefix;
            if let Some(cache) = self.cache.as_mut() {
                let _ = cache.remove(oldest_idx, 0, -1); // clear
                cache.copy_prefix(longest_idx, oldest_idx, longest.0 as i32);
            }
        }
        Ok((oldest_idx, longest.0))
    }

    pub fn shift_discard(&self, input_len: usize, num_keep: usize) -> usize {
        let mut target_free = (self.num_ctx - num_keep) as isize / 2;
        if target_free < 1 {
            target_free = 1;
        }
        let current_free = self.num_ctx as isize - input_len as isize;
        let mut discard = target_free - current_free;
        if discard < 0 {
            discard = 0;
        }
        discard as usize
    }

    /// Frees up space in the cache by discarding oldest entries and shifting the rest.
    pub fn shift_cache_slot(
        &mut self,
        slot_id: usize,
        num_keep: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if num_keep >= self.num_ctx {
            return Err(format!(
                "unable to shift context - keep exceeds context (keep: {num_keep} context: {ctx})",
                ctx = self.num_ctx
            )
            .into());
        }
        let input_len = self.slots[slot_id].inputs.len();
        let discard = self.shift_discard(input_len, num_keep);
        let slot = &mut self.slots[slot_id];
        if discard == 0 {
            return Ok(());
        }
        if let Some(cache) = self.cache.as_mut() {
            if cache.can_shift() {
                if cache
                    .remove(slot_id, num_keep as i32, (num_keep + discard) as i32)
                    .is_err()
                {
                    let prefix = slot.inputs[..num_keep].to_vec();
                    let new_inputs = slot.inputs[num_keep + discard..].to_vec();
                    let _ = cache.remove(slot_id, 0, -1); // clear
                    slot.inputs.clear();
                    return Err(Box::new(ErrReprocessInputs {
                        inputs: [prefix, new_inputs].concat(),
                    }));
                } else {
                    cache.shift(
                        slot_id,
                        (num_keep + discard) as i32,
                        input_len as i32,
                        -(discard as i32),
                    );
                }
            } else {
                let prefix = slot.inputs[..num_keep].to_vec();
                let new_inputs = slot.inputs[num_keep + discard..].to_vec();
                let _ = cache.remove(slot_id, 0, -1);
                slot.inputs.clear();
                return Err(Box::new(ErrReprocessInputs {
                    inputs: [prefix, new_inputs].concat(),
                }));
            }
        }
        for i in num_keep + discard..input_len {
            slot.inputs[i - discard] = slot.inputs[i].clone();
        }
        slot.inputs.truncate(input_len - discard);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ErrReprocessInputs {
    pub inputs: Vec<Input>,
}

impl std::fmt::Display for ErrReprocessInputs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "kv cache shift not supported, inputs need reprocessing (input count: {})",
            self.inputs.len()
        )
    }
}

impl std::error::Error for ErrReprocessInputs {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn test_count_common() {
        let cases = vec![
            (
                vec![Input::Token(1), Input::Token(2), Input::Token(3)],
                vec![Input::Token(1), Input::Token(2), Input::Token(3)],
                3,
            ),
            (
                vec![Input::Token(1)],
                vec![Input::Token(1), Input::Token(2), Input::Token(3)],
                1,
            ),
            (
                vec![Input::Embed(vec![0.1, 0.2, 0.3])],
                vec![
                    Input::Embed(vec![0.1, 0.2, 0.3]),
                    Input::Embed(vec![0.4, 0.5, 0.6]),
                    Input::Embed(vec![0.7]),
                ],
                1,
            ),
            (
                vec![Input::Embed(vec![0.1, 0.2, 0.3])],
                vec![
                    Input::Embed(vec![0.1, 0.2]),
                    Input::Embed(vec![0.4, 0.5, 0.6]),
                    Input::Embed(vec![0.7]),
                ],
                0,
            ),
            (
                vec![Input::Token(1), Input::Embed(vec![0.2, 0.3, 0.4])],
                vec![
                    Input::Token(1),
                    Input::Embed(vec![0.2, 0.3, 0.4]),
                    Input::Token(5),
                ],
                2,
            ),
            (vec![], vec![Input::Token(1), Input::Token(2)], 0),
            (vec![], vec![], 0),
        ];
        for (a, b, expected) in cases {
            let result = InputCache::count_common_prefix(&a, &b);
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn test_find_cache_slot() {
        // Setup tests similar to Go version
        let now = SystemTime::now();
        let mut tests = Vec::new();
        // Case Extend
        tests.push((
            "Extend",
            InputCache {
                num_ctx: 0,
                slots: vec![
                    InputCacheSlot {
                        id: 0,
                        inputs: vec![Input::Token(1)],
                        in_use: false,
                        last_used: now - Duration::from_secs(1),
                    },
                    InputCacheSlot {
                        id: 1,
                        inputs: vec![Input::Token(1), Input::Token(2)],
                        in_use: false,
                        last_used: now - Duration::from_secs(2),
                    },
                ],
                multi_user_cache: false,
                cache: None,
            },
            vec![Input::Token(1), Input::Token(2)],
            (1, 2),
            (1, 2),
        ));
        // In use case
        tests.push((
            "In use",
            InputCache {
                num_ctx: 0,
                slots: vec![
                    InputCacheSlot {
                        id: 0,
                        inputs: vec![Input::Token(1), Input::Token(2)],
                        in_use: true,
                        last_used: now - Duration::from_secs(1),
                    },
                    InputCacheSlot {
                        id: 1,
                        inputs: vec![Input::Token(1)],
                        in_use: false,
                        last_used: now - Duration::from_secs(2),
                    },
                ],
                multi_user_cache: false,
                cache: None,
            },
            vec![Input::Token(1), Input::Token(2)],
            (1, 1),
            (1, 2),
        ));

        for (name, mut cache, prompt, longest_expect, best_expect) in tests {
            let (idx, len) = cache.find_longest_cache_slot(&prompt).unwrap();
            assert_eq!((idx, len), longest_expect, "{} longest", name);
            let (idxb, lenb) = cache.find_best_cache_slot(&prompt).unwrap();
            assert_eq!((idxb, lenb), best_expect, "{} best", name);
        }
    }

    #[test]
    fn test_shift_discard() {
        let cases = vec![
            ("Shift", 2048usize, 5usize, 2048usize, 1021usize),
            ("Max Keep", 2048, 2047, 2048, 1),
            ("No Keep", 2048, 0, 2048, 1024),
            ("Truncate", 2048, 5, 5000, 3973),
            ("Truncate Keep", 2048, 2047, 5000, 2953),
            ("No Op", 2048, 5, 512, 0),
        ];
        for (name, num_ctx, num_keep, input_len, expected) in cases {
            let c = InputCache::new(num_ctx, 1);
            let result = c.shift_discard(input_len, num_keep);
            assert_eq!(result, expected, "{}", name);
        }
    }
}
