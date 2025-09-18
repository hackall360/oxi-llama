use rand::{Rng, SeedableRng, rngs::StdRng};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub id: i32,
    pub value: f32,
}

/// Sampler provides probabilistic sampling with optional
/// top-k, top-p, min-p and temperature transforms.
pub struct Sampler {
    rng: Option<StdRng>,
    top_k: Option<usize>,
    top_p: f32,
    min_p: f32,
    temperature: f32,
}

impl Sampler {
    /// Create a new sampler.
    pub fn new(temperature: f32, top_k: i32, top_p: f32, min_p: f32, seed: i32) -> Self {
        let rng = if seed != -1 {
            Some(StdRng::seed_from_u64(seed as u64))
        } else {
            None
        };

        let temperature = if temperature < 0.0 { 0.0 } else { temperature };
        let mut top_p = top_p;
        if top_p < 0.0 {
            top_p = 0.0;
        }
        if top_p >= 1.0 {
            top_p = 1.0;
        }
        let mut min_p = min_p;
        if min_p < 0.0 {
            min_p = 0.0;
        }
        if min_p >= 1.0 {
            min_p = 1.0;
        }
        let top_k = if top_k <= 0 {
            None
        } else {
            Some(top_k as usize)
        };

        Sampler {
            rng,
            top_k,
            top_p,
            min_p,
            temperature,
        }
    }

    /// Sample from the provided logits and return the selected token id.
    pub fn sample(&mut self, logits: &[f32]) -> Result<i32, String> {
        if logits.is_empty() {
            return Err("sample: no logits provided to sample".into());
        }
        let mut tokens: Vec<Token> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| Token {
                id: i as i32,
                value: v,
            })
            .collect();
        let token = self.inner_sample(&mut tokens)?;
        Ok(token.id)
    }

    fn inner_sample(&mut self, tokens: &mut Vec<Token>) -> Result<Token, String> {
        if self.temperature == 0.0 {
            return Ok(greedy(tokens));
        }

        top_k(tokens, self.top_k);
        temperature(tokens, self.temperature);
        softmax(tokens);
        top_p(tokens, self.top_p);
        min_p(tokens, self.min_p);

        let r = match &mut self.rng {
            Some(rng) => rng.r#gen::<f32>(),
            None => {
                let mut rng = rand::thread_rng();
                rng.r#gen::<f32>()
            }
        };

        // cumulative distribution
        let mut sum = 0.0f32;
        for t in tokens.iter_mut() {
            sum += t.value;
            t.value = sum;
        }
        if sum.is_nan() {
            return Err("sample: logits sum to NaN, check model output".into());
        }
        let r = r * sum;
        let idx = match tokens.binary_search_by(|t| {
            if t.value < r {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }) {
            Ok(i) | Err(i) => i,
        };
        Ok(tokens[idx])
    }
}

pub fn greedy(tokens: &[Token]) -> Token {
    *tokens
        .iter()
        .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal))
        .unwrap()
}

pub fn temperature(ts: &mut [Token], temp: f32) {
    let temp = temp.max(1e-7);
    for t in ts {
        t.value /= temp;
    }
}

pub fn softmax(ts: &mut [Token]) {
    let max_logit = ts.iter().map(|t| t.value).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for t in ts.iter_mut() {
        t.value = (t.value - max_logit).exp();
        sum += t.value;
    }
    for t in ts.iter_mut() {
        t.value /= sum;
    }
}

pub fn top_k(ts: &mut Vec<Token>, k: Option<usize>) {
    match k {
        None => ts.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal)),
        Some(k) => {
            if k >= ts.len() {
                ts.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal));
            } else {
                ts.select_nth_unstable_by(k, |a, b| {
                    b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal)
                });
                ts.truncate(k);
                ts.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal));
            }
        }
    }
}

pub fn top_p(ts: &mut Vec<Token>, p: f32) {
    if (p - 1.0).abs() < f32::EPSILON {
        return;
    }
    let mut sum = 0.0f32;
    let mut retain = ts.len();
    for (i, t) in ts.iter().enumerate() {
        sum += t.value;
        if sum > p {
            retain = i + 1;
            break;
        }
    }
    ts.truncate(retain);
}

pub fn min_p(ts: &mut Vec<Token>, p: f32) {
    if ts.is_empty() {
        return;
    }
    let max_prob = ts[0].value;
    let threshold = max_prob * p;
    let mut retain = ts.len();
    for (i, t) in ts.iter().enumerate() {
        if t.value < threshold {
            retain = i;
            break;
        }
    }
    ts.truncate(retain);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_tokens(values: &[f32]) -> Vec<Token> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| Token {
                id: i as i32,
                value: v,
            })
            .collect()
    }

    fn compare_logits(want: &[f32], got: &[Token]) {
        assert_eq!(want.len(), got.len());
        for (w, t) in want.iter().zip(got.iter()) {
            assert!((t.value - w).abs() < 1e-6, "expected {w}, got {}", t.value);
        }
    }

    #[test]
    fn weighted_basic() {
        let mut sampler = Sampler::new(0.0, 0, 0.0, 0.0, 0);
        let logits = vec![-10.0, 3.0, -10.0, -10.0];
        let id = sampler.sample(&logits).unwrap();
        assert_eq!(id, 1);

        let logits = vec![-100.0, -10.0, 0.0, 10.0];
        let id = sampler.sample(&logits).unwrap();
        assert_eq!(id, 3);

        let logits = vec![1.0, 0.9999999999999999, 0.5, 0.1];
        let mut sampler = Sampler::new(1.0, 0, 1e-10, 0.0, 0);
        let id = sampler.sample(&logits).unwrap();
        assert_eq!(id, 0);
    }

    #[test]
    fn nan_logits_error() {
        let logits = vec![f32::NAN, f32::NAN, f32::NAN];
        let mut sampler = Sampler::new(1.0, 0, 0.95, 0.05, 0);
        assert!(sampler.sample(&logits).is_err());
    }

    #[test]
    fn temperature_transform() {
        let mut tokens = to_tokens(&[1.0, 4.0, -2.0, 0.0]);
        temperature(&mut tokens, 0.5);
        compare_logits(&[2.0, 8.0, -4.0, 0.0], &tokens);

        let mut tokens = to_tokens(&[1.0, 4.0, -2.0, 0.0]);
        temperature(&mut tokens, 1.0);
        compare_logits(&[1.0, 4.0, -2.0, 0.0], &tokens);

        let mut tokens = to_tokens(&[1.0, 4.0, -2.0, 0.0]);
        temperature(&mut tokens, 0.0);
        compare_logits(&[1e7, 4e7, -2e7, 0.0], &tokens);
    }

    #[test]
    fn softmax_basic() {
        let mut tokens = to_tokens(&[1.0, -2.0, 3.0, 0.0]);
        softmax(&mut tokens);
        compare_logits(&[0.113550, 0.005653, 0.839024, 0.041773], &tokens);

        let mut tokens = to_tokens(&[
            0.026986899,
            0.043722924,
            0.036774673,
            0.27755088,
            0.0046718004,
            0.08582123,
            0.20409796,
            0.00412893,
            0.15720603,
            0.045046154,
            0.0030491839,
            0.01681367,
        ]);
        softmax(&mut tokens);
        let sum: f32 = tokens.iter().map(|t| t.value).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for t in tokens {
            assert!(t.value >= 0.0 && t.value <= 1.0);
        }
    }

    #[test]
    fn topk_basic() {
        let input = [
            0.026986899,
            0.043722924,
            0.036774673,
            0.27755088,
            0.0046718004,
            0.08582123,
            0.20409796,
            0.00412893,
            0.15720603,
            0.045046154,
            0.0030491839,
            0.01681367,
        ];
        let mut tokens = to_tokens(&input);
        top_k(&mut tokens, Some(5));
        compare_logits(
            &[0.27755088, 0.20409796, 0.15720603, 0.08582123, 0.045046154],
            &tokens,
        );

        let mut tokens = to_tokens(&input);
        top_k(&mut tokens, Some(20));
        assert_eq!(tokens.len(), input.len());

        let mut tokens = to_tokens(&input);
        top_k(&mut tokens, None);
        compare_logits(
            &[
                0.27755088,
                0.20409796,
                0.15720603,
                0.08582123,
                0.045046154,
                0.043722924,
                0.036774673,
                0.026986899,
                0.01681367,
                0.0046718004,
                0.00412893,
                0.0030491839,
            ],
            &tokens,
        );
    }

    #[test]
    fn topp_basic() {
        let input = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0];
        let mut tokens = to_tokens(&input);
        softmax(&mut tokens);
        top_k(&mut tokens, None);
        let mut got = tokens.clone();
        top_p(&mut got, 1.0);
        assert_eq!(got.len(), input.len());

        let mut got = tokens.clone();
        top_p(&mut got, 0.95);
        assert!(got.len() <= 3);

        let input = [-1e6, -1e6, -1e7];
        let mut tokens = to_tokens(&input);
        top_k(&mut tokens, None);
        softmax(&mut tokens);
        let mut got = tokens.clone();
        top_p(&mut got, 0.0);
        assert_eq!(got.len(), 1);

        let mut got = tokens.clone();
        top_p(&mut got, 1e-10);
        assert!(got.len() >= 1);
    }

    #[test]
    fn minp_basic() {
        let input = [-2.0, 0.0, -1.0, -3.0, 2.0, 1.0, 4.0, 3.0];
        let mut tokens = to_tokens(&input);
        top_k(&mut tokens, None);
        softmax(&mut tokens);
        let mut got = tokens.clone();
        min_p(&mut got, 1.0);
        assert_eq!(got.len(), 1);

        let mut tokens = to_tokens(&input);
        top_k(&mut tokens, None);
        softmax(&mut tokens);
        let mut got = tokens.clone();
        min_p(&mut got, 0.2);
        assert!(got.len() <= 3);

        let mut tokens = to_tokens(&input);
        top_k(&mut tokens, None);
        softmax(&mut tokens);
        let mut got = tokens.clone();
        min_p(&mut got, 0.0);
        assert_eq!(got.len(), tokens.len());

        let mut tokens = to_tokens(&input[..1]);
        top_k(&mut tokens, None);
        softmax(&mut tokens);
        let mut got = tokens.clone();
        min_p(&mut got, 0.1);
        assert_eq!(got.len(), 1);

        let input = [1e-10f32, 1e-10, 1e-10];
        let mut tokens = to_tokens(&input);
        softmax(&mut tokens);
        let mut got = tokens.clone();
        min_p(&mut got, 1.0);
        assert!(got.len() >= 1);
    }
}
