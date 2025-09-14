use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
struct Token {
    id: i32,
    value: f32,
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
        if top_p < 0.0 { top_p = 0.0; }
        if top_p >= 1.0 { top_p = 1.0; }
        let mut min_p = min_p;
        if min_p < 0.0 { min_p = 0.0; }
        if min_p >= 1.0 { min_p = 1.0; }
        let top_k = if top_k <= 0 { None } else { Some(top_k as usize) };

        Sampler { rng, top_k, top_p, min_p, temperature }
    }

    /// Sample from the provided logits and return the selected token id.
    pub fn sample(&mut self, logits: &[f32]) -> Result<i32, String> {
        if logits.is_empty() {
            return Err("sample: no logits provided to sample".into());
        }
        let mut tokens: Vec<Token> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| Token { id: i as i32, value: v })
            .collect();
        let token = self.inner_sample(&mut tokens)?;
        Ok(token.id)
    }

    fn inner_sample(&mut self, tokens: &mut Vec<Token>) -> Result<Token, String> {
        if self.temperature == 0.0 {
            return Ok(greedy(tokens));
        }

        top_k(tokens, self.top_k);
        temperature_transform(tokens, self.temperature);
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

fn greedy(tokens: &[Token]) -> Token {
    *tokens
        .iter()
        .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal))
        .unwrap()
}

fn temperature_transform(ts: &mut [Token], temp: f32) {
    let temp = temp.max(1e-7);
    for t in ts {
        t.value /= temp;
    }
}

fn softmax(ts: &mut [Token]) {
    let max_logit = ts
        .iter()
        .map(|t| t.value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for t in ts.iter_mut() {
        t.value = (t.value - max_logit).exp();
        sum += t.value;
    }
    for t in ts.iter_mut() {
        t.value /= sum;
    }
}

fn top_k(ts: &mut Vec<Token>, k: Option<usize>) {
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

fn top_p(ts: &mut Vec<Token>, p: f32) {
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

fn min_p(ts: &mut Vec<Token>, p: f32) {
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
}
