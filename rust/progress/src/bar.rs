use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::progress::State;
use console::Term;

pub struct Bar {
    inner: Mutex<BarInner>,
}

struct BarInner {
    message: String,
    message_width: isize,
    max_value: i64,
    initial_value: i64,
    current_value: i64,
    started: Instant,
    stopped: Option<Instant>,
    buckets: Vec<Bucket>,
    max_buckets: usize,
}

struct Bucket {
    updated: Instant,
    value: i64,
}

impl Bar {
    pub fn new(message: impl Into<String>, max_value: i64, initial_value: i64) -> Self {
        let inner = BarInner {
            message: message.into(),
            message_width: -1,
            max_value,
            initial_value,
            current_value: initial_value,
            started: Instant::now(),
            stopped: if initial_value >= max_value {
                Some(Instant::now())
            } else {
                None
            },
            buckets: Vec::new(),
            max_buckets: 10,
        };
        Bar {
            inner: Mutex::new(inner),
        }
    }

    pub fn set(&self, value: i64) {
        let mut inner = self.inner.lock().unwrap();
        let value = value.min(inner.max_value);
        inner.current_value = value;
        if inner.current_value >= inner.max_value {
            if inner.stopped.is_none() {
                inner.stopped = Some(Instant::now());
            }
        }
        if inner.buckets.is_empty()
            || inner.buckets.last().unwrap().updated.elapsed() > Duration::from_secs(1)
        {
            inner.buckets.push(Bucket {
                updated: Instant::now(),
                value,
            });
            if inner.buckets.len() > inner.max_buckets {
                inner.buckets.remove(0);
            }
        }
    }

    fn percent(inner: &BarInner) -> f64 {
        if inner.max_value > 0 {
            inner.current_value as f64 / inner.max_value as f64 * 100.0
        } else {
            0.0
        }
    }

    fn rate(inner: &BarInner) -> f64 {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        if let Some(stopped) = inner.stopped {
            numerator = (inner.current_value - inner.initial_value) as f64;
            denominator = stopped.duration_since(inner.started).as_secs_f64();
        } else {
            match inner.buckets.len() {
                0 => {}
                1 => {
                    numerator = (inner.buckets[0].value - inner.initial_value) as f64;
                    denominator = inner.buckets[0]
                        .updated
                        .duration_since(inner.started)
                        .as_secs_f64();
                }
                _ => {
                    let first = &inner.buckets[0];
                    let last = inner.buckets.last().unwrap();
                    numerator = (last.value - first.value) as f64;
                    denominator = last.updated.duration_since(first.updated).as_secs_f64();
                }
            }
        }
        if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }

    pub fn to_string(&self) -> String {
        let inner = self.inner.lock().unwrap();
        let term = Term::stderr();
        let (_, term_width) = term.size();
        let term_width = term_width as usize;

        let mut pre = String::new();
        if !inner.message.is_empty() {
            let mut message = inner.message.trim().to_string();
            if inner.message_width > 0 && message.len() > inner.message_width as usize {
                message.truncate(inner.message_width as usize);
            }
            pre.push_str(&message);
            if inner.message_width > 0 {
                let pad = inner.message_width as usize - message.len();
                if pad > 0 {
                    pre.push_str(&" ".repeat(pad));
                }
            }
            pre.push(' ');
        }
        pre.push_str(&format!("{:>3.0}%", Bar::percent(&inner)));

        let mut suf = String::new();
        if inner.stopped.is_none() {
            let cur = human_bytes(inner.current_value);
            suf.push_str(&" ".repeat(6 - cur.len()));
            suf.push_str(&cur);
            suf.push('/');
            let max = human_bytes(inner.max_value);
            suf.push_str(&" ".repeat(6 - max.len()));
            suf.push_str(&max);
        } else {
            let max = human_bytes(inner.max_value);
            suf.push_str(&" ".repeat(6 - max.len()));
            suf.push_str(&max);
            suf.push_str(&" ".repeat(7));
        }

        let rate = Bar::rate(&inner);
        if inner.stopped.is_none() && rate > 0.0 {
            suf.push_str("  ");
            let hr = human_bytes(rate as i64);
            suf.push_str(&" ".repeat(6 - hr.len()));
            suf.push_str(&hr);
            suf.push_str("/s");
        } else {
            suf.push_str(&" ".repeat(10));
        }

        if inner.stopped.is_none() && rate > 0.0 {
            suf.push_str("  ");
            let remaining =
                Duration::from_secs_f64((inner.max_value - inner.current_value) as f64 / rate);
            let human_rem = format_duration(remaining);
            suf.push_str(&" ".repeat(6 - human_rem.len()));
            suf.push_str(&human_rem);
        } else {
            suf.push_str(&" ".repeat(8));
        }

        let mut mid = String::new();
        let f = term_width as isize - pre.len() as isize - suf.len() as isize - 5;
        let f = if f > 0 { f as usize } else { 0 };
        let n = ((f as f64) * Bar::percent(&inner) / 100.0) as usize;
        mid.push_str(" ▕");
        if n > 0 {
            mid.push_str(&"█".repeat(n));
        }
        if f > n {
            mid.push_str(&" ".repeat(f - n));
        }
        mid.push_str("▏ ");

        format!("{}{}{}", pre, mid, suf)
    }
}

impl State for Bar {
    fn render(&self) -> String {
        self.to_string()
    }
}

fn human_bytes(b: i64) -> String {
    const KB: i64 = 1000;
    const MB: i64 = KB * 1000;
    const GB: i64 = MB * 1000;
    const TB: i64 = GB * 1000;
    let (value, unit) = if b >= TB {
        (b as f64 / TB as f64, "TB")
    } else if b >= GB {
        (b as f64 / GB as f64, "GB")
    } else if b >= MB {
        (b as f64 / MB as f64, "MB")
    } else if b >= KB {
        (b as f64 / KB as f64, "KB")
    } else {
        return format!("{} B", b);
    };
    if value >= 10.0 {
        format!("{} {}", value as i64, unit)
    } else if (value - value.trunc()).abs() > f64::EPSILON {
        format!("{:.1} {}", value, unit)
    } else {
        format!("{} {}", value as i64, unit)
    }
}

fn format_duration(d: Duration) -> String {
    if d >= Duration::from_secs(360000) {
        "99h+".into()
    } else if d >= Duration::from_secs(3600) {
        format!("{}h{}m", d.as_secs() / 3600, (d.as_secs() / 60) % 60)
    } else {
        format!("{}s", d.as_secs())
    }
}
