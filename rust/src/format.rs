use std::time::{Duration, SystemTime};

fn decimal_separator() -> char {
    std::env::var("LC_NUMERIC")
        .or_else(|_| std::env::var("LANG"))
        .ok()
        .and_then(|locale| {
            let l = locale.to_lowercase();
            if l.starts_with("fr")
                || l.starts_with("de")
                || l.starts_with("es")
                || l.starts_with("it")
                || l.starts_with("pt")
                || l.starts_with("ru")
                || l.starts_with("pl")
                || l.starts_with("nl")
                || l.starts_with("tr")
            {
                Some(',')
            } else {
                Some('.')
            }
        })
        .unwrap_or('.')
}

pub fn format_bytes(b: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;

    let sep = decimal_separator();

    let fmt = |value: f64, unit: &str| -> String {
        let mut s = format!("{:.1}", value);
        if sep != '.' {
            s = s.replace('.', &sep.to_string());
        }
        format!("{} {}", s, unit)
    };

    if b >= GIB {
        fmt(b as f64 / GIB as f64, "GiB")
    } else if b >= MIB {
        fmt(b as f64 / MIB as f64, "MiB")
    } else if b >= KIB {
        fmt(b as f64 / KIB as f64, "KiB")
    } else {
        format!("{} B", b)
    }
}

pub fn format_time(t: Option<SystemTime>, zero_value: &str) -> String {
    match t {
        None => zero_value.to_string(),
        Some(ts) => {
            let now = SystemTime::now();
            match now.duration_since(ts) {
                Ok(delta) => format!("{} ago", human_duration(delta)),
                Err(e) => {
                    let delta = e.duration();
                    if delta.as_secs() / (60 * 60 * 24 * 365) > 20 {
                        "Forever".to_string()
                    } else {
                        format!("{} from now", human_duration(delta))
                    }
                }
            }
        }
    }
}

fn human_duration(d: Duration) -> String {
    let seconds = d.as_secs();
    if seconds < 1 {
        return "Less than a second".to_string();
    } else if seconds == 1 {
        return "1 second".to_string();
    } else if seconds < 60 {
        return format!("{} seconds", seconds);
    }

    let minutes = seconds / 60;
    if minutes == 1 {
        return "About a minute".to_string();
    } else if minutes < 60 {
        return format!("{} minutes", minutes);
    }

    let hours = (d.as_secs_f64() / 3600.0).round() as u64;
    if hours == 1 {
        return "About an hour".to_string();
    } else if hours < 48 {
        return format!("{} hours", hours);
    } else if hours < 24 * 7 * 2 {
        return format!("{} days", hours / 24);
    } else if hours < 24 * 30 * 2 {
        return format!("{} weeks", hours / 24 / 7);
    } else if hours < 24 * 365 * 2 {
        return format!("{} months", hours / 24 / 30);
    }

    format!("{} years", seconds / 3600 / 24 / 365)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        struct TestCase {
            input: u64,
            expected: &'static str,
        }
        let tests = vec![
            TestCase {
                input: 0,
                expected: "0 B",
            },
            TestCase {
                input: 1,
                expected: "1 B",
            },
            TestCase {
                input: 1023,
                expected: "1023 B",
            },
            TestCase {
                input: 1024,
                expected: "1.0 KiB",
            },
            TestCase {
                input: 1536,
                expected: "1.5 KiB",
            },
            TestCase {
                input: 1048575,
                expected: "1024.0 KiB",
            },
            TestCase {
                input: 1048576,
                expected: "1.0 MiB",
            },
            TestCase {
                input: 1572864,
                expected: "1.5 MiB",
            },
            TestCase {
                input: 1073741823,
                expected: "1024.0 MiB",
            },
            TestCase {
                input: 1073741824,
                expected: "1.0 GiB",
            },
            TestCase {
                input: 1610612736,
                expected: "1.5 GiB",
            },
            TestCase {
                input: 2147483648,
                expected: "2.0 GiB",
            },
        ];

        for tc in tests {
            assert_eq!(format_bytes(tc.input), tc.expected, "input {}", tc.input);
        }
    }

    #[test]
    fn test_format_time() {
        let now = SystemTime::now();
        assert_eq!(format_time(None, "never"), "never");

        let v = now + Duration::from_secs(48 * 3600);
        assert_eq!(format_time(Some(v), ""), "2 days from now");

        let v = now - Duration::from_secs(48 * 3600);
        assert_eq!(format_time(Some(v), ""), "2 days ago");

        let v = now + Duration::from_millis(800);
        assert_eq!(format_time(Some(v), ""), "Less than a second from now");

        let v = now + Duration::from_secs(24 * 3600 * 365 * 200);
        assert_eq!(format_time(Some(v), ""), "Forever");

        let v = now + Duration::from_secs(24 * 3600 * 365 * 200);
        assert_eq!(format_time(Some(v), "").to_lowercase(), "forever");
    }
}
