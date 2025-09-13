use std::str;

pub fn find_stop<'a>(sequence: &'a str, stops: &[&'a str]) -> Option<&'a str> {
    for stop in stops {
        if sequence.contains(stop) {
            return Some(stop);
        }
    }
    None
}

pub fn contains_stop_suffix(sequence: &str, stops: &[&str]) -> bool {
    for stop in stops {
        for i in 1..=stop.len() {
            if sequence.ends_with(&stop[..i]) {
                return true;
            }
        }
    }
    false
}

pub fn truncate_stop(pieces: &[String], stop: &str) -> (Vec<String>, bool) {
    let joined: String = pieces.join("");
    if let Some(idx) = joined.find(stop) {
        let truncated = &joined[..idx];
        let lengths: Vec<usize> = pieces.iter().map(|p| p.len()).collect();
        let mut result = Vec::new();
        let mut token_truncated = false;
        let mut start = 0usize;
        for len in lengths {
            if start >= truncated.len() {
                break;
            }
            let mut end = start + len;
            if end > truncated.len() {
                end = truncated.len();
                token_truncated = true;
            }
            result.push(truncated[start..end].to_string());
            start = end;
        }
        return (result, token_truncated);
    }
    (pieces.to_vec(), false)
}

pub fn incomplete_unicode(token: &str) -> bool {
    let bytes = token.as_bytes();
    for i in 1..=std::cmp::min(4, bytes.len()) {
        let c = bytes[bytes.len() - i];
        if c & 0b1100_0000 == 0b1000_0000 {
            continue;
        }
        let incomplete = if c & 0b1110_0000 == 0b1100_0000 {
            i < 2
        } else if c & 0b1111_0000 == 0b1110_0000 {
            i < 3
        } else if c & 0b1111_1000 == 0b1111_0000 {
            i < 4
        } else {
            false
        };
        return incomplete;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_stop() {
        struct Case {
            pieces: Vec<&'static str>,
            stop: &'static str,
            expected: Vec<&'static str>,
            trunc: bool,
        }
        let cases = vec![
            Case {
                pieces: vec!["hello", "world"],
                stop: "world",
                expected: vec!["hello"],
                trunc: false,
            },
            Case {
                pieces: vec!["hello", "wor"],
                stop: "or",
                expected: vec!["hello", "w"],
                trunc: true,
            },
            Case {
                pieces: vec!["Hello", " there", "!"],
                stop: "!",
                expected: vec!["Hello", " there"],
                trunc: false,
            },
            Case {
                pieces: vec!["Hello", " the", "re!"],
                stop: "there!",
                expected: vec!["Hello", " "],
                trunc: true,
            },
            Case {
                pieces: vec!["hello", " wor"],
                stop: "llo w",
                expected: vec!["he"],
                trunc: true,
            },
        ];
        for case in cases {
            let (res, trunc) = truncate_stop(
                &case
                    .pieces
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>(),
                case.stop,
            );
            let res_str: Vec<&str> = res.iter().map(|s| s.as_str()).collect();
            assert_eq!(res_str, case.expected);
            assert_eq!(trunc, case.trunc);
        }
    }

    #[test]
    fn test_incomplete_unicode() {
        let cases: Vec<(String, bool)> = vec![
            ("hi".to_string(), false),
            (
                "hi".to_owned() + &String::from_utf8(vec![0xc2, 0xa3]).unwrap(),
                false,
            ),
            (
                {
                    let mut v = b"hi".to_vec();
                    v.push(0xc2);
                    unsafe { String::from_utf8_unchecked(v) }
                },
                true,
            ),
            (
                "hi".to_owned() + &String::from_utf8(vec![0xe0, 0xA0, 0x80]).unwrap(),
                false,
            ),
            (
                {
                    let mut v = b"hi".to_vec();
                    v.extend_from_slice(&[0xe0, 0xA0]);
                    unsafe { String::from_utf8_unchecked(v) }
                },
                true,
            ),
            (
                {
                    let mut v = b"hi".to_vec();
                    v.push(0xe0);
                    unsafe { String::from_utf8_unchecked(v) }
                },
                true,
            ),
            (
                "hi".to_owned() + &String::from_utf8(vec![0xf0, 0x92, 0x8a, 0xb7]).unwrap(),
                false,
            ),
            (
                {
                    let mut v = b"hi".to_vec();
                    v.extend_from_slice(&[0xf0, 0x92, 0x8a]);
                    unsafe { String::from_utf8_unchecked(v) }
                },
                true,
            ),
            (
                {
                    let mut v = b"hi".to_vec();
                    v.extend_from_slice(&[0xf0, 0x92]);
                    unsafe { String::from_utf8_unchecked(v) }
                },
                true,
            ),
            (
                {
                    let mut v = b"hi".to_vec();
                    v.push(0xf0);
                    unsafe { String::from_utf8_unchecked(v) }
                },
                true,
            ),
        ];
        for (input, expect) in cases {
            assert_eq!(incomplete_unicode(&input), expect, "input {}", input);
        }
    }
}
