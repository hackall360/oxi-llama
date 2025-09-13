pub use crate::llamarunner::cache::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvCache;
    use std::time::{Duration, SystemTime};

    #[test]
    fn test_count_common() {
        let cases = vec![
            (
                vec![Input::Token(1), Input::Token(2), Input::Token(3)],
                vec![Input::Token(1), Input::Token(2), Input::Token(3)],
                3usize,
            ),
            (
                vec![Input::Token(1)],
                vec![Input::Token(1), Input::Token(2), Input::Token(3)],
                1,
            ),
            (
                vec![Input::Multimodal(1)],
                vec![
                    Input::Multimodal(1),
                    Input::Multimodal(2),
                    Input::Multimodal(3),
                ],
                1,
            ),
            (
                vec![Input::Token(1), Input::Multimodal(1)],
                vec![Input::Token(1), Input::Multimodal(1), Input::Token(5)],
                2,
            ),
            (
                vec![Input::Token(1), Input::Multimodal(1)],
                vec![Input::Token(1), Input::Multimodal(2)],
                1,
            ),
            (vec![], vec![Input::Token(1)], 0),
            (vec![], vec![], 0),
        ];
        for (a, b, expected) in cases {
            assert_eq!(InputCache::count_common_prefix(&a, &b), expected);
        }
    }

    #[test]
    fn test_load_cache_slot() {
        let now = SystemTime::now();
        let mut cache = InputCache {
            num_ctx: 10,
            slots: vec![
                InputCacheSlot {
                    id: 0,
                    inputs: vec![],
                    in_use: false,
                    last_used: now - Duration::from_secs(2),
                },
                InputCacheSlot {
                    id: 1,
                    inputs: vec![Input::Token(1)],
                    in_use: false,
                    last_used: now - Duration::from_secs(1),
                },
            ],
            multi_user_cache: false,
            cache: None,
        };
        let prompt = vec![Input::Token(1), Input::Token(2)];
        let (slot_idx, remaining) = cache.load_cache_slot(prompt, true).unwrap();
        assert_eq!(slot_idx, 1);
        assert_eq!(remaining.len(), 1); // one token remaining
    }

    // Mock cache backend implementing KvCache
    struct MockCache {
        should_fail: bool,
    }
    impl KvCache for MockCache {
        fn remove(
            &mut self,
            _slot: usize,
            _begin: i32,
            _end: i32,
        ) -> Result<(), Box<dyn std::error::Error>> {
            if self.should_fail {
                Err("mock".into())
            } else {
                Ok(())
            }
        }
        fn copy_prefix(&mut self, _src: usize, _dst: usize, _len: i32) {}
        fn can_shift(&self) -> bool {
            true
        }
        fn shift(&mut self, _slot: usize, _start: i32, _end: i32, _delta: i32) {}
    }

    #[test]
    fn test_shift_cache_slot() {
        let tests = vec![("normal", false, None), ("cache_error", true, Some("mock"))];
        for (name, fail, expected_err) in tests {
            let mut cache = InputCache {
                num_ctx: 10,
                slots: vec![InputCacheSlot {
                    id: 0,
                    inputs: (1..=10).map(|i| Input::Token(i)).collect(),
                    in_use: false,
                    last_used: SystemTime::now(),
                }],
                multi_user_cache: false,
                cache: Some(Box::new(MockCache { should_fail: fail })),
            };
            let res = cache.shift_cache_slot(0, 2);
            match expected_err {
                None => assert!(res.is_ok(), "{}", name),
                Some(_) => assert!(res.is_err(), "{}", name),
            }
        }
    }
}
