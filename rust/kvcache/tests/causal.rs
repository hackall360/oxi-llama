use kvcache::{Batch, Causal, DType, Tensor};
use std::f32;

struct TestCase {
    input: Vec<f32>,
    shape: [usize;3],
    pos: Vec<i32>,
    expected: Vec<f32>,
    expected_shape: [usize;3],
    expected_mask: Vec<f32>,
}

fn run(cache: &mut Causal, cases: Vec<TestCase>) {
    for c in cases {
        cache.start_forward(Batch { positions: c.pos.clone() }, false);
        cache.set_layer(0);
        let t = Tensor::from_slice(&c.input, &c.shape);
        cache.put(&t, &t);
        let (out, _v, mask) = cache.get();
        assert_eq!(out.floats(), c.expected);
        assert_eq!(out.shape(), c.expected_shape);
        assert_eq!(mask.floats(), c.expected_mask);
    }
}

#[test]
fn test_store() {
    let mut cache = Causal::new();
    cache.init(DType::F16, 1, 16, 16);
    run(
        &mut cache,
        vec![
            TestCase {
                input: vec![
                    111., 211., 121., 221., 131., 231., 112., 212., 122., 222., 132., 232., 113., 213., 123., 223., 133., 233.,
                    114., 214., 124., 224., 134., 234.,
                ],
                shape: [2, 3, 4],
                pos: vec![0, 1, 2, 3],
                expected: vec![
                    111., 211., 121., 221., 131., 231., 112., 212., 122., 222., 132., 232., 113., 213., 123., 223., 133., 233.,
                    114., 214., 124., 224., 134., 234.,
                ],
                expected_shape: [2, 3, 4],
                expected_mask: vec![
                    0., f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY,
                    0., 0., f32::NEG_INFINITY, f32::NEG_INFINITY,
                    0., 0., 0., f32::NEG_INFINITY,
                    0., 0., 0., 0.,
                ],
            },
            TestCase {
                input: vec![115., 215., 125., 225., 135., 235.],
                shape: [2, 3, 1],
                pos: vec![4],
                expected: vec![
                    111., 211., 121., 221., 131., 231., 112., 212., 122., 222., 132., 232., 113., 213., 123., 223., 133., 233.,
                    114., 214., 124., 224., 134., 234., 115., 215., 125., 225., 135., 235.,
                ],
                expected_shape: [2, 3, 5],
                expected_mask: vec![0., 0., 0., 0., 0.],
            },
        ],
    );
}

#[test]
fn test_swa() {
    let mut cache = Causal::new_swa(1);
    cache.init(DType::F16, 1, 16, 16);
    let x = f32::NEG_INFINITY;
    run(
        &mut cache,
        vec![
            TestCase {
                input: vec![1., 2., 3., 4.],
                shape: [1, 1, 4],
                pos: vec![0, 1, 2, 3],
                expected: vec![1., 2., 3., 4.],
                expected_shape: [1, 1, 4],
                expected_mask: vec![
                    0., x, x, x,
                    0., 0., x, x,
                    x, 0., 0., x,
                    x, x, 0., 0.,
                ],
            },
            TestCase {
                input: vec![5., 6.],
                shape: [1, 1, 2],
                pos: vec![4, 5],
                expected: vec![1., 2., 3., 4., 5., 6.],
                expected_shape: [1, 1, 6],
                expected_mask: vec![
                    x, x, x, 0., 0., x,
                    x, x, x, x, 0., 0.,
                ],
            },
        ],
    );
}

#[test]
fn test_swa_mem() {
    let mut cache = Causal::new_swa_mem(1, 3);
    cache.init(DType::F16, 1, 16, 16);
    let x = f32::NEG_INFINITY;
    run(
        &mut cache,
        vec![
            TestCase {
                input: vec![1., 2., 3., 4.],
                shape: [1, 1, 4],
                pos: vec![0, 1, 2, 3],
                expected: vec![1., 2., 3., 4.],
                expected_shape: [1, 1, 4],
                expected_mask: vec![
                    0., x, x, x,
                    0., 0., x, x,
                    x, 0., 0., x,
                    x, x, 0., 0.,
                ],
            },
            TestCase {
                input: vec![5., 6.],
                shape: [1, 1, 2],
                pos: vec![4, 5],
                expected: vec![4., 5., 6.],
                expected_shape: [1, 1, 3],
                expected_mask: vec![
                    0., 0., x,
                    x, 0., 0.,
                ],
            },
        ],
    );
}

#[test]
fn test_chunked() {
    let mut cache = Causal::new_chunked(2);
    cache.init(DType::F16, 1, 16, 16);
    let x = f32::NEG_INFINITY;
    run(
        &mut cache,
        vec![
            TestCase {
                input: vec![1., 2., 3., 4.],
                shape: [1, 1, 4],
                pos: vec![0, 1, 2, 3],
                expected: vec![1., 2., 3., 4.],
                expected_shape: [1, 1, 4],
                expected_mask: vec![
                    0., x, x, x,
                    0., 0., x, x,
                    x, x, 0., x,
                    x, x, 0., 0.,
                ],
            },
            TestCase {
                input: vec![5., 6., 7.],
                shape: [1, 1, 3],
                pos: vec![4, 5, 6],
                expected: vec![1., 2., 3., 4., 5., 6., 7.],
                expected_shape: [1, 1, 7],
                expected_mask: vec![
                    x, x, x, x, 0., x, x,
                    x, x, x, x, 0., 0., x,
                    x, x, x, x, x, x, 0.,
                ],
            },
            TestCase {
                input: vec![8., 9.],
                shape: [1, 1, 2],
                pos: vec![7, 8],
                expected: vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
                expected_shape: [1, 1, 9],
                expected_mask: vec![
                    x, x, x, x, x, x, 0., 0., x,
                    x, x, x, x, x, x, x, x, 0.,
                ],
            },
        ],
    );
}

