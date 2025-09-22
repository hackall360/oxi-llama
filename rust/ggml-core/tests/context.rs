use ggml_core::{Context, ContextBuilder, Error};

#[test]
fn context_initializes_arena() {
    let ctx = Context::builder().memory_size(1024).build();
    let tensor = ctx.tensor_from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(tensor.shape().as_slice(), &[2, 2]);
    assert_eq!(tensor.dtype(), ggml_core::DType::F32);
    assert_eq!(tensor.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn parameter_requires_initialization() {
    let ctx = Context::builder().memory_size(4096).build();
    let weight = ctx.parameter(&[2, 2]).unwrap();
    let graph = weight.graph();
    let err = graph.compute().unwrap_err();
    assert!(matches!(err, Error::UninitializedTensor(_)));

    weight.set_f32(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    graph.compute().unwrap();
    assert_eq!(weight.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn arena_out_of_memory() {
    let ctx = ContextBuilder::default().memory_size(64).build();
    ctx.tensor_from_f32(&[2, 2], &[0.0, 0.0, 0.0, 0.0]).unwrap();
    let err = ctx
        .tensor_from_f32(&[8, 8], &[0.0f32; 64])
        .expect_err("should exceed arena");
    assert!(matches!(err, Error::OutOfMemory { .. }));
}
