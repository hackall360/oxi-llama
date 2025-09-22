use approx::assert_abs_diff_eq;
use ggml_core::{ComputationGraph, Context, Error};

#[test]
fn executes_elementwise_add_mul() {
    let ctx = Context::builder().memory_size(4096).build();
    let a = ctx.tensor_from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = ctx.tensor_from_f32(&[4], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let sum = a.add(&b).unwrap();
    let prod = sum.mul(&b).unwrap();

    let mut graph = ComputationGraph::new(ctx.clone());
    graph.add(&prod);
    graph.compute().unwrap();

    assert_eq!(sum.to_vec().unwrap(), vec![6.0, 8.0, 10.0, 12.0]);
    assert_eq!(prod.to_vec().unwrap(), vec![30.0, 48.0, 70.0, 96.0]);
}

#[test]
fn matmul_matches_reference() {
    let ctx = Context::builder().memory_size(16 * 1024).build();
    let lhs = ctx
        .tensor_from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let rhs = ctx
        .tensor_from_f32(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        .unwrap();
    let product = lhs.matmul(&rhs).unwrap();
    product.graph().compute().unwrap();

    let result = product.to_vec().unwrap();
    assert_eq!(result.len(), 4);
    assert_abs_diff_eq!(result[0], 58.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result[1], 64.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result[2], 139.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result[3], 154.0, epsilon = 1e-6);
}

#[test]
fn unary_relu_and_neg() {
    let ctx = Context::builder().build();
    let input = ctx.tensor_from_f32(&[3], &[-1.0, 0.5, -3.0]).unwrap();
    let relu = input.apply_unary(ggml_core::UnaryOpKind::Relu).unwrap();
    let neg = input.apply_unary(ggml_core::UnaryOpKind::Neg).unwrap();

    let mut graph = ComputationGraph::new(ctx.clone());
    graph.add(&relu);
    graph.add(&neg);
    graph.compute().unwrap();

    assert_eq!(relu.to_vec().unwrap(), vec![0.0, 0.5, 0.0]);
    assert_eq!(neg.to_vec().unwrap(), vec![1.0, -0.5, 3.0]);
}

#[test]
fn context_mismatch_is_reported() {
    let a_ctx = Context::builder().build();
    let b_ctx = Context::builder().build();
    let a = a_ctx.tensor_from_f32(&[2], &[1.0, 2.0]).unwrap();
    let b = b_ctx.tensor_from_f32(&[2], &[3.0, 4.0]).unwrap();
    let err = a.add(&b).unwrap_err();
    assert!(matches!(err, Error::ContextMismatch));
}
