use ggml_core::{ComputationGraph, Context};

#[test]
fn graph_produces_topological_order() {
    let ctx = Context::builder().build();
    let a = ctx.tensor_from_f32(&[2], &[1.0, 2.0]).unwrap();
    let b = ctx.tensor_from_f32(&[2], &[3.0, 4.0]).unwrap();
    let sum = a.add(&b).unwrap();
    let prod = sum.mul(&b).unwrap();

    let mut graph = ComputationGraph::new(ctx.clone());
    graph.add(&prod);
    let executor = graph.compile().unwrap();

    let order = executor.order();
    assert_eq!(order, &[a.id(), b.id(), sum.id(), prod.id()]);
}
