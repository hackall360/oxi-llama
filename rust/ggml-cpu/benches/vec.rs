use criterion::{criterion_group, criterion_main, Criterion};
use ggml_cpu::vec_dot_f32;
use rand::Rng;

fn random_vec(len: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..len).map(|_| rng.gen()).collect()
}

fn bench_vec_dot(c: &mut Criterion) {
    let a = random_vec(4096);
    let b = random_vec(4096);
    c.bench_function("vec_dot_f32", |bencher| {
        bencher.iter(|| vec_dot_f32(&a, &b));
    });
}

criterion_group!(benches, bench_vec_dot);
criterion_main!(benches);
