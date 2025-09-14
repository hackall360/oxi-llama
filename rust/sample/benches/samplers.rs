use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use sample::Sampler;

fn bench_weighted_sampler(c: &mut Criterion) {
    let sizes = [10usize, 100, 1000, 10000];
    for &size in &sizes {
        let mut logits = vec![0f32; size];
        let mut rng = rand::thread_rng();
        for v in &mut logits { *v = rng.r#gen::<f32>() * 10.0 - 5.0; }
        let mut sampler = Sampler::new(0.8, 0, 0.0, 0.0, 42);
        c.bench_with_input(BenchmarkId::new("weighted_size", size), &size, |b, &_s| {
            b.iter(|| sampler.sample(&logits).unwrap());
        });
    }

    let configs = [
        ("Greedy", 0.0, -1, 0.0, 0.0, -1),
        ("Temperature", 0.8, -1, 0.0, 0.0, -1),
        ("TopK", 0.8, 50, 0.0, 0.0, -1),
        ("TopP", 0.8, -1, 0.9, 0.0, -1),
        ("MinP", 0.8, -1, 0.0, 0.05, -1),
        ("WithSeed", 0.8, 50, 0.0, 0.0, 42),
    ];

    let mut logits = vec![0f32; 128_000];
    let mut rng = rand::thread_rng();
    for v in &mut logits { *v = rng.r#gen::<f32>() * 10.0 - 5.0; }

    for &(name, temp, top_k, top_p, min_p, seed) in &configs {
        let mut sampler = Sampler::new(temp, top_k, top_p, min_p, seed);
        sampler.sample(&logits).unwrap();
        c.bench_function(&format!("weighted_{}", name), |b| {
            b.iter(|| sampler.sample(&logits).unwrap());
        });
    }

    let mut sampler = Sampler::new(0.8, 50, 0.9, 0.05, 42);
    c.bench_function("weighted_combined", |b| {
        b.iter(|| sampler.sample(&logits).unwrap());
    });
}

fn bench_greedy_sampler(c: &mut Criterion) {
    let sizes = [10usize, 100, 1000, 10000, 100000];
    for &size in &sizes {
        let mut logits = vec![0f32; size];
        let mut rng = rand::thread_rng();
        for v in &mut logits { *v = rng.r#gen::<f32>() * 10.0 - 5.0; }
        let mut sampler = Sampler::new(0.0, -1, 0.0, 0.0, -1);
        c.bench_with_input(BenchmarkId::new("greedy_size", size), &size, |b, &_s| {
            b.iter(|| sampler.sample(&logits).unwrap());
        });
    }
}

criterion_group!(benches, bench_weighted_sampler, bench_greedy_sampler);
criterion_main!(benches);
