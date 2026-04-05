use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use intelligence_core::utils::{
    cosine_similarity,
    cosine_similarity_scalar,
    normalize,
    make_vec,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn rand_vec(seed: f32, len: usize) -> Vec<f32> {
    make_vec(seed, len)
}

// ── cosine similarity: SIMD vs scalar ────────────────────────────────────────

fn bench_cosine_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for len in [32, 64, 128, 256, 512] {
        let a = rand_vec(1.3, len);
        let b = rand_vec(2.7, len);

        group.throughput(Throughput::Elements(len as u64));

        group.bench_with_input(
            BenchmarkId::new("simd_dispatched", len),
            &len,
            |bench, _| {
                bench.iter(|| {
                    black_box(cosine_similarity(
                        black_box(&a),
                        black_box(&b),
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", len),
            &len,
            |bench, _| {
                bench.iter(|| {
                    black_box(cosine_similarity_scalar(
                        black_box(&a),
                        black_box(&b),
                    ))
                })
            },
        );
    }

    group.finish();
}

// ── cosine: batch of queries (realistic face recognition workload) ────────────

fn bench_cosine_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_batch");

    // 128-dim is your ArcFace embedding size
    let len = 128;
    let queries: Vec<Vec<f32>> = (0..30)
        .map(|i| rand_vec(i as f32 * 0.17 + 0.1, len))
        .collect();
    let db_vec = rand_vec(3.14, len);

    // simulates scanning a class of 30 students
    group.throughput(Throughput::Elements(30));

    group.bench_function("30_queries_simd", |bench| {
        bench.iter(|| {
            for q in &queries {
                black_box(cosine_similarity(black_box(q), black_box(&db_vec)));
            }
        })
    });

    group.bench_function("30_queries_scalar", |bench| {
        bench.iter(|| {
            for q in &queries {
                black_box(cosine_similarity_scalar(black_box(q), black_box(&db_vec)));
            }
        })
    });

    group.finish();
}

// ── normalize ─────────────────────────────────────────────────────────────────

fn bench_normalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize");

    for len in [32, 128, 256, 512] {
        group.throughput(Throughput::Elements(len as u64));

        group.bench_with_input(
            BenchmarkId::new("normalize", len),
            &len,
            |bench, &len| {
                let original = rand_vec(2.22, len);
                bench.iter(|| {
                    // clone each time — we're measuring normalize itself
                    let mut v = original.clone();
                    normalize(black_box(&mut v));
                    black_box(v)
                })
            },
        );
    }

    group.finish();
}

// ── normalize then cosine — the real hot path in add_face_embedding ──────────

fn bench_normalize_then_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_then_cosine");
    group.throughput(Throughput::Elements(1));

    let raw = rand_vec(1.11, 128);
    let reference = {
        let mut v = rand_vec(2.22, 128);
        normalize(&mut v);
        v
    };

    group.bench_function("full_pipeline_128d", |bench| {
        bench.iter(|| {
            let mut v = raw.clone();
            normalize(black_box(&mut v));
            black_box(cosine_similarity(black_box(&v), black_box(&reference)))
        })
    });

    group.finish();
}

// ── tail handling — non-multiple-of-4 lengths ─────────────────────────────────

fn bench_tail_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tail_handling");

    // these lengths exercise different tail sizes (0, 1, 2, 3 remainder)
    for len in [128, 129, 130, 131] {
        let a = rand_vec(1.0, len);
        let b = rand_vec(2.0, len);

        group.bench_with_input(
            BenchmarkId::new("simd", len),
            &len,
            |bench, _| {
                bench.iter(|| black_box(cosine_similarity(black_box(&a), black_box(&b))))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_simd_vs_scalar,
    bench_cosine_batch,
    bench_normalize,
    bench_normalize_then_cosine,
    bench_tail_handling,
);
criterion_main!(benches);