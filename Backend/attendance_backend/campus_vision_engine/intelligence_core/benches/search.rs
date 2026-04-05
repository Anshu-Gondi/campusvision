use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use intelligence_core::embeddings::{
    add_face_embedding,
    search_in_role,
    batch_search,
    get_total_faces,
    count_by_role,
    clear_all,
};
use intelligence_core::utils::make_vec;

// ── setup helpers ─────────────────────────────────────────────────────────────

fn writer() {
    unsafe { std::env::set_var("FACE_DB_ROLE", "writer"); }
}

fn populate(school: &str, n_students: usize, n_teachers: usize) {
    for i in 0..n_students {
        let emb = make_vec(i as f32 * 0.013 + 0.1, 128);
        add_face_embedding(
            school, emb,
            format!("Student_{}", i),
            i as u64,
            format!("R{:04}", i),
            "student".into(),
        ).unwrap();
    }
    for i in 0..n_teachers {
        let emb = make_vec(i as f32 * 0.031 + 5.0, 128);
        add_face_embedding(
            school, emb,
            format!("Teacher_{}", i),
            (10_000 + i) as u64,
            format!("E{:04}", i),
            "teacher".into(),
        ).unwrap();
    }
}

// ── HNSW search latency at realistic class sizes ──────────────────────────────

fn bench_search_latency(c: &mut Criterion) {
    writer();
    let mut group = c.benchmark_group("search_latency");

    // realistic classroom sizes for CampusVision deployment
    for n_students in [50, 100, 250, 500, 1000] {
        let school = format!("bench_search_{}", n_students);
        clear_all();
        populate(&school, n_students, 10);

        let query = make_vec(99.9, 128);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("search_in_role_student", n_students),
            &n_students,
            |bench, _| {
                bench.iter(|| {
                    black_box(search_in_role(
                        black_box(&school),
                        black_box(&query),
                        "student",
                        black_box(5),
                    ))
                })
            },
        );
    }

    group.finish();
}

// ── batch search — multiple frames at once (CCTV use case) ───────────────────

fn bench_batch_search(c: &mut Criterion) {
    writer();
    let mut group = c.benchmark_group("batch_search");

    let school = "bench_batch_search_school";
    clear_all();
    populate(school, 500, 20);

    // simulate 1, 5, 10 camera frames processed together
    for batch_size in [1usize, 5, 10] {
        let queries: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| make_vec(i as f32 * 0.77 + 1.0, 128))
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_student", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| {
                    black_box(batch_search(
                        black_box(school),
                        black_box(&queries),
                        "student",
                        black_box(5),
                    ))
                })
            },
        );
    }

    group.finish();
}

// ── add_face_embedding throughput — enrollment speed ─────────────────────────

fn bench_insert_throughput(c: &mut Criterion) {
    writer();
    let mut group = c.benchmark_group("insert_throughput");
    group.throughput(Throughput::Elements(1));

    // fresh school per bench run to avoid index getting huge
    let mut counter = 0usize;

    group.bench_function("add_student_128d", |bench| {
        bench.iter(|| {
            let school = format!("bench_insert_{}", counter % 10);
            let emb = make_vec(counter as f32 * 0.001 + 0.1, 128);
            black_box(
                add_face_embedding(
                    black_box(&school),
                    black_box(emb),
                    "Bench".into(),
                    counter as u64,
                    format!("R{}", counter),
                    "student".into(),
                ).unwrap()
            );
            counter += 1;
        })
    });

    group.finish();
}

// ── count_by_role — how fast is the hot metadata scan ────────────────────────

fn bench_metadata_scan(c: &mut Criterion) {
    writer();
    let mut group = c.benchmark_group("metadata_scan");

    for n in [100, 500, 1000, 5000] {
        let school = format!("bench_meta_scan_{}", n);
        clear_all();
        populate(&school, n, n / 10);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("get_total_faces", n),
            &n,
            |bench, _| {
                bench.iter(|| black_box(get_total_faces(black_box(&school))))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("count_by_role_student", n),
            &n,
            |bench, _| {
                bench.iter(|| black_box(count_by_role(black_box(&school), "student")))
            },
        );
    }

    group.finish();
}

// ── search vs batch_search — which is faster for N queries ───────────────────

fn bench_search_vs_batch(c: &mut Criterion) {
    writer();
    let mut group = c.benchmark_group("search_vs_batch");

    let school = "bench_search_vs_batch_school";
    clear_all();
    populate(school, 300, 10);

    let n_queries = 5;
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| make_vec(i as f32 * 1.1 + 0.5, 128))
        .collect();

    group.throughput(Throughput::Elements(n_queries as u64));

    // N sequential single searches
    group.bench_function("5x_search_in_role", |bench| {
        bench.iter(|| {
            for q in &queries {
                black_box(search_in_role(school, black_box(q), "student", 5));
            }
        })
    });

    // 1 batch search of N queries
    group.bench_function("1x_batch_search_5", |bench| {
        bench.iter(|| {
            black_box(batch_search(school, black_box(&queries), "student", 5))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_search_latency,
    bench_batch_search,
    bench_insert_throughput,
    bench_metadata_scan,
    bench_search_vs_batch,
);
criterion_main!(benches);