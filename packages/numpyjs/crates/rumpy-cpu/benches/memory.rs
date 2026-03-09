//! Benchmarks for memory operations
//!
//! Measures copy overhead vs direct access for array data

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{ArrayD, IxDyn};

/// Generate a random array of given shape
fn random_array(shape: &[usize]) -> ArrayD<f64> {
    use rand::Rng;
    let len: usize = shape.iter().product();
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..len).map(|_| rng.random::<f64>()).collect();
    ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
}

fn bench_data_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_access");

    let sizes = [1_000, 10_000, 100_000, 1_000_000];

    for size in sizes.iter() {
        let arr = random_array(&[*size]);

        // Simulates toTypedArray() - clone the data
        group.bench_with_input(BenchmarkId::new("copy", size), &arr, |bench, arr| {
            bench.iter(|| {
                let copied: Vec<f64> = arr.iter().cloned().collect();
                black_box(copied)
            });
        });

        // Simulates zero-copy - just get a slice reference
        group.bench_with_input(
            BenchmarkId::new("zero_copy_slice", size),
            &arr,
            |bench, arr| {
                bench.iter(|| {
                    // This is what zero-copy does - returns a view, no allocation
                    let slice = arr.as_slice().unwrap();
                    black_box(slice)
                });
            },
        );

        // Ptr + len lookup (what dataPtr() + len() do)
        group.bench_with_input(BenchmarkId::new("ptr_len", size), &arr, |bench, arr| {
            bench.iter(|| {
                let ptr = arr.as_ptr();
                let len = arr.len();
                black_box((ptr, len))
            });
        });
    }

    group.finish();
}

fn bench_copy_vs_view_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_vs_view_ops");

    let sizes = [10_000, 100_000, 1_000_000];

    for size in sizes.iter() {
        let arr = random_array(&[*size]);

        // Sum after copying (simulates: toTypedArray().reduce((a,b) => a+b))
        group.bench_with_input(
            BenchmarkId::new("sum_after_copy", size),
            &arr,
            |bench, arr| {
                bench.iter(|| {
                    let copied: Vec<f64> = arr.iter().cloned().collect();
                    let sum: f64 = copied.iter().sum();
                    black_box(sum)
                });
            },
        );

        // Sum directly (simulates: direct access through view)
        group.bench_with_input(BenchmarkId::new("sum_direct", size), &arr, |bench, arr| {
            bench.iter(|| {
                let sum: f64 = arr.iter().sum();
                black_box(sum)
            });
        });

        // Multiple operations with copy each time
        group.bench_with_input(
            BenchmarkId::new("multi_op_copy", size),
            &arr,
            |bench, arr| {
                bench.iter(|| {
                    let v1: Vec<f64> = arr.iter().cloned().collect();
                    let sum: f64 = v1.iter().sum();
                    let v2: Vec<f64> = arr.iter().cloned().collect();
                    let max = v2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    black_box((sum, max))
                });
            },
        );

        // Multiple operations with single view
        group.bench_with_input(
            BenchmarkId::new("multi_op_view", size),
            &arr,
            |bench, arr| {
                bench.iter(|| {
                    let slice = arr.as_slice().unwrap();
                    let sum: f64 = slice.iter().sum();
                    let max = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    black_box((sum, max))
                });
            },
        );
    }

    group.finish();
}

fn bench_chained_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("chained_ops");

    // Test chained operations staying in Rust/WASM vs crossing boundary each time
    let size = 100_000;
    let a = random_array(&[size]);
    let b = random_array(&[size]);

    // Chained in Rust (simulates: keep data in WASM)
    group.bench_function("chain_in_rust", |bench| {
        bench.iter(|| {
            // All ops stay in ndarray, no JS crossing
            let c = &a + &b;
            let d = &c * 2.0;
            let sum: f64 = d.iter().sum();
            black_box(sum)
        });
    });

    // Simulate crossing boundary each time
    group.bench_function("chain_with_copies", |bench| {
        bench.iter(|| {
            // Add
            let c = &a + &b;
            // Copy out (simulates toTypedArray)
            let _copy1: Vec<f64> = c.iter().cloned().collect();
            // Copy back in (simulates arrayFromTyped)
            let c_back =
                ArrayD::from_shape_vec(IxDyn(&[size]), c.iter().cloned().collect()).unwrap();
            // Multiply
            let d = &c_back * 2.0;
            // Copy out again
            let _copy2: Vec<f64> = d.iter().cloned().collect();
            // Copy back in
            let d_back =
                ArrayD::from_shape_vec(IxDyn(&[size]), d.iter().cloned().collect()).unwrap();
            // Sum
            let sum: f64 = d_back.iter().sum();
            black_box(sum)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_data_access,
    bench_copy_vs_view_operations,
    bench_chained_operations
);
criterion_main!(benches);
