//! Benchmarks for broadcasting operations
//!
//! Compares broadcasting vs manual shape alignment for various array sizes

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{ArrayD, IxDyn};
use rumpy_cpu::broadcast_binary_op; // Use public re-export

/// Generate a random array of given shape
fn random_array(shape: &[usize]) -> ArrayD<f64> {
    use rand::Rng;
    let len: usize = shape.iter().product();
    let mut rng = rand::rng();
    let data: Vec<f64> = (0..len).map(|_| rng.random::<f64>()).collect();
    ArrayD::from_shape_vec(IxDyn(shape), data).unwrap()
}

fn bench_broadcasting_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting_add");

    // Different array sizes to test
    let sizes = [
        (vec![100, 100], vec![100]),       // Matrix + row vector
        (vec![1000, 1000], vec![1000]),    // Large matrix + row vector
        (vec![100, 100], vec![100, 1]),    // Matrix + column vector
        (vec![1000, 1000], vec![1000, 1]), // Large matrix + column vector
        (vec![100, 1], vec![1, 100]),      // Outer product style
        (vec![1000, 1], vec![1, 1000]),    // Large outer product style
    ];

    for (shape_a, shape_b) in sizes.iter() {
        let a = random_array(shape_a);
        let b = random_array(shape_b);

        let id = format!("{:?}_+_{:?}", shape_a, shape_b);
        group.bench_with_input(
            BenchmarkId::new("broadcast", &id),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| black_box(broadcast_binary_op(a, b, |x, y| x + y).unwrap()));
            },
        );

        // Compare with manual broadcast (pre-expand arrays)
        let output_shape: Vec<usize> = {
            let ndim = shape_a.len().max(shape_b.len());
            let mut result = vec![0; ndim];
            for i in 0..ndim {
                let dim_a = if i < shape_a.len() {
                    shape_a[shape_a.len() - 1 - i]
                } else {
                    1
                };
                let dim_b = if i < shape_b.len() {
                    shape_b[shape_b.len() - 1 - i]
                } else {
                    1
                };
                result[ndim - 1 - i] = dim_a.max(dim_b);
            }
            result
        };

        // Pre-materialize both arrays to target shape
        let a_expanded = a.broadcast(IxDyn(&output_shape)).unwrap().to_owned();
        let b_expanded = b.broadcast(IxDyn(&output_shape)).unwrap().to_owned();

        group.bench_with_input(
            BenchmarkId::new("manual_expand", &id),
            &(a_expanded, b_expanded),
            |bench, (a, b)| {
                bench.iter(|| black_box(a + b));
            },
        );
    }

    group.finish();
}

fn bench_broadcasting_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcasting_chain");

    // Test chained operations with broadcasting
    let shape_matrix = vec![1000, 1000];
    let shape_row = vec![1000];
    let shape_col = vec![1000, 1];

    let matrix = random_array(&shape_matrix);
    let row = random_array(&shape_row);
    let col = random_array(&shape_col);

    // Chained broadcast: matrix + row * col
    group.bench_function("chain_broadcast", |bench| {
        bench.iter(|| {
            let temp = broadcast_binary_op(&row, &col, |x, y| x * y).unwrap();
            black_box(broadcast_binary_op(&matrix, &temp, |x, y| x + y).unwrap())
        });
    });

    // Same but with pre-expanded arrays
    let row_expanded = row.broadcast(IxDyn(&shape_matrix)).unwrap().to_owned();
    let col_expanded = col.broadcast(IxDyn(&shape_matrix)).unwrap().to_owned();

    group.bench_function("chain_preexpand", |bench| {
        bench.iter(|| {
            let temp = &row_expanded * &col_expanded;
            black_box(&matrix + &temp)
        });
    });

    group.finish();
}

fn bench_same_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("same_shape_ops");

    // Test that same-shape fast path is truly fast
    let sizes = [100, 500, 1000, 2000];

    for size in sizes.iter() {
        let shape = vec![*size, *size];
        let a = random_array(&shape);
        let b = random_array(&shape);

        group.bench_with_input(
            BenchmarkId::new("broadcast_fn", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| black_box(broadcast_binary_op(a, b, |x, y| x + y).unwrap()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("direct_add", size),
            &(a.clone(), b.clone()),
            |bench, (a, b)| {
                bench.iter(|| black_box(a + b));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_broadcasting_add,
    bench_broadcasting_chain,
    bench_same_shape
);
criterion_main!(benches);
