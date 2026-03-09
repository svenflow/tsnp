//! Random number generation for CPU backend

use crate::{CpuArray, CpuBackend};
use ndarray::{ArrayD, IxDyn};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rumpy_core::ops::RandomOps;
use rumpy_core::Array;
use std::cell::RefCell;

thread_local! {
    static RNG: RefCell<SmallRng> = RefCell::new(SmallRng::from_os_rng());
}

impl RandomOps for CpuBackend {
    type Array = CpuArray;

    fn seed(s: u64) {
        RNG.with(|rng| {
            *rng.borrow_mut() = SmallRng::seed_from_u64(s);
        });
    }

    fn rand(shape: Vec<usize>) -> CpuArray {
        let size: usize = shape.iter().product();
        let values: Vec<f64> = RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let dist = Uniform::new(0.0, 1.0).unwrap();
            (0..size).map(|_| dist.sample(&mut *rng)).collect()
        });
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap())
    }

    fn randn(shape: Vec<usize>) -> CpuArray {
        let size: usize = shape.iter().product();
        let values: Vec<f64> = RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let dist = Normal::new(0.0, 1.0).unwrap();
            (0..size).map(|_| dist.sample(&mut *rng)).collect()
        });
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap())
    }

    fn randint(low: i64, high: i64, shape: Vec<usize>) -> CpuArray {
        let size: usize = shape.iter().product();
        let values: Vec<f64> = RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let dist = Uniform::new(low, high).unwrap();
            (0..size).map(|_| dist.sample(&mut *rng) as f64).collect()
        });
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap())
    }

    fn uniform(low: f64, high: f64, shape: Vec<usize>) -> CpuArray {
        let size: usize = shape.iter().product();
        let values: Vec<f64> = RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let dist = Uniform::new(low, high).unwrap();
            (0..size).map(|_| dist.sample(&mut *rng)).collect()
        });
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap())
    }

    fn normal(loc: f64, scale: f64, shape: Vec<usize>) -> CpuArray {
        let size: usize = shape.iter().product();
        let values: Vec<f64> = RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let dist = Normal::new(loc, scale).unwrap();
            (0..size).map(|_| dist.sample(&mut *rng)).collect()
        });
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap())
    }

    fn shuffle(arr: &mut CpuArray) {
        RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            let data = arr.as_ndarray_mut();
            let n = data.len();
            let flat = data.as_slice_mut().unwrap();

            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                flat.swap(i, j);
            }
        });
    }

    fn permutation(n: usize) -> CpuArray {
        let mut values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        RNG.with(|rng| {
            let mut rng = rng.borrow_mut();
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                values.swap(i, j);
            }
        });
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[n]), values).unwrap())
    }

    fn choice(arr: &CpuArray, size: usize, replace: bool) -> CpuArray {
        let data = arr.as_f64_slice();
        let n = data.len();

        // NumPy raises ValueError when size > n and replace=False
        // We panic here (which becomes a JS error in WASM)
        if !replace && size > n {
            panic!(
                "Cannot take a larger sample ({}) than population ({}) when replace=False",
                size, n
            );
        }

        let values: Vec<f64> = RNG.with(|rng| {
            let mut rng = rng.borrow_mut();

            if replace {
                (0..size).map(|_| data[rng.random_range(0..n)]).collect()
            } else {
                // Without replacement using Fisher-Yates partial shuffle - O(size) instead of O(n)
                // Only shuffle the first `size` elements we need
                let mut indices: Vec<usize> = (0..n).collect();

                for i in 0..size {
                    // Swap element at i with a random element from [i, n)
                    let j = rng.random_range(i..n);
                    indices.swap(i, j);
                }

                // Take the first size elements
                indices[..size].iter().map(|&i| data[i]).collect()
            }
        });

        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rumpy_core::Array;

    #[test]
    fn test_rand() {
        CpuBackend::seed(42);
        let arr = CpuBackend::rand(vec![3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.size(), 12);

        // All values should be in [0, 1)
        for &x in arr.as_f64_slice().iter() {
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn test_randn() {
        CpuBackend::seed(42);
        let arr = CpuBackend::randn(vec![1000]);

        // Check mean is approximately 0
        let mean: f64 = arr.as_f64_slice().iter().sum::<f64>() / 1000.0;
        assert!(mean.abs() < 0.1);
    }

    #[test]
    fn test_randint() {
        CpuBackend::seed(42);
        let arr = CpuBackend::randint(0, 10, vec![100]);

        // All values should be in [0, 10)
        for &x in arr.as_f64_slice().iter() {
            assert!((0.0..10.0).contains(&x));
            assert_eq!(x, x.floor()); // Should be integers
        }
    }

    #[test]
    fn test_uniform() {
        CpuBackend::seed(42);
        let arr = CpuBackend::uniform(5.0, 10.0, vec![100]);

        // All values should be in [5, 10)
        for &x in arr.as_f64_slice().iter() {
            assert!((5.0..10.0).contains(&x));
        }
    }

    #[test]
    fn test_normal() {
        CpuBackend::seed(42);
        let arr = CpuBackend::normal(100.0, 15.0, vec![1000]);

        // Check mean is approximately 100
        let mean: f64 = arr.as_f64_slice().iter().sum::<f64>() / 1000.0;
        assert!((mean - 100.0).abs() < 2.0);
    }

    #[test]
    fn test_permutation() {
        CpuBackend::seed(42);
        let perm = CpuBackend::permutation(10);

        // Should contain all integers 0-9
        let mut sorted: Vec<i64> = perm.as_f64_slice().iter().map(|&x| x as i64).collect();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<i64>>());
    }

    #[test]
    fn test_choice() {
        CpuBackend::seed(42);
        let arr = CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();

        let chosen = CpuBackend::choice(&arr, 3, true);
        assert_eq!(chosen.size(), 3);

        // All values should be from the original array
        let original: std::collections::HashSet<i64> =
            arr.as_f64_slice().iter().map(|&x| x as i64).collect();
        for &x in chosen.as_f64_slice().iter() {
            assert!(original.contains(&(x as i64)));
        }
    }

    #[test]
    fn test_seed_reproducibility() {
        CpuBackend::seed(12345);
        let a = CpuBackend::rand(vec![5]);

        CpuBackend::seed(12345);
        let b = CpuBackend::rand(vec![5]);

        assert_eq!(a.as_f64_slice(), b.as_f64_slice());
    }
}
