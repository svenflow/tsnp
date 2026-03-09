//! Linear algebra tests - NumPy compatible

#[cfg(test)]
mod tests {
    use crate::utils::*;
    use rumpy_core::{
        ops::{CreationOps, LinalgOps},
        Array,
    };
    use rumpy_cpu::{CpuArray, CpuBackend};

    fn mat(data: Vec<f64>, rows: usize, cols: usize) -> CpuArray {
        CpuArray::from_f64_vec(data, vec![rows, cols]).unwrap()
    }

    fn vec1d(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    // ============ matmul ============

    #[test]
    fn test_matmul_2x2() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = CpuBackend::matmul(&a, &b).unwrap();

        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(c.as_f64_slice(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = mat(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
        let c = CpuBackend::matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]]
        // = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        assert_eq!(c.as_f64_slice(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        assert!(CpuBackend::matmul(&a, &b).is_err());
    }

    // ============ dot ============

    #[test]
    fn test_dot_vectors() {
        let a = vec1d(vec![1.0, 2.0, 3.0]);
        let b = vec1d(vec![4.0, 5.0, 6.0]);
        let c = CpuBackend::dot(&a, &b).unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(c.as_f64_slice()[0], 32.0);
    }

    #[test]
    fn test_dot_matrices() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = CpuBackend::dot(&a, &b).unwrap();
        // Same as matmul for 2D
        assert_eq!(c.as_f64_slice(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    // ============ inner ============

    #[test]
    fn test_inner() {
        let a = vec1d(vec![1.0, 2.0, 3.0]);
        let b = vec1d(vec![4.0, 5.0, 6.0]);
        let result = CpuBackend::inner(&a, &b).unwrap();
        assert_eq!(result, 32.0);
    }

    // ============ outer ============

    #[test]
    fn test_outer() {
        let a = vec1d(vec![1.0, 2.0]);
        let b = vec1d(vec![3.0, 4.0, 5.0]);
        let c = CpuBackend::outer(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.as_f64_slice(), vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    // ============ inv ============

    #[test]
    fn test_inv_2x2() {
        let a = mat(vec![4.0, 7.0, 2.0, 6.0], 2, 2);
        let a_inv = CpuBackend::inv(&a).unwrap();

        // A @ A^-1 should be identity
        let identity = CpuBackend::matmul(&a, &a_inv).unwrap();
        let data = identity.as_f64_slice();
        assert!(approx_eq(data[0], 1.0, RELAXED_TOL));
        assert!(approx_eq(data[1], 0.0, RELAXED_TOL));
        assert!(approx_eq(data[2], 0.0, RELAXED_TOL));
        assert!(approx_eq(data[3], 1.0, RELAXED_TOL));
    }

    #[test]
    fn test_inv_3x3() {
        let a = mat(vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0], 3, 3);
        let a_inv = CpuBackend::inv(&a).unwrap();

        // A @ A^-1 should be identity
        let identity = CpuBackend::matmul(&a, &a_inv).unwrap();
        let data = identity.as_f64_slice();
        assert!(approx_eq(data[0], 1.0, RELAXED_TOL));
        assert!(approx_eq(data[4], 1.0, RELAXED_TOL));
        assert!(approx_eq(data[8], 1.0, RELAXED_TOL));
    }

    #[test]
    fn test_inv_not_square() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert!(CpuBackend::inv(&a).is_err());
    }

    // ============ det ============

    #[test]
    fn test_det_2x2() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let det = CpuBackend::det(&a).unwrap();
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        assert!(approx_eq(det, -2.0, RELAXED_TOL));
    }

    #[test]
    fn test_det_3x3() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let det = CpuBackend::det(&a).unwrap();
        // This matrix is singular, det = 0
        assert!(approx_eq(det, 0.0, RELAXED_TOL));
    }

    #[test]
    fn test_det_identity() {
        let a = CpuBackend::eye(3);
        let det = CpuBackend::det(&a).unwrap();
        assert!(approx_eq(det, 1.0, RELAXED_TOL));
    }

    // ============ trace ============

    #[test]
    fn test_trace() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let tr = CpuBackend::trace(&a).unwrap();
        assert_eq!(tr, 15.0); // 1 + 5 + 9
    }

    #[test]
    fn test_trace_identity() {
        let a = CpuBackend::eye(5);
        let tr = CpuBackend::trace(&a).unwrap();
        assert_eq!(tr, 5.0);
    }

    // ============ norm ============

    #[test]
    fn test_norm_l2() {
        let a = vec1d(vec![3.0, 4.0]);
        let n = CpuBackend::norm(&a, Some(2.0)).unwrap();
        assert!(approx_eq(n, 5.0, DEFAULT_TOL));
    }

    #[test]
    fn test_norm_l1() {
        let a = vec1d(vec![-3.0, 4.0]);
        let n = CpuBackend::norm(&a, Some(1.0)).unwrap();
        assert!(approx_eq(n, 7.0, DEFAULT_TOL));
    }

    #[test]
    fn test_norm_linf() {
        let a = vec1d(vec![-3.0, 4.0, 2.0]);
        let n = CpuBackend::norm(&a, Some(f64::INFINITY)).unwrap();
        assert!(approx_eq(n, 4.0, DEFAULT_TOL));
    }

    // ============ solve ============

    #[test]
    fn test_solve() {
        // Solve Ax = b where A = [[3,1],[1,2]], b = [[9],[8]]
        // Solution: x = [[2],[3]]
        let a = mat(vec![3.0, 1.0, 1.0, 2.0], 2, 2);
        let b = mat(vec![9.0, 8.0], 2, 1);
        let x = CpuBackend::solve(&a, &b).unwrap();

        assert!(approx_eq(x.as_f64_slice()[0], 2.0, RELAXED_TOL));
        assert!(approx_eq(x.as_f64_slice()[1], 3.0, RELAXED_TOL));
    }

    #[test]
    fn test_solve_verify() {
        // Verify A @ x = b
        let a = mat(vec![3.0, 1.0, 1.0, 2.0], 2, 2);
        let b = mat(vec![9.0, 8.0], 2, 1);
        let x = CpuBackend::solve(&a, &b).unwrap();

        // Reshape x for matmul
        let x_2d = CpuArray::from_f64_vec(x.as_f64_slice(), vec![2, 1]).unwrap();
        let ax = CpuBackend::matmul(&a, &x_2d).unwrap();

        assert!(approx_eq(ax.as_f64_slice()[0], 9.0, RELAXED_TOL));
        assert!(approx_eq(ax.as_f64_slice()[1], 8.0, RELAXED_TOL));
    }

    // ============ qr ============

    #[test]
    fn test_qr() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let (q, r) = CpuBackend::qr(&a).unwrap();

        // Q @ R should equal A
        let reconstructed = CpuBackend::matmul(&q, &r).unwrap();
        for (x, y) in a
            .as_f64_slice()
            .iter()
            .zip(reconstructed.as_f64_slice().iter())
        {
            assert!(approx_eq(*x, *y, RELAXED_TOL));
        }
    }

    #[test]
    fn test_qr_orthogonal_q() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let (q, _) = CpuBackend::qr(&a).unwrap();

        // Q @ Q^T should be identity
        let qt = CpuBackend::transpose(&q);
        let qqt = CpuBackend::matmul(&q, &qt).unwrap();
        let data = qqt.as_f64_slice();
        assert!(approx_eq(data[0], 1.0, RELAXED_TOL));
        assert!(approx_eq(data[1], 0.0, RELAXED_TOL));
        assert!(approx_eq(data[2], 0.0, RELAXED_TOL));
        assert!(approx_eq(data[3], 1.0, RELAXED_TOL));
    }

    // ============ svd ============

    #[test]
    fn test_svd() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let (u, s, vt) = CpuBackend::svd(&a).unwrap();

        // Verify shapes
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 3]);

        // Singular values should be positive and sorted descending
        let s_data = s.as_f64_slice();
        assert!(s_data[0] >= s_data[1]);
        assert!(s_data[1] >= 0.0);
    }

    // ============ transpose ============

    #[test]
    fn test_transpose_2x3() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let at = CpuBackend::transpose(&a);

        assert_eq!(at.shape(), &[3, 2]);
        assert_eq!(at.as_f64_slice(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_square() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let at = CpuBackend::transpose(&a);

        assert_eq!(at.as_f64_slice(), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_1d() {
        let a = vec1d(vec![1.0, 2.0, 3.0]);
        let at = CpuBackend::transpose(&a);

        // 1D transpose is a no-op
        assert_eq!(at.shape(), a.shape());
        assert_eq!(at.as_f64_slice(), a.as_f64_slice());
    }

    #[test]
    fn test_transpose_double() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let att = CpuBackend::transpose(&CpuBackend::transpose(&a));

        assert_eq!(att.shape(), a.shape());
        assert_eq!(att.as_f64_slice(), a.as_f64_slice());
    }
}
