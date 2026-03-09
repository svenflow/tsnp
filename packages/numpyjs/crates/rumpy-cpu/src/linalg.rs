//! Linear algebra operations for CPU backend using faer
//! On WASM targets with simd128, uses hand-optimized SIMD GEMM kernels.

#[cfg(target_arch = "wasm32")]
use crate::simd_gemm;
use crate::{CpuArray, CpuBackend};
use faer::Mat;
use ndarray::{ArrayD, IxDyn};
use rumpy_core::{ops::LinalgOps, Array, Result, RumpyError};

/// Convert CpuArray to faer Mat (assumes 2D, row-major)
fn to_faer(arr: &CpuArray) -> Result<Mat<f64>> {
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(RumpyError::InvalidArgument("Matrix must be 2D".to_string()));
    }
    let (m, n) = (shape[0], shape[1]);
    let data = arr.as_f64_slice();

    // faer uses column-major, we have row-major
    Ok(Mat::from_fn(m, n, |i, j| data[i * n + j]))
}

/// Convert faer Mat to CpuArray (row-major)
fn from_faer(mat: &Mat<f64>) -> CpuArray {
    let (m, n) = (mat.nrows(), mat.ncols());
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(mat.read(i, j));
        }
    }
    CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[m, n]), data).unwrap())
}

/// Convert faer column vector to CpuArray
fn from_faer_vec(mat: &Mat<f64>) -> CpuArray {
    let n = mat.nrows();
    let data: Vec<f64> = (0..n).map(|i| mat.read(i, 0)).collect();
    CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[n]), data).unwrap())
}

impl LinalgOps for CpuBackend {
    type Array = CpuArray;

    fn matmul(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(RumpyError::InvalidArgument("Matrix must be 2D".to_string()));
        }

        let (m, k1) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);

        if k1 != k2 {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        #[cfg(target_arch = "wasm32")]
        let k = k1;
        #[cfg(not(target_arch = "wasm32"))]
        let _ = (m, n, k1);

        // Use SIMD-optimized GEMM on WASM (simd128 is enabled via .cargo/config.toml rustflags)
        // Fall back to faer for native targets
        #[cfg(target_arch = "wasm32")]
        {
            let a_data = a.as_f64_slice();
            let b_data = b.as_f64_slice();
            let c_data = simd_gemm::matmul_dispatch_f64(&a_data, &b_data, m, n, k);
            Ok(CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&[m, n]), c_data).unwrap(),
            ))
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mat_a = to_faer(a)?;
            let mat_b = to_faer(b)?;
            let result = &mat_a * &mat_b;
            Ok(from_faer(&result))
        }
    }

    fn dot(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() == 1 && b_shape.len() == 1 {
            // Vector dot product
            if a_shape[0] != b_shape[0] {
                return Err(RumpyError::IncompatibleShapes(
                    a_shape.to_vec(),
                    b_shape.to_vec(),
                ));
            }
            let result: f64 = a
                .as_f64_slice()
                .iter()
                .zip(b.as_f64_slice().iter())
                .map(|(x, y)| x * y)
                .sum();
            Ok(CpuArray::from_f64_vec(vec![result], vec![1])?)
        } else if a_shape.len() == 2 && b_shape.len() == 2 {
            // Matrix multiplication
            Self::matmul(a, b)
        } else {
            Err(RumpyError::InvalidArgument(
                "dot requires 1D or 2D arrays".to_string(),
            ))
        }
    }

    fn inner(a: &CpuArray, b: &CpuArray) -> Result<f64> {
        if a.size() != b.size() {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }
        Ok(a.as_f64_slice()
            .iter()
            .zip(b.as_f64_slice().iter())
            .map(|(x, y)| x * y)
            .sum())
    }

    fn outer(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let a_data = a.as_f64_slice();
        let b_data = b.as_f64_slice();
        let m = a_data.len();
        let n = b_data.len();

        let mut result = Vec::with_capacity(m * n);
        for &ai in &a_data {
            for &bi in &b_data {
                result.push(ai * bi);
            }
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&[m, n]), result).unwrap(),
        ))
    }

    fn inv(arr: &CpuArray) -> Result<CpuArray> {
        let mat = to_faer(arr)?;
        if mat.nrows() != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        // Use Gauss-Jordan elimination for inverse
        let n = mat.nrows();
        let mut aug = Mat::zeros(n, 2 * n);

        // Copy matrix to left half, identity to right half
        for i in 0..n {
            for j in 0..n {
                aug.write(i, j, mat.read(i, j));
                aug.write(i, n + j, if i == j { 1.0 } else { 0.0 });
            }
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug.read(k, i).abs() > aug.read(max_row, i).abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..(2 * n) {
                    let tmp = aug.read(i, j);
                    aug.write(i, j, aug.read(max_row, j));
                    aug.write(max_row, j, tmp);
                }
            }

            let pivot = aug.read(i, i);
            if pivot.abs() < 1e-14 {
                return Err(RumpyError::SingularMatrix);
            }

            // Scale pivot row
            for j in 0..(2 * n) {
                aug.write(i, j, aug.read(i, j) / pivot);
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = aug.read(k, i);
                    for j in 0..(2 * n) {
                        aug.write(k, j, aug.read(k, j) - factor * aug.read(i, j));
                    }
                }
            }
        }

        // Extract inverse from right half
        let mut inv = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.write(i, j, aug.read(i, n + j));
            }
        }

        Ok(from_faer(&inv))
    }

    fn pinv(arr: &CpuArray) -> Result<CpuArray> {
        // SVD-based pseudoinverse that handles rank-deficient matrices
        // A+ = V S+ U^T where S+ has 1/sigma for non-zero singular values
        let (u, s, vt) = Self::svd(arr)?;

        let u_mat = to_faer(&u)?;
        let vt_mat = to_faer(&vt)?;
        let s_data = s.as_f64_slice();

        let m = u_mat.nrows();
        let n = vt_mat.ncols();
        let k = s_data.len();

        // Compute tolerance based on machine epsilon and matrix size
        let max_dim = m.max(n) as f64;
        let max_s = s_data.iter().cloned().fold(0.0_f64, f64::max);
        let tol = f64::EPSILON * max_dim * max_s;

        // Build S+ (pseudoinverse of singular values)
        let s_pinv: Vec<f64> = s_data.iter()
            .map(|&sigma| if sigma > tol { 1.0 / sigma } else { 0.0 })
            .collect();

        // Compute V S+ (each column of V scaled by 1/sigma)
        // Note: vt_mat is k x n, so V is n x k
        let mut vs_pinv = Mat::zeros(n, k);
        for i in 0..n {
            for j in 0..k {
                vs_pinv.write(i, j, vt_mat.read(j, i) * s_pinv[j]);
            }
        }

        // Compute (V S+) U^T = A+
        // Result is n x m
        let mut result = Mat::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += vs_pinv.read(i, l) * u_mat.read(j, l);
                }
                result.write(i, j, sum);
            }
        }

        Ok(from_faer(&result))
    }

    fn det(arr: &CpuArray) -> Result<f64> {
        let mat = to_faer(arr)?;
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        // LU decomposition for determinant
        let mut work = mat.clone();
        let mut det = 1.0;
        let mut swaps = 0;

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if work.read(k, i).abs() > work.read(max_row, i).abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                for j in 0..n {
                    let tmp = work.read(i, j);
                    work.write(i, j, work.read(max_row, j));
                    work.write(max_row, j, tmp);
                }
                swaps += 1;
            }

            let pivot = work.read(i, i);
            if pivot.abs() < 1e-14 {
                return Ok(0.0);
            }

            det *= pivot;

            for k in (i + 1)..n {
                let factor = work.read(k, i) / pivot;
                for j in i..n {
                    work.write(k, j, work.read(k, j) - factor * work.read(i, j));
                }
            }
        }

        if swaps % 2 == 1 {
            det = -det;
        }

        Ok(det)
    }

    fn trace(arr: &CpuArray) -> Result<f64> {
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(RumpyError::InvalidArgument(
                "trace requires 2D array".to_string(),
            ));
        }
        let n = shape[0].min(shape[1]);
        let data = arr.as_ndarray();
        let mut sum = 0.0;
        for i in 0..n {
            sum += data[IxDyn(&[i, i])];
        }
        Ok(sum)
    }

    fn rank(arr: &CpuArray) -> Result<usize> {
        // Simple rank estimation via row echelon form
        let mat = to_faer(arr)?;
        let (m, n) = (mat.nrows(), mat.ncols());
        let mut work = mat.clone();

        let mut rank = 0;
        let mut col = 0;

        for row in 0..m {
            if col >= n {
                break;
            }

            // Find pivot
            let mut max_row = row;
            for k in (row + 1)..m {
                if work.read(k, col).abs() > work.read(max_row, col).abs() {
                    max_row = k;
                }
            }

            if work.read(max_row, col).abs() < 1e-10 {
                col += 1;
                continue;
            }

            // Swap rows
            if max_row != row {
                for j in 0..n {
                    let tmp = work.read(row, j);
                    work.write(row, j, work.read(max_row, j));
                    work.write(max_row, j, tmp);
                }
            }

            // Eliminate
            let pivot = work.read(row, col);
            for k in (row + 1)..m {
                let factor = work.read(k, col) / pivot;
                for j in col..n {
                    work.write(k, j, work.read(k, j) - factor * work.read(row, j));
                }
            }

            rank += 1;
            col += 1;
        }

        Ok(rank)
    }

    fn norm(arr: &CpuArray, ord: Option<f64>) -> Result<f64> {
        let data = arr.as_f64_slice();
        let ord = ord.unwrap_or(2.0);

        if ord == 2.0 {
            Ok(data.iter().map(|x| x * x).sum::<f64>().sqrt())
        } else if ord == 1.0 {
            Ok(data.iter().map(|x| x.abs()).sum())
        } else if ord == f64::INFINITY {
            Ok(data.iter().map(|x| x.abs()).fold(0.0, f64::max))
        } else if ord == f64::NEG_INFINITY {
            Ok(data.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min))
        } else {
            Ok(data
                .iter()
                .map(|x| x.abs().powf(ord))
                .sum::<f64>()
                .powf(1.0 / ord))
        }
    }

    fn solve(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // Solve Ax = b using Gaussian elimination with partial pivoting
        let mat_a = to_faer(a)?;
        let mat_b = to_faer(b)?;

        let n = mat_a.nrows();
        if n != mat_a.ncols() {
            return Err(RumpyError::NotSquare(a.shape().to_vec()));
        }
        if n != mat_b.nrows() {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        let m = mat_b.ncols();

        // Augmented matrix [A | b]
        let mut aug = Mat::zeros(n, n + m);
        for i in 0..n {
            for j in 0..n {
                aug.write(i, j, mat_a.read(i, j));
            }
            for j in 0..m {
                aug.write(i, n + j, mat_b.read(i, j));
            }
        }

        // Forward elimination
        for i in 0..n {
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug.read(k, i).abs() > aug.read(max_row, i).abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                for j in 0..(n + m) {
                    let tmp = aug.read(i, j);
                    aug.write(i, j, aug.read(max_row, j));
                    aug.write(max_row, j, tmp);
                }
            }

            let pivot = aug.read(i, i);
            if pivot.abs() < 1e-14 {
                return Err(RumpyError::SingularMatrix);
            }

            for k in (i + 1)..n {
                let factor = aug.read(k, i) / pivot;
                for j in i..(n + m) {
                    aug.write(k, j, aug.read(k, j) - factor * aug.read(i, j));
                }
            }
        }

        // Back substitution
        let mut x = Mat::zeros(n, m);
        for i in (0..n).rev() {
            for j in 0..m {
                let mut sum = aug.read(i, n + j);
                for k in (i + 1)..n {
                    sum -= aug.read(i, k) * x.read(k, j);
                }
                x.write(i, j, sum / aug.read(i, i));
            }
        }

        if b.shape().len() == 1 || b.shape()[1] == 1 {
            Ok(from_faer_vec(&x))
        } else {
            Ok(from_faer(&x))
        }
    }

    fn lstsq(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let a_pinv = Self::pinv(a)?;
        Self::matmul(&a_pinv, b)
    }

    fn qr(arr: &CpuArray) -> Result<(CpuArray, CpuArray)> {
        // Gram-Schmidt QR decomposition
        let mat = to_faer(arr)?;
        let (m, n) = (mat.nrows(), mat.ncols());
        let k = m.min(n);

        let mut q = Mat::zeros(m, k);
        let mut r = Mat::zeros(k, n);

        for j in 0..k {
            // Start with column j of A
            let mut v: Vec<f64> = (0..m).map(|i| mat.read(i, j)).collect();

            // Orthogonalize against previous columns
            for i in 0..j {
                let dot: f64 = (0..m).map(|row| q.read(row, i) * mat.read(row, j)).sum();
                r.write(i, j, dot);
                for (row, v_elem) in v.iter_mut().enumerate().take(m) {
                    *v_elem -= dot * q.read(row, i);
                }
            }

            // Normalize
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            r.write(j, j, norm);

            if norm > 1e-14 {
                for (row, &v_elem) in v.iter().enumerate().take(m) {
                    q.write(row, j, v_elem / norm);
                }
            }
        }

        // Fill remaining R columns
        for j in k..n {
            for i in 0..k {
                let mut dot = 0.0;
                for row in 0..m {
                    dot += q.read(row, i) * mat.read(row, j);
                }
                r.write(i, j, dot);
            }
        }

        Ok((from_faer(&q), from_faer(&r)))
    }

    fn lu(arr: &CpuArray) -> Result<(CpuArray, CpuArray, CpuArray)> {
        let mat = to_faer(arr)?;
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        let mut l = Mat::zeros(n, n);
        let mut u = mat.clone();
        let mut p = Mat::zeros(n, n);

        // Initialize P as identity
        for i in 0..n {
            p.write(i, i, 1.0);
        }

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if u.read(k, i).abs() > u.read(max_row, i).abs() {
                    max_row = k;
                }
            }

            // Swap in U, L, and P
            if max_row != i {
                for j in 0..n {
                    let tmp = u.read(i, j);
                    u.write(i, j, u.read(max_row, j));
                    u.write(max_row, j, tmp);

                    let tmp = p.read(i, j);
                    p.write(i, j, p.read(max_row, j));
                    p.write(max_row, j, tmp);
                }
                for j in 0..i {
                    let tmp = l.read(i, j);
                    l.write(i, j, l.read(max_row, j));
                    l.write(max_row, j, tmp);
                }
            }

            l.write(i, i, 1.0);

            let pivot = u.read(i, i);
            if pivot.abs() < 1e-14 {
                continue;
            }

            for k in (i + 1)..n {
                let factor = u.read(k, i) / pivot;
                l.write(k, i, factor);
                for j in i..n {
                    u.write(k, j, u.read(k, j) - factor * u.read(i, j));
                }
            }
        }

        Ok((from_faer(&p), from_faer(&l), from_faer(&u)))
    }

    fn cholesky(arr: &CpuArray) -> Result<CpuArray> {
        let mat = to_faer(arr)?;
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        let mut l = Mat::zeros(n, n);

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                if j == i {
                    for k in 0..j {
                        sum += l.read(j, k) * l.read(j, k);
                    }
                    let diag = mat.read(j, j) - sum;
                    if diag <= 0.0 {
                        return Err(RumpyError::InvalidArgument(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }
                    l.write(j, j, diag.sqrt());
                } else {
                    for k in 0..j {
                        sum += l.read(i, k) * l.read(j, k);
                    }
                    l.write(i, j, (mat.read(i, j) - sum) / l.read(j, j));
                }
            }
        }

        Ok(from_faer(&l))
    }

    fn svd(arr: &CpuArray) -> Result<(CpuArray, CpuArray, CpuArray)> {
        // SIMPLIFIED SVD IMPLEMENTATION
        // =================================
        // This is a basic SVD computed via eigendecomposition of A^T A.
        // For production use, consider using a proper algorithm like:
        // - Golub-Kahan bidiagonalization with implicit QR shifts
        // - Divide-and-conquer SVD
        // - Jacobi SVD for small matrices
        //
        // Limitations of this implementation:
        // - May have numerical issues for ill-conditioned matrices
        // - Eigenvalue ordering may not be perfect for near-equal singular values
        // - Not optimized for large matrices
        //
        // For better accuracy and performance, consider linking to LAPACK/BLAS
        // via the faer crate's advanced features when available.

        let mat = to_faer(arr)?;
        let (m, n) = (mat.nrows(), mat.ncols());
        let k = m.min(n);

        // Compute A^T A for eigendecomposition
        let mt = mat.transpose();
        let ata = mt * &mat;

        // Eigenvalue computation for symmetric matrix A^T A
        let ata_arr = from_faer(&ata);
        let (eigenvalues, v) = Self::eig(&ata_arr)?;

        // Singular values are sqrt of eigenvalues (take only first k)
        // Clamp negative values to 0 (can occur due to numerical errors)
        let s_data: Vec<f64> = eigenvalues
            .as_f64_slice()
            .iter()
            .take(k)
            .map(|&x| x.max(0.0).sqrt())
            .collect();
        let s = CpuArray::from_f64_vec(s_data, vec![k])?;

        // V is n x n, we need only first k columns for V_k (n x k)
        let v_mat = to_faer(&v)?;

        // Compute tolerance for small singular values
        let max_s = s.as_f64_slice().iter().cloned().fold(0.0_f64, f64::max);
        let tol = f64::EPSILON * (m.max(n) as f64) * max_s;

        // U = A V_k S^-1 (m x k)
        let mut u = Mat::zeros(m, k);
        for j in 0..k {
            let sigma = s.get_flat(j);
            if sigma > tol {
                // Compute (A * v_j) / sigma
                for i in 0..m {
                    let mut sum = 0.0;
                    for l in 0..n {
                        sum += mat.read(i, l) * v_mat.read(l, j);
                    }
                    u.write(i, j, sum / sigma);
                }
            }
        }

        // V^T is k x n (first k rows of V transposed)
        let mut vt_data = vec![0.0; k * n];
        for i in 0..k {
            for j in 0..n {
                vt_data[i * n + j] = v_mat.read(j, i);
            }
        }
        let vt_arr = CpuArray::from_f64_vec(vt_data, vec![k, n])?;

        Ok((from_faer(&u), s, vt_arr))
    }

    fn eig(arr: &CpuArray) -> Result<(CpuArray, CpuArray)> {
        // Power iteration with deflation for eigendecomposition
        // Uses tolerance-based convergence instead of fixed iterations
        // NOTE: This is a simplified implementation suitable for symmetric matrices.
        // For non-symmetric matrices or better accuracy, consider QR iteration.
        let mat = to_faer(arr)?;
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        // Compute matrix Frobenius norm for relative tolerance
        let mut frob_norm_sq = 0.0;
        for i in 0..n {
            for j in 0..n {
                frob_norm_sq += mat.read(i, j).powi(2);
            }
        }
        let mat_scale = frob_norm_sq.sqrt().max(1.0);

        // Tolerance scaled to matrix magnitude and machine epsilon
        let tol = f64::EPSILON.sqrt() * mat_scale;
        let max_iterations = 1000; // Increased from 100

        let mut eigenvalues = Vec::with_capacity(n);
        let mut eigenvectors = Mat::zeros(n, n);

        let mut work = mat.clone();

        for k in 0..n {
            // Power iteration for dominant eigenvalue
            // Initialize with a vector that has some variation
            let mut v: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.01).collect();

            // Normalize initial vector
            let init_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for elem in &mut v {
                *elem /= init_norm;
            }

            let mut eigenvalue = 0.0;
            let mut prev_eigenvalue = f64::INFINITY;

            for _iter in 0..max_iterations {
                // Multiply: new_v = A * v
                let mut new_v = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        new_v[i] += work.read(i, j) * v[j];
                    }
                }

                // Compute Rayleigh quotient: λ = v^T A v / v^T v = v^T new_v / v^T v
                let v_dot_v: f64 = v.iter().map(|x| x * x).sum();
                let v_dot_new_v: f64 = v.iter().zip(new_v.iter()).map(|(a, b)| a * b).sum();

                if v_dot_v < f64::EPSILON {
                    break;
                }

                eigenvalue = v_dot_new_v / v_dot_v;

                // Normalize
                let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < f64::EPSILON {
                    break;
                }

                for i in 0..n {
                    v[i] = new_v[i] / norm;
                }

                // Check convergence: |λ_new - λ_old| < tol
                if (eigenvalue - prev_eigenvalue).abs() < tol {
                    break;
                }
                prev_eigenvalue = eigenvalue;
            }

            eigenvalues.push(eigenvalue);
            for (i, &v_elem) in v.iter().enumerate() {
                eigenvectors.write(i, k, v_elem);
            }

            // Deflate: A = A - λ v v^T
            for i in 0..n {
                for j in 0..n {
                    work.write(i, j, work.read(i, j) - eigenvalue * v[i] * v[j]);
                }
            }
        }

        let eig_arr = CpuArray::from_f64_vec(eigenvalues, vec![n])?;
        Ok((eig_arr, from_faer(&eigenvectors)))
    }

    fn eigvals(arr: &CpuArray) -> Result<CpuArray> {
        let (vals, _) = Self::eig(arr)?;
        Ok(vals)
    }

    fn transpose(arr: &CpuArray) -> CpuArray {
        let data = arr.as_ndarray();
        let shape = data.shape();

        if shape.len() == 1 {
            return arr.clone();
        }

        if shape.len() == 2 {
            let (m, n) = (shape[0], shape[1]);
            let mut result = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    result[j * m + i] = data[IxDyn(&[i, j])];
                }
            }
            return CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[n, m]), result).unwrap());
        }

        // General transpose: reverse axes
        let reversed: Vec<usize> = (0..shape.len()).rev().collect();
        CpuArray::from_ndarray(data.clone().permuted_axes(reversed))
    }

    fn tensordot(a: &CpuArray, b: &CpuArray, axes: usize) -> Result<CpuArray> {
        // Simplified tensordot: sum over last `axes` axes of a and first `axes` axes of b
        let a_shape = a.shape();
        let b_shape = b.shape();

        if axes > a_shape.len() || axes > b_shape.len() {
            return Err(RumpyError::InvalidArgument(format!(
                "axes={} exceeds array dimensions (a={}, b={})",
                axes, a_shape.len(), b_shape.len()
            )));
        }

        // Check that contracted dimensions match
        let a_contract: Vec<usize> = a_shape[(a_shape.len() - axes)..].to_vec();
        let b_contract: Vec<usize> = b_shape[..axes].to_vec();
        if a_contract != b_contract {
            return Err(RumpyError::IncompatibleShapes(a_shape.to_vec(), b_shape.to_vec()));
        }

        // Result shape: a_shape[:-axes] + b_shape[axes:]
        let mut result_shape: Vec<usize> = a_shape[..(a_shape.len() - axes)].to_vec();
        result_shape.extend_from_slice(&b_shape[axes..]);

        // If result is empty (scalar), return single value
        if result_shape.is_empty() {
            result_shape.push(1);
        }

        let contract_size: usize = a_contract.iter().product();
        let a_outer_size: usize = a_shape[..(a_shape.len() - axes)].iter().product();
        let b_outer_size: usize = b_shape[axes..].iter().product();

        let a_data = a.as_f64_slice();
        let b_data = b.as_f64_slice();

        let mut result = Vec::with_capacity(a_outer_size * b_outer_size);

        for i in 0..a_outer_size {
            for j in 0..b_outer_size {
                let mut sum = 0.0;
                for k in 0..contract_size {
                    let a_idx = i * contract_size + k;
                    let b_idx = k * b_outer_size + j;
                    sum += a_data[a_idx] * b_data[b_idx];
                }
                result.push(sum);
            }
        }

        Ok(CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&result_shape), result).unwrap()))
    }

    fn trapz(arr: &CpuArray, dx: f64, axis: Option<usize>) -> Result<CpuArray> {
        // Numerical integration using trapezoidal rule
        let shape = arr.shape();
        let ndim = shape.len();

        match axis {
            None => {
                // Integrate flattened array
                let data = arr.as_f64_slice();
                if data.len() < 2 {
                    return Ok(CpuArray::from_f64_vec(vec![0.0], vec![1])?);
                }
                let mut result = 0.0;
                for i in 0..(data.len() - 1) {
                    result += (data[i] + data[i + 1]) * dx / 2.0;
                }
                Ok(CpuArray::from_f64_vec(vec![result], vec![1])?)
            }
            Some(ax) => {
                if ax >= ndim {
                    return Err(RumpyError::InvalidAxis { axis: ax, ndim });
                }

                let axis_len = shape[ax];
                if axis_len < 2 {
                    let mut new_shape = shape.to_vec();
                    new_shape.remove(ax);
                    if new_shape.is_empty() {
                        new_shape.push(1);
                    }
                    let zeros = vec![0.0; new_shape.iter().product()];
                    return Ok(CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&new_shape), zeros).unwrap()));
                }

                let data = arr.as_ndarray();
                let mut new_shape = shape.to_vec();
                new_shape.remove(ax);
                if new_shape.is_empty() {
                    new_shape.push(1);
                }

                let outer_size: usize = shape[..ax].iter().product();
                let inner_size: usize = shape[ax + 1..].iter().product();
                let result_size = new_shape.iter().product();

                let mut result = vec![0.0; result_size];

                for outer in 0..outer_size {
                    for inner in 0..inner_size {
                        let result_idx = outer * inner_size + inner;
                        let mut sum = 0.0;
                        for i in 0..(axis_len - 1) {
                            let idx_i = outer * (axis_len * inner_size) + i * inner_size + inner;
                            let idx_i1 = outer * (axis_len * inner_size) + (i + 1) * inner_size + inner;
                            sum += (data.as_slice().unwrap()[idx_i] + data.as_slice().unwrap()[idx_i1]) * dx / 2.0;
                        }
                        result[result_idx] = sum;
                    }
                }

                Ok(CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(data: Vec<f64>, rows: usize, cols: usize) -> CpuArray {
        CpuArray::from_f64_vec(data, vec![rows, cols]).unwrap()
    }

    fn vec1d(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_matmul() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = CpuBackend::matmul(&a, &b).unwrap();

        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(c.as_f64_slice(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_dot_vectors() {
        let a = vec1d(vec![1.0, 2.0, 3.0]);
        let b = vec1d(vec![4.0, 5.0, 6.0]);
        let c = CpuBackend::dot(&a, &b).unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(c.as_f64_slice()[0], 32.0);
    }

    #[test]
    fn test_inv() {
        let a = mat(vec![4.0, 7.0, 2.0, 6.0], 2, 2);
        let a_inv = CpuBackend::inv(&a).unwrap();

        // A @ A^-1 should be identity
        let identity = CpuBackend::matmul(&a, &a_inv).unwrap();
        let data = identity.as_f64_slice();
        assert!(approx_eq(data[0], 1.0));
        assert!(approx_eq(data[1], 0.0));
        assert!(approx_eq(data[2], 0.0));
        assert!(approx_eq(data[3], 1.0));
    }

    #[test]
    fn test_det() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let det = CpuBackend::det(&a).unwrap();
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        assert!(approx_eq(det, -2.0));
    }

    #[test]
    fn test_transpose() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let at = CpuBackend::transpose(&a);
        assert_eq!(at.shape(), &[3, 2]);
        assert_eq!(at.as_f64_slice(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
