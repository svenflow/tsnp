//! CPU Array implementation using ndarray

use ndarray::{ArrayD, IxDyn};
use rumpy_core::{Array, ArrayMeta, DType, Result, RumpyError};

/// CPU-backed N-dimensional array
#[derive(Debug, Clone)]
pub struct CpuArray {
    data: ArrayD<f64>,
    meta: ArrayMeta,
}

impl CpuArray {
    /// Create from ndarray
    pub fn from_ndarray(data: ArrayD<f64>) -> Self {
        let shape = data.shape().to_vec();
        let meta = ArrayMeta::new(shape, DType::Float64);
        Self { data, meta }
    }

    /// Get underlying ndarray reference
    pub fn as_ndarray(&self) -> &ArrayD<f64> {
        &self.data
    }

    /// Get mutable ndarray reference
    pub fn as_ndarray_mut(&mut self) -> &mut ArrayD<f64> {
        &mut self.data
    }

    /// Consume and return ndarray
    pub fn into_ndarray(self) -> ArrayD<f64> {
        self.data
    }

    /// Create from shape with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let data = ArrayD::zeros(IxDyn(&shape));
        Self::from_ndarray(data)
    }

    /// Create from shape with ones
    pub fn ones(shape: Vec<usize>) -> Self {
        let data = ArrayD::ones(IxDyn(&shape));
        Self::from_ndarray(data)
    }

    /// Create from shape with fill value
    pub fn full(shape: Vec<usize>, value: f64) -> Self {
        let data = ArrayD::from_elem(IxDyn(&shape), value);
        Self::from_ndarray(data)
    }
}

impl Array for CpuArray {
    fn meta(&self) -> &ArrayMeta {
        &self.meta
    }

    fn as_f64_slice(&self) -> Vec<f64> {
        self.data.iter().cloned().collect()
    }

    fn from_f64_vec(data: Vec<f64>, shape: Vec<usize>) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(RumpyError::InvalidShape(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_size
            )));
        }

        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| RumpyError::InvalidShape(e.to_string()))?;
        Ok(Self::from_ndarray(arr))
    }

    fn get_flat(&self, index: usize) -> f64 {
        self.data.iter().nth(index).cloned().unwrap_or(f64::NAN)
    }

    fn set_flat(&mut self, index: usize, value: f64) {
        if let Some(elem) = self.data.iter_mut().nth(index) {
            *elem = value;
        }
    }
}

// Implement PartialEq for testing
impl PartialEq for CpuArray {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.as_f64_slice() == other.as_f64_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_zeros() {
        let arr = CpuArray::zeros(vec![2, 3]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.size(), 6);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_create_ones() {
        let arr = CpuArray::ones(vec![3, 2]);
        assert_eq!(arr.shape(), &[3, 2]);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_vec() {
        let arr = CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.get_flat(0), 1.0);
        assert_eq!(arr.get_flat(3), 4.0);
    }

    #[test]
    fn test_from_vec_shape_mismatch() {
        let result = CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0], vec![2, 2]);
        assert!(result.is_err());
    }
}
