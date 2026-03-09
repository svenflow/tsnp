//! Abstract array type that backends implement

use crate::dtype::DType;
use serde::{Deserialize, Serialize};

/// Metadata about an array (backend-agnostic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayMeta {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub strides: Vec<usize>,
    pub is_contiguous: bool,
}

impl ArrayMeta {
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        let strides = Self::compute_strides(&shape, dtype.size());
        Self {
            shape,
            dtype,
            strides,
            is_contiguous: true,
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.size() * self.dtype.size()
    }

    fn compute_strides(shape: &[usize], item_size: usize) -> Vec<usize> {
        let mut strides = vec![item_size; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

/// Core array trait that all backends implement
///
/// This trait defines the abstract interface for N-dimensional arrays.
/// Each backend (CPU, WASM, WebGPU) provides its own implementation.
pub trait Array: Clone + std::fmt::Debug {
    /// Get array metadata
    fn meta(&self) -> &ArrayMeta;

    /// Get shape
    fn shape(&self) -> &[usize] {
        &self.meta().shape
    }

    /// Get number of dimensions
    fn ndim(&self) -> usize {
        self.meta().ndim()
    }

    /// Get total number of elements
    fn size(&self) -> usize {
        self.meta().size()
    }

    /// Get data type
    fn dtype(&self) -> DType {
        self.meta().dtype
    }

    /// Get raw data as f64 slice (for testing/comparison)
    fn as_f64_slice(&self) -> Vec<f64>;

    /// Create from f64 data and shape
    fn from_f64_vec(data: Vec<f64>, shape: Vec<usize>) -> crate::Result<Self>
    where
        Self: Sized;

    /// Get element at flat index
    fn get_flat(&self, index: usize) -> f64;

    /// Set element at flat index
    fn set_flat(&mut self, index: usize, value: f64);
}
