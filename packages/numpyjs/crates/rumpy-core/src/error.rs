//! Error types for RumPy

use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum RumpyError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Incompatible shapes for operation: {0:?} and {1:?}")]
    IncompatibleShapes(Vec<usize>, Vec<usize>),

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Index out of bounds: index {index} for axis of size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("Invalid axis: {axis} for array with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    #[error("Matrix must be square, got shape {0:?}")]
    NotSquare(Vec<usize>),

    #[error("Matrix is singular")]
    SingularMatrix,

    #[error("Dimension mismatch for {op}: {shapes:?}")]
    DimensionMismatch {
        op: &'static str,
        shapes: (Vec<usize>, Vec<usize>),
    },

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("zero-size array to reduction operation {0} which has no identity")]
    EmptyArrayReduction(&'static str),
}

pub type Result<T> = std::result::Result<T, RumpyError>;
