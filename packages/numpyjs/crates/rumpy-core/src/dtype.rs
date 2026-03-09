//! Data types supported by RumPy

use serde::{Deserialize, Serialize};

/// Supported data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DType {
    Float32,
    #[default]
    Float64,
    Int32,
    Int64,
    UInt32,
    UInt64,
    Bool,
    Complex64,
    Complex128,
}

impl DType {
    /// Size in bytes
    pub fn size(&self) -> usize {
        match self {
            DType::Bool => 1,
            DType::Float32 | DType::Int32 | DType::UInt32 => 4,
            DType::Float64 | DType::Int64 | DType::UInt64 | DType::Complex64 => 8,
            DType::Complex128 => 16,
        }
    }

    /// String representation (NumPy compatible)
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Bool => "bool",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
