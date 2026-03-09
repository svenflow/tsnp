//! RumPy Core - Backend traits and common types
//!
//! This crate defines the trait interface that all backends must implement.
//! It enables pluggable backends (CPU, WASM, WebGPU) with a consistent API.

pub mod array;
pub mod backend;
pub mod dtype;
pub mod error;
pub mod ops;

pub use array::{Array, ArrayMeta};
pub use backend::Backend;
pub use dtype::DType;
pub use error::{Result, RumpyError};
