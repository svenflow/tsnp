//! Backend trait that combines all operations

use crate::array::Array;
use crate::ops::*;

/// A complete backend implementation
///
/// This trait is automatically implemented for any type that implements
/// all the required operation traits. Backends just need to implement
/// the individual operation traits.
pub trait Backend:
    CreationOps
    + MathOps<Array = <Self as CreationOps>::Array>
    + StatsOps<Array = <Self as CreationOps>::Array>
    + LinalgOps<Array = <Self as CreationOps>::Array>
    + ManipulationOps<Array = <Self as CreationOps>::Array>
    + RandomOps<Array = <Self as CreationOps>::Array>
    + SortOps<Array = <Self as CreationOps>::Array>
    + CompareOps<Array = <Self as CreationOps>::Array>
where
    <Self as CreationOps>::Array: Array,
{
    /// Backend name for identification
    fn name() -> &'static str;

    /// Backend version
    fn version() -> &'static str;

    /// Whether SIMD is available
    fn has_simd() -> bool {
        false
    }

    /// Whether GPU acceleration is available
    fn has_gpu() -> bool {
        false
    }
}
