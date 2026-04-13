//! Tensor shape representation.

use serde::{Deserialize, Serialize};

/// N-dimensional shape for tensors.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a shape from a slice of dimensions.
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    /// Scalar (0-dimensional).
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    /// Dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get a specific dimension, or None if out of range.
    pub fn dim(&self, i: usize) -> Option<usize> {
        self.dims.get(i).copied()
    }

    /// Whether this is a scalar shape (rank 0).
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self { dims }
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_basics() {
        let s = Shape::new(&[2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.dim(1), Some(3));
        assert_eq!(s.dim(5), None);
    }

    #[test]
    fn test_scalar_shape() {
        let s = Shape::scalar();
        assert!(s.is_scalar());
        assert_eq!(s.numel(), 1);
        assert_eq!(s.rank(), 0);
    }
}
