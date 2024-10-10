//! # Shamir Shared Curve Points
//!
//! This module contains the implementation of Shamir-shared curve points.

use std::mem::ManuallyDrop;

use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// This type represents a Shamir-shared EC point. Since a Shamir-share of a point is a point, this is a wrapper over a point.
#[derive(Debug, Clone, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
#[repr(transparent)]
pub struct ShamirPointShare<C: CurveGroup> {
    /// The point share
    pub a: C,
}
impl<C: CurveGroup> ShamirPointShare<C> {
    /// Wraps the point into a ShamirPointShare
    pub fn new(a: C) -> Self {
        Self { a }
    }

    /// Unwraps a ShamirPointShare into a point
    pub fn inner(self) -> C {
        self.a
    }
}

// Conversions
impl<C: CurveGroup> ShamirPointShare<C> {
    /// Transforms a slice of `ShamirPointShare<C>` to a slice of `C`
    // Safe because ShamirPointShare has repr(transparent)
    pub fn convert_slice(vec: &[Self]) -> &[C] {
        // SAFETY: ShamirPointShare has repr(transparent)
        unsafe { &*(vec as *const [Self] as *const [C]) }
    }

    /// Transforms a vector of `ShamirPointShare<C>` to a vector of `C`
    // Safe because ShamirPointShare has repr(transparent)
    pub fn convert_vec(vec: Vec<Self>) -> Vec<C> {
        let me = ManuallyDrop::new(vec);
        // SAFETY: ShamirPointShare has repr(transparent)
        unsafe { Vec::from_raw_parts(me.as_ptr() as *mut C, me.len(), me.capacity()) }
    }

    /// Transforms a slice of `C` to a slice of `ShamirPointShare<C>`
    // Safe because ShamirPointShare has repr(transparent)
    pub fn convert_slice_rev(vec: &[C]) -> &[Self] {
        // SAFETY: ShamirPointShare has repr(transparent)
        unsafe { &*(vec as *const [C] as *const [Self]) }
    }

    /// Transforms a vector of `C` to a vector of `ShamirPointShare<C>`
    // Safe because ShamirPointShare has repr(transparent)
    pub fn convert_vec_rev(vec: Vec<C>) -> Vec<Self> {
        let me = ManuallyDrop::new(vec);
        // SAFETY: ShamirPointShare has repr(transparent)
        unsafe { Vec::from_raw_parts(me.as_ptr() as *mut Self, me.len(), me.capacity()) }
    }

    /// Transforms a `ShamirPointShare<C>` to `C`
    pub fn convert(self) -> C {
        self.a
    }
}

#[cfg(test)]
mod unsafe_test {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    const ELEMENTS: usize = 100;

    fn conversion_test<C: CurveGroup>() {
        let mut rng = ChaCha12Rng::from_entropy();
        let t_vec: Vec<C> = (0..ELEMENTS).map(|_| C::rand(&mut rng)).collect();
        let rt_vec: Vec<ShamirPointShare<C>> = (0..ELEMENTS)
            .map(|_| ShamirPointShare::new(C::rand(&mut rng)))
            .collect();

        // Convert vec<C> to vec<G<C>>
        let t_conv = ShamirPointShare::convert_vec_rev(t_vec.to_owned());
        assert_eq!(t_conv.len(), t_vec.len());
        for (a, b) in t_conv.iter().zip(t_vec.iter()) {
            assert_eq!(a.a, *b)
        }

        // Convert slice vec<C> to vec<G<C>>
        let t_conv = ShamirPointShare::convert_slice_rev(&t_vec);
        assert_eq!(t_conv.len(), t_vec.len());
        for (a, b) in t_conv.iter().zip(t_vec.iter()) {
            assert_eq!(a.a, *b)
        }

        // Convert vec<G<C>> to vec<C>
        let rt_conv = ShamirPointShare::convert_vec(rt_vec.to_owned());
        assert_eq!(rt_conv.len(), rt_vec.len());
        for (a, b) in rt_conv.iter().zip(rt_vec.iter()) {
            assert_eq!(*a, b.a)
        }

        // Convert slice vec<G<C>> to vec<C>
        let rt_conv = ShamirPointShare::convert_slice(&rt_vec);
        assert_eq!(rt_conv.len(), rt_vec.len());
        for (a, b) in rt_conv.iter().zip(rt_vec.iter()) {
            assert_eq!(*a, b.a)
        }
    }

    macro_rules! test_impl {
        ($([$ty:ty,$fn:ident]),*) => ($(
            #[test]
            fn $fn() {
                conversion_test::<$ty>();
            }
        )*)
    }

    test_impl! {
        [ark_bn254::G1Projective, bn254_test]
    }
}
