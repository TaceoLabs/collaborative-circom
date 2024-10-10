use ark_ec::CurveGroup;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// This type represents a replicated shared point. Since a replicated share of a point contains additive shares of two parties, this type contains two point.
#[derive(Debug, Clone, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Rep3PointShare<C: CurveGroup> {
    /// Share of this party
    pub a: C,
    /// Share of the prev party
    pub b: C,
}

impl<C: CurveGroup> Rep3PointShare<C> {
    /// Contruct a new [`Rep3PointShare`]
    pub fn new(a: C, b: C) -> Self {
        Self { a, b }
    }
}
