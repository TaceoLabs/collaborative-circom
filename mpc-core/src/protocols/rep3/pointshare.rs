//! Pointshare
//!
//! This module contains operations with point shares

mod ops;
mod types;

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use itertools::{izip, Itertools};
use rayon::prelude::*;
pub use types::Rep3PointShare;

use super::{
    id::PartyID,
    network::{IoContext, Rep3Network},
    IoResult, Rep3PrimeFieldShare,
};

/// Type alias for a [`Rep3PrimeFieldShare`]
type FieldShare<C> = Rep3PrimeFieldShare<C>;
/// Type alias for a [`Rep3PointShare`]
type PointShare<C> = Rep3PointShare<C>;

/// Performs addition between two shared values.
pub fn add<C: CurveGroup>(a: &PointShare<C>, b: &PointShare<C>) -> PointShare<C> {
    a + b
}

/// Performs subtraction between two shared values.
pub fn sub<C: CurveGroup>(a: &PointShare<C>, b: &PointShare<C>) -> PointShare<C> {
    a - b
}

/// Performs addition between two shared values in place
pub fn add_assign<C: CurveGroup>(a: &mut PointShare<C>, b: &PointShare<C>) {
    *a += b;
}

/// Performs subtraction between two shared values in place
pub fn sub_assign<C: CurveGroup>(a: &mut PointShare<C>, b: &PointShare<C>) {
    *a -= b;
}

/// Performs addition between a shared value and a public value in place.
pub fn add_assign_public<C: CurveGroup>(a: &mut PointShare<C>, b: &C, id: PartyID) {
    match id {
        PartyID::ID0 => a.a += b,
        PartyID::ID1 => a.b += b,
        PartyID::ID2 => {}
    }
}

/// Performs subtraction between a shared value and a public value in place.
pub fn sub_assign_public<C: CurveGroup>(a: &mut PointShare<C>, b: &C, id: PartyID) {
    match id {
        PartyID::ID0 => a.a -= b,
        PartyID::ID1 => a.b -= b,
        PartyID::ID2 => {}
    }
}

/// Perform scalar multiplication of point * shared scalar
pub fn scalar_mul_public_point<C: CurveGroup>(
    a: &C,
    b: FieldShare<C::ScalarField>,
) -> PointShare<C> {
    PointShare {
        a: a.mul(b.a),
        b: a.mul(b.b),
    }
}

/// Perform scalar multiplication of shared point * scalar
pub fn scalar_mul_public_scalar<C: CurveGroup>(
    a: &PointShare<C>,
    b: C::ScalarField,
) -> PointShare<C> {
    a * b
}

/// Perform scalar multiplication
pub async fn scalar_mul<C: CurveGroup, N: Rep3Network>(
    a: &PointShare<C>,
    b: FieldShare<C::ScalarField>,
    io_context: &mut IoContext<N>,
) -> IoResult<PointShare<C>> {
    let local_a = b * a + io_context.rngs.rand.masking_ec_element::<C>();
    let local_b = io_context.network.reshare(local_a).await?;
    Ok(PointShare {
        a: local_a,
        b: local_b,
    })
}

/// Open the shared point
pub async fn open_point<C: CurveGroup, N: Rep3Network>(
    a: &PointShare<C>,
    io_context: &mut IoContext<N>,
) -> IoResult<C> {
    let c = io_context.network.reshare(a.b).await?;
    Ok(a.a + a.b + c)
}

/// Open the vector of [`PointShare`]s
pub async fn open_point_many<C: CurveGroup, N: Rep3Network>(
    a: &[PointShare<C>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<C>> {
    let bs = a.iter().map(|x| x.b).collect_vec();
    let cs = io_context.network.reshare(bs).await?;
    Ok(izip!(a, cs).map(|(x, c)| x.a + x.b + c).collect_vec())
}

/// Perform msm between `points` and `scalars`
pub fn msm_public_points<C: CurveGroup>(
    points: &[C::Affine],
    scalars: &[FieldShare<C::ScalarField>],
) -> PointShare<C> {
    tracing::trace!("> MSM public points for {} elements", points.len());
    debug_assert_eq!(points.len(), scalars.len());
    let (a_bigints, b_bigints) = scalars
        .into_par_iter()
        .map(|share| (share.a.into_bigint(), share.b.into_bigint()))
        .collect::<(Vec<_>, Vec<_>)>();
    let mut res_a = None;
    let mut res_b = None;
    rayon::scope(|s| {
        s.spawn(|_| res_a = Some(C::msm_bigint(points, &a_bigints)));
        s.spawn(|_| res_b = Some(C::msm_bigint(points, &b_bigints)));
    });
    tracing::trace!("< MSM public points for {} elements", points.len());
    //we can unwrap as the we have Some values after rayon scope
    PointShare::new(res_a.unwrap(), res_b.unwrap())
}
