//! Pointshare
//!
//! This module contains operations with point shares

mod ops;
pub(super) mod types;

use ark_ec::CurveGroup;

use super::{
    core, network::ShamirNetwork, IoResult, ShamirPointShare, ShamirPrimeFieldShare,
    ShamirProtocol, ShamirShare,
};

type FieldShare<C> = ShamirPrimeFieldShare<C>;
type PointShare<C> = ShamirPointShare<C>;

/// Performs addition between two shares.
pub fn add<C: CurveGroup>(a: &PointShare<C>, b: &PointShare<C>) -> PointShare<C> {
    a + b
}

/// Performs subtraction between two shares.
pub fn sub<C: CurveGroup>(a: &PointShare<C>, b: &PointShare<C>) -> PointShare<C> {
    a - b
}

/// Performs addition between two shares and stores the result in `a`.
pub fn add_assign<C: CurveGroup>(a: &mut PointShare<C>, b: &PointShare<C>) {
    *a += b;
}

/// Performs subtraction between two shares and stores the result in `a`.
pub fn sub_assign<C: CurveGroup>(a: &mut PointShare<C>, b: &PointShare<C>) {
    *a -= b;
}

/// Performs addition between a share and a public value and stores the result in `a`.
pub fn add_assign_public<C: CurveGroup>(a: &mut PointShare<C>, b: &C) {
    a.a += b
}

/// Performs subtraction between a share and a public value and stores the result in `a`.
pub fn sub_assign_public<C: CurveGroup>(a: &mut PointShare<C>, b: &C) {
    a.a -= b
}

/// Performs addition between a share and a public affine value and stores the result in `a`.
pub fn add_assign_public_affine<C: CurveGroup>(
    a: &mut PointShare<C>,
    b: &<C as CurveGroup>::Affine,
) {
    a.a += b
}

/// Performs subtraction between a share and a public affine value and stores the result in `a`.
pub fn sub_assign_public_affine<C: CurveGroup>(
    a: &mut PointShare<C>,
    b: &<C as CurveGroup>::Affine,
) {
    a.a -= b
}

/// Performs multiplication between a field share and a public curve group value and stores the result in `a`.
pub fn scalar_mul_public_point<C: CurveGroup>(
    shared: ShamirPrimeFieldShare<C::ScalarField>,
    public: &C,
) -> PointShare<C> {
    PointShare::<C> {
        a: public.mul(shared.a),
    }
}

/// Performs scalar multiplication between a point share and a public scalar.
pub fn scalar_mul_public_scalar<C: CurveGroup>(
    a: &PointShare<C>,
    b: &C::ScalarField,
) -> PointShare<C> {
    a * b
}

/// Performs scalar multiplication between a point share and a field share.
pub async fn scalar_mul<C: CurveGroup, N: ShamirNetwork>(
    a: &PointShare<C>,
    b: ShamirShare<C::ScalarField>,
    shamir: &mut ShamirProtocol<C::ScalarField, N>,
) -> IoResult<PointShare<C>> {
    let mul = (b * a).a;
    shamir.degree_reduce_point(mul).await
}

/// Performs opening of a point share.
pub async fn open_point<C: CurveGroup, N: ShamirNetwork>(
    a: &PointShare<C>,
    shamir: &mut ShamirProtocol<C::ScalarField, N>,
) -> IoResult<C> {
    let rcv = shamir
        .network
        .broadcast_next(a.a, shamir.threshold + 1)
        .await?;
    let res = core::reconstruct_point(&rcv, &shamir.open_lagrange_t);
    Ok(res)
}

/// Performs opening of a vector of point shares.
pub async fn open_point_many<C: CurveGroup, N: ShamirNetwork>(
    a: &[PointShare<C>],
    shamir: &mut ShamirProtocol<C::ScalarField, N>,
) -> IoResult<Vec<C>> {
    let a_a = ShamirPointShare::convert_slice(a);

    let rcv = shamir
        .network
        .broadcast_next(a_a.to_owned(), shamir.threshold + 1)
        .await?;

    let mut transposed = vec![vec![C::zero(); shamir.threshold + 1]; a.len()];

    for (j, r) in rcv.into_iter().enumerate() {
        for (i, val) in r.into_iter().enumerate() {
            transposed[i][j] = val;
        }
    }

    let res = transposed
        .into_iter()
        .map(|r| core::reconstruct_point(&r, &shamir.open_lagrange_t))
        .collect();
    Ok(res)
}

/// Perfoms MSM between curve points and field shares.
pub fn msm_public_points<C: CurveGroup>(
    points: &[C::Affine],
    scalars: &[FieldShare<C::ScalarField>],
) -> PointShare<C> {
    tracing::trace!("> MSM public points for {} elements", points.len());
    debug_assert_eq!(points.len(), scalars.len());
    let res = C::msm_unchecked(points, &scalars.iter().map(|s| s.a).collect::<Vec<_>>());
    tracing::trace!("< MSM public points for {} elements", points.len());
    PointShare::<C> { a: res }
}
