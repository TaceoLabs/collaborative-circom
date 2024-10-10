use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_ff::PrimeField;
use itertools::izip;
use mpc_core::protocols::shamir::{
    arithmetic, network::ShamirNetwork, pointshare, ShamirPointShare, ShamirPrimeFieldShare,
    ShamirProtocol,
};
use num_traits::Zero;
use rayon::prelude::*;

use super::NoirUltraHonkProver;

/// A UltraHonk dirver unsing shamir secret sharing
///
/// Contains two [`ShamirProtocol`]s, `protocol0` for the main execution and `protocol0` for parts that can run concurrently.
// TODO use protocol1?
pub struct ShamirUltraHonkDriver<F: PrimeField, N: ShamirNetwork> {
    protocol0: ShamirProtocol<F, N>,
    _protocol1: ShamirProtocol<F, N>,
}

impl<F: PrimeField, N: ShamirNetwork> ShamirUltraHonkDriver<F, N> {
    /// Create a new [`ShamirGroth16Driver`] with two [`ShamirProtocol`]s
    pub fn new(protocol0: ShamirProtocol<F, N>, protocol1: ShamirProtocol<F, N>) -> Self {
        Self {
            protocol0,
            _protocol1: protocol1,
        }
    }
}

impl<P: Pairing, N: ShamirNetwork> NoirUltraHonkProver<P>
    for ShamirUltraHonkDriver<P::ScalarField, N>
{
    type ArithmeticShare = ShamirPrimeFieldShare<P::ScalarField>;
    type PointShareG1 = ShamirPointShare<P::G1>;
    type PointShareG2 = ShamirPointShare<P::G2>;
    type PartyID = usize;

    fn rand(&mut self) -> std::io::Result<Self::ArithmeticShare> {
        self.protocol0.rand()
    }

    fn get_party_id(&self) -> Self::PartyID {
        self.protocol0.network.get_id()
    }

    fn sub(&self, a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare {
        arithmetic::sub(a, b)
    }

    fn add(&self, a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare {
        arithmetic::add(a, b)
    }

    fn neg(&mut self, a: Self::ArithmeticShare) -> Self::ArithmeticShare {
        arithmetic::neg(a)
    }

    fn mul_with_public(
        &self,
        public: P::ScalarField,
        shared: Self::ArithmeticShare,
    ) -> Self::ArithmeticShare {
        arithmetic::mul_public(shared, public)
    }

    async fn mul_many(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> std::io::Result<Vec<Self::ArithmeticShare>> {
        arithmetic::mul_vec(a, b, &mut self.protocol0).await
    }

    fn add_with_public(
        &self,
        public: P::ScalarField,
        shared: Self::ArithmeticShare,
    ) -> Self::ArithmeticShare {
        arithmetic::add_public(shared, public)
    }

    fn promote_to_trivial_share(
        _id: Self::PartyID,
        public_value: P::ScalarField,
    ) -> Self::ArithmeticShare {
        arithmetic::promote_to_trivial_share(public_value)
    }

    fn promote_to_trivial_shares(
        id: Self::PartyID,
        public_values: &[P::ScalarField],
    ) -> Vec<Self::ArithmeticShare> {
        public_values
            .par_iter()
            .with_min_len(1024)
            .map(|value| <ShamirUltraHonkDriver<P::ScalarField, N> as NoirUltraHonkProver<P>>::promote_to_trivial_share(id, *value))
            .collect()
    }

    async fn open_point(&mut self, a: Self::PointShareG1) -> std::io::Result<P::G1> {
        pointshare::open_point(&a, &mut self.protocol0).await
    }

    async fn open_point_many(&mut self, a: &[Self::PointShareG1]) -> std::io::Result<Vec<P::G1>> {
        pointshare::open_point_many(a, &mut self.protocol0).await
    }

    async fn open_many(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> std::io::Result<Vec<P::ScalarField>> {
        arithmetic::open_vec(a, &mut self.protocol0).await
    }

    async fn mul_open_many(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> std::io::Result<Vec<P::ScalarField>> {
        arithmetic::mul_open_vec(a, b, &mut self.protocol0).await
    }

    async fn inv_many(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> std::io::Result<Vec<Self::ArithmeticShare>> {
        arithmetic::inv_vec(a, &mut self.protocol0).await
    }

    async fn inv_many_in_place(&mut self, a: &mut [Self::ArithmeticShare]) -> std::io::Result<()> {
        let r = (0..a.len())
            .map(|_| {
                <ShamirUltraHonkDriver<P::ScalarField, N> as NoirUltraHonkProver<P>>::rand(self)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let y: Vec<P::ScalarField> =
            <ShamirUltraHonkDriver<P::ScalarField, N> as NoirUltraHonkProver<P>>::mul_open_many(
                self, a, &r,
            )
            .await?;

        for (a, r, y) in izip!(a.iter_mut(), r, y) {
            if y.is_zero() {
                *a = Self::ArithmeticShare::default();
            } else {
                *a = r * y.inverse().unwrap();
            }
        }

        Ok(())
    }

    fn msm_public_points_g1(
        points: &[P::G1Affine],
        scalars: &[Self::ArithmeticShare],
    ) -> Self::PointShareG1 {
        pointshare::msm_public_points(points, scalars)
    }

    fn msm_public_points_g2(
        points: &[P::G2Affine],
        scalars: &[Self::ArithmeticShare],
    ) -> Self::PointShareG2 {
        pointshare::msm_public_points(points, scalars)
    }
}
