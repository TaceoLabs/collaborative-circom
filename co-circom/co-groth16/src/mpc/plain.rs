use std::sync::Arc;

use ark_ec::pairing::Pairing;
use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
use ark_ff::UniformRand;
use rand::thread_rng;

use super::CircomGroth16Prover;

type IoResult<T> = std::io::Result<T>;

/// A plain Groth16 driver
pub struct PlainGroth16Driver;

impl<P: Pairing> CircomGroth16Prover<P> for PlainGroth16Driver {
    type ArithmeticShare = P::ScalarField;

    type PointShareG1 = P::G1;

    type PointShareG2 = P::G2;

    type PartyID = usize;

    async fn close_network(self) -> super::IoResult<()> {
        Ok(())
    }

    fn rand(&mut self) -> IoResult<Self::ArithmeticShare> {
        let mut rng = thread_rng();
        Ok(Self::ArithmeticShare::rand(&mut rng))
    }

    fn get_party_id(&self) -> Self::PartyID {
        //does't matter
        0
    }

    fn evaluate_constraint(
        _: Self::PartyID,
        lhs: &[(P::ScalarField, usize)],
        public_inputs: &[P::ScalarField],
        private_witness: &[Self::ArithmeticShare],
    ) -> Self::ArithmeticShare {
        let mut acc = P::ScalarField::default();
        for (coeff, index) in lhs {
            if index < &public_inputs.len() {
                acc += *coeff * public_inputs[*index];
            } else {
                acc += *coeff * private_witness[*index - public_inputs.len()];
            }
        }
        acc
    }

    fn promote_to_trivial_shares(
        _: Self::PartyID,
        public_values: &[P::ScalarField],
    ) -> Vec<Self::ArithmeticShare> {
        public_values.to_vec()
    }

    fn local_mul_vec(
        &mut self,
        a: Vec<Self::ArithmeticShare>,
        b: Vec<Self::ArithmeticShare>,
    ) -> Vec<P::ScalarField> {
        a.iter().zip(b.iter()).map(|(a, b)| *a * b).collect()
    }

    async fn msm_and_mul(
        &mut self,
        h: Vec<<P as Pairing>::ScalarField>,
        h_query: Arc<Vec<P::G1Affine>>,
        r: Self::ArithmeticShare,
        s: Self::ArithmeticShare,
    ) -> IoResult<(Self::PointShareG1, Self::ArithmeticShare)> {
        Ok((P::G1::msm_unchecked(h_query.as_ref(), &h), r * s))
    }

    fn distribute_powers_and_mul_by_const(
        coeffs: &mut [Self::ArithmeticShare],
        roots: &[P::ScalarField],
    ) {
        #[allow(unused_mut)]
        for (mut c, pow) in coeffs.iter_mut().zip(roots) {
            *c *= pow;
        }
    }

    fn msm_public_points_g1(
        points: &[P::G1Affine],
        scalars: &[Self::ArithmeticShare],
    ) -> Self::PointShareG1 {
        P::G1::msm_unchecked(points, scalars)
    }

    fn msm_public_points_g2(
        points: &[P::G2Affine],
        scalars: &[Self::ArithmeticShare],
    ) -> Self::PointShareG2 {
        P::G2::msm_unchecked(points, scalars)
    }

    fn scalar_mul_public_point_g1(a: &P::G1, b: Self::ArithmeticShare) -> Self::PointShareG1 {
        *a * b
    }

    fn add_assign_points_g1(a: &mut Self::PointShareG1, b: &Self::PointShareG1) {
        *a += b;
    }

    fn add_assign_points_public_g1(_: Self::PartyID, a: &mut Self::PointShareG1, b: &P::G1) {
        *a += b;
    }

    async fn open_point_g1(&mut self, a: &Self::PointShareG1) -> super::IoResult<P::G1> {
        Ok(*a)
    }

    async fn scalar_mul_g1(
        &mut self,
        a: &Self::PointShareG1,
        b: Self::ArithmeticShare,
    ) -> super::IoResult<Self::PointShareG1> {
        Ok(*a * b)
    }

    fn sub_assign_points_g1(a: &mut Self::PointShareG1, b: &Self::PointShareG1) {
        *a -= b;
    }

    fn scalar_mul_public_point_g2(a: &P::G2, b: Self::ArithmeticShare) -> Self::PointShareG2 {
        *a * b
    }

    fn add_assign_points_g2(a: &mut Self::PointShareG2, b: &Self::PointShareG2) {
        *a += b;
    }

    fn add_assign_points_public_g2(_: Self::PartyID, a: &mut Self::PointShareG2, b: &P::G2) {
        *a += b;
    }

    async fn open_two_points(
        &mut self,
        a: Self::PointShareG1,
        b: Self::PointShareG2,
    ) -> std::io::Result<(P::G1, P::G2)> {
        Ok((a, b))
    }

    async fn open_point_and_scalar_mul(
        &mut self,
        g_a: &Self::PointShareG1,
        g1_b: &Self::PointShareG1,
        r: Self::ArithmeticShare,
    ) -> super::IoResult<(P::G1, Self::PointShareG1)> {
        Ok((*g_a, *g1_b * r))
    }
}
