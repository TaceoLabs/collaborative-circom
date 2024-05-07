// TODO move this file

use crate::groth16::CollaborativeGroth16;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_groth16::{Proof, ProvingKey};
use ark_relations::r1cs::Result as R1CSResult;
use ark_std::{end_timer, start_timer};
use mpc_core::traits::{EcMpcProtocol, FFTProvider, MSMProvider, PrimeFieldMpcProtocol};

type FieldShare<T, P> = <T as PrimeFieldMpcProtocol<<P as Pairing>::ScalarField>>::FieldShare;
type FieldShareSlice<'a, T, P> =
    <T as PrimeFieldMpcProtocol<<P as Pairing>::ScalarField>>::FieldShareSlice<'a>;
type PointShare<T, C> = <T as EcMpcProtocol<C>>::PointShare;

impl<T, P: Pairing> CollaborativeGroth16<T, P>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>
        + EcMpcProtocol<P::G1>
        + EcMpcProtocol<P::G2>
        + FFTProvider<P::ScalarField>
        + MSMProvider<P::G1>,
{
    fn calculate_coeff<C: CurveGroup>(
        initial: PointShare<T, C>,
        query: &[C::Affine],
        vk_param: C::Affine,
        input_assignment: &[P::ScalarField],
        aux_assignment: FieldShareSlice<'_, T, P>,
    ) -> PointShare<T, C>
    where
        T: EcMpcProtocol<C>,
    {
        todo!()
    }

    pub fn create_proof_with_assignment(
        &mut self,
        pk: &ProvingKey<P>,
        r: FieldShare<T, P>,
        s: FieldShare<T, P>,
        h: FieldShareSlice<'_, T, P>,
        input_assignment: &[P::ScalarField],
        aux_assignment: FieldShareSlice<'_, T, P>,
    ) -> R1CSResult<Proof<P>> {
        let c_acc_time = start_timer!(|| "Compute C");
        let h_acc = self.driver.msm_public_points(&pk.h_query, h);

        // Compute C
        let l_aux_acc = self.driver.msm_public_points(&pk.l_query, aux_assignment);

        // TODO define an error code
        let delta_g1 = pk.delta_g1.into_group();
        let rs = self.driver.mul(&r, &s).unwrap();
        let r_s_delta_g1 = self.driver.scalar_mul_public_point(&delta_g1, &rs);

        end_timer!(c_acc_time);

        // Compute A
        let a_acc_time = start_timer!(|| "Compute A");
        // TODO second time pk.delta_g1 transformed into group
        let r_g1 = self.driver.scalar_mul_public_point(&delta_g1, &r);

        let g_a = Self::calculate_coeff::<P::G1>(
            r_g1,
            &pk.a_query,
            pk.vk.alpha_g1,
            input_assignment,
            aux_assignment,
        );

        // Open here since g_a is part of proof
        // TODO define an error code
        let g_a_opened = EcMpcProtocol::<P::G1>::open_point(&mut self.driver, &g_a).unwrap();
        let s_g_a = self.driver.scalar_mul_public_point(&g_a_opened, &s);

        // Compute B in G1
        // In original implementation this is skipped if r==0, however r is shared in our case
        let b_g1_acc_time = start_timer!(|| "Compute B in G1");
        let s_g1 = self.driver.scalar_mul_public_point(&delta_g1, &s);
        let g1_b = Self::calculate_coeff::<P::G1>(
            s_g1,
            &pk.b_g1_query,
            pk.beta_g1,
            input_assignment,
            aux_assignment,
        );
        // TODO define an error code
        let r_g1_b = EcMpcProtocol::<P::G1>::scalar_mul(&mut self.driver, &g1_b, &r).unwrap();
        end_timer!(b_g1_acc_time);

        // Compute B in G2
        let delta_g2 = pk.vk.delta_g2.into_group();
        let b_g2_acc_time = start_timer!(|| "Compute B in G2");
        let s_g2 = self.driver.scalar_mul_public_point(&delta_g2, &s);
        let g2_b = Self::calculate_coeff::<P::G2>(
            s_g2,
            &pk.b_g2_query,
            pk.vk.beta_g2,
            input_assignment,
            aux_assignment,
        );
        end_timer!(b_g2_acc_time);

        let c_time = start_timer!(|| "Finish C");
        let mut g_c = s_g_a;
        EcMpcProtocol::<P::G1>::add_assign_points(&mut self.driver, &mut g_c, &r_g1_b);
        EcMpcProtocol::<P::G1>::sub_assign_points(&mut self.driver, &mut g_c, &r_s_delta_g1);
        EcMpcProtocol::<P::G1>::add_assign_points(&mut self.driver, &mut g_c, &l_aux_acc);
        EcMpcProtocol::<P::G1>::add_assign_points(&mut self.driver, &mut g_c, &h_acc);
        end_timer!(c_time);

        todo!()
    }
}
