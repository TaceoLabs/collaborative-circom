use super::Relation;
use crate::{
    co_decider::{
        types::{ProverUnivariates, RelationParameters},
        univariates::SharedUnivariate,
    },
    mpc::NoirUltraHonkProver,
};
use ark_ec::pairing::Pairing;
use ark_ff::Zero;
use num_bigint::BigUint;
use ultrahonk::prelude::{
    HonkCurve, HonkProofResult, TranscriptFieldType, Univariate, POSEIDON2_BN254_T4_PARAMS,
};

#[derive(Clone, Debug)]
pub(crate) struct Poseidon2InternalRelationAcc<T: NoirUltraHonkProver<P>, P: Pairing> {
    pub(crate) r0: SharedUnivariate<T, P, 7>,
    pub(crate) r1: SharedUnivariate<T, P, 7>,
    pub(crate) r2: SharedUnivariate<T, P, 7>,
    pub(crate) r3: SharedUnivariate<T, P, 7>,
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> Default for Poseidon2InternalRelationAcc<T, P> {
    fn default() -> Self {
        Self {
            r0: Default::default(),
            r1: Default::default(),
            r2: Default::default(),
            r3: Default::default(),
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> Poseidon2InternalRelationAcc<T, P> {
    pub(crate) fn scale(&mut self, driver: &mut T, elements: &[P::ScalarField]) {
        assert!(elements.len() == Poseidon2InternalRelation::NUM_RELATIONS);
        self.r0.scale_inplace(driver, elements[0]);
        self.r1.scale_inplace(driver, elements[1]);
        self.r2.scale_inplace(driver, elements[2]);
        self.r3.scale_inplace(driver, elements[3]);
    }

    pub(crate) fn extend_and_batch_univariates<const SIZE: usize>(
        &self,
        driver: &mut T,
        result: &mut SharedUnivariate<T, P, SIZE>,
        extended_random_poly: &Univariate<P::ScalarField, SIZE>,
        partial_evaluation_result: &P::ScalarField,
    ) {
        self.r0.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
            true,
        );

        self.r1.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
            true,
        );

        self.r2.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
            true,
        );

        self.r3.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
            true,
        );
    }
}

pub(crate) struct Poseidon2InternalRelation {}

impl Poseidon2InternalRelation {
    pub(crate) const NUM_RELATIONS: usize = 4;
    pub(crate) const CRAND_PAIRS_FACTOR: usize = 3;
}

impl<T: NoirUltraHonkProver<P>, P: HonkCurve<TranscriptFieldType>> Relation<T, P>
    for Poseidon2InternalRelation
{
    type Acc = Poseidon2InternalRelationAcc<T, P>;
    const SKIPPABLE: bool = true;

    fn skip(input: &ProverUnivariates<T, P>) -> bool {
        <Self as Relation<T, P>>::check_skippable();
        input.precomputed.q_poseidon2_internal().is_zero()
    }

    /**
     * @brief Expression for the poseidon2 internal round relation, based on I_i in Section 6 of
     * https://eprint.iacr.org/2023/323.pdf.
     * @details This relation is defined as C(in(X)...) :=
     * q_poseidon2_internal * ( (v1 - w_1_shift) + \alpha * (v2 - w_2_shift) +
     * \alpha^2 * (v3 - w_3_shift) + \alpha^3 * (v4 - w_4_shift) ) = 0 where:
     *      u1 := (w_1 + q_1)^5
     *      sum := u1 + w_2 + w_3 + w_4
     *      v1 := u1 * D1 + sum
     *      v2 := w_2 * D2 + sum
     *      v3 := w_3 * D3 + sum
     *      v4 := w_4 * D4 + sum
     *      Di is the ith internal diagonal value - 1 of the internal matrix M_I
     *
     * @param evals transformed to `evals + C(in(X)...)*scaling_factor`
     * @param in an std::array containing the fully extended Univariate edges.
     * @param parameters contains beta, gamma, and public_input_delta, ....
     * @param scaling_factor optional term to scale the evaluation before adding to evals.
     */
    async fn accumulate(
        driver: &mut T,
        univariate_accumulator: &mut Self::Acc,
        input: &ProverUnivariates<T, P>,
        _relation_parameters: &RelationParameters<P::ScalarField>,
        scaling_factor: &P::ScalarField,
    ) -> HonkProofResult<()> {
        tracing::trace!("Accumulate Poseidon2InternalRelation");

        let w_l = input.witness.w_l();
        let w_r = input.witness.w_r();
        let w_o = input.witness.w_o();
        let w_4 = input.witness.w_4();
        let w_l_shift = input.shifted_witness.w_l();
        let w_r_shift = input.shifted_witness.w_r();
        let w_o_shift = input.shifted_witness.w_o();
        let w_4_shift = input.shifted_witness.w_4();
        let q_l = input.precomputed.q_l();
        let q_poseidon2_internal = input.precomputed.q_poseidon2_internal();

        // add round constants
        let s1 = w_l.add_public(driver, q_l);

        // apply s-box round
        let u1 = driver.mul_many(s1.as_ref(), s1.as_ref()).await?;
        let u1 = driver.mul_many(u1.as_ref(), u1.as_ref()).await?;
        let u1 = driver.mul_many(u1.as_ref(), s1.as_ref()).await?;
        let mut u2 = w_r.to_owned();
        let mut u3 = w_o.to_owned();
        let mut u4 = w_4.to_owned();
        let mut u1 = SharedUnivariate::from_vec(&u1);

        // matrix mul with v = M_I * u 4 muls and 7 additions
        let sum = u1.add(driver, &u2).add(driver, &u3).add(driver, &u4);

        let q_pos_by_scaling = q_poseidon2_internal.to_owned() * scaling_factor;

        // TACEO TODO this poseidon instance is very hardcoded to the bn254 curve
        let internal_matrix_diag_0 = P::ScalarField::from(BigUint::from(
            POSEIDON2_BN254_T4_PARAMS.mat_internal_diag_m_1[0],
        ));
        let internal_matrix_diag_1 = P::ScalarField::from(BigUint::from(
            POSEIDON2_BN254_T4_PARAMS.mat_internal_diag_m_1[1],
        ));
        let internal_matrix_diag_2 = P::ScalarField::from(BigUint::from(
            POSEIDON2_BN254_T4_PARAMS.mat_internal_diag_m_1[2],
        ));
        let internal_matrix_diag_3 = P::ScalarField::from(BigUint::from(
            POSEIDON2_BN254_T4_PARAMS.mat_internal_diag_m_1[3],
        ));

        u1.scale_inplace(driver, internal_matrix_diag_0);
        let v1 = u1.add(driver, &sum);
        let tmp = v1
            .sub(driver, w_l_shift)
            .mul_public(driver, &q_pos_by_scaling);

        for i in 0..univariate_accumulator.r0.evaluations.len() {
            univariate_accumulator.r0.evaluations[i] =
                driver.add(univariate_accumulator.r0.evaluations[i], tmp.evaluations[i]);
        }

        ///////////////////////////////////////////////////////////////////////

        u2.scale_inplace(driver, internal_matrix_diag_1);
        let v2 = u2.add(driver, &sum);
        let tmp = v2
            .sub(driver, w_r_shift)
            .mul_public(driver, &q_pos_by_scaling);

        for i in 0..univariate_accumulator.r1.evaluations.len() {
            univariate_accumulator.r1.evaluations[i] =
                driver.add(univariate_accumulator.r1.evaluations[i], tmp.evaluations[i]);
        }

        ///////////////////////////////////////////////////////////////////////

        u3.scale_inplace(driver, internal_matrix_diag_2);
        let v3 = u3.add(driver, &sum);
        let tmp = v3
            .sub(driver, w_o_shift)
            .mul_public(driver, &q_pos_by_scaling);

        for i in 0..univariate_accumulator.r2.evaluations.len() {
            univariate_accumulator.r2.evaluations[i] =
                driver.add(univariate_accumulator.r2.evaluations[i], tmp.evaluations[i]);
        }

        ///////////////////////////////////////////////////////////////////////
        u4.scale_inplace(driver, internal_matrix_diag_3);
        let v4 = u4.add(driver, &sum);
        let tmp = v4
            .sub(driver, w_4_shift)
            .mul_public(driver, &q_pos_by_scaling);

        for i in 0..univariate_accumulator.r3.evaluations.len() {
            univariate_accumulator.r3.evaluations[i] =
                driver.add(univariate_accumulator.r3.evaluations[i], tmp.evaluations[i]);
        }

        Ok(())
    }
}
