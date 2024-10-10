use super::Relation;
use crate::{
    co_decider::{
        types::{ProverUnivariates, RelationParameters, MAX_PARTIAL_RELATION_LENGTH},
        univariates::SharedUnivariate,
    },
    mpc::NoirUltraHonkProver,
};
use ark_ec::pairing::Pairing;
use ark_ff::{One, Zero};
use ultrahonk::prelude::{HonkCurve, HonkProofResult, TranscriptFieldType, Univariate};

#[derive(Clone, Debug)]
pub(crate) struct DeltaRangeConstraintRelationAcc<T: NoirUltraHonkProver<P>, P: Pairing> {
    pub(crate) r0: SharedUnivariate<T, P, 6>,
    pub(crate) r1: SharedUnivariate<T, P, 6>,
    pub(crate) r2: SharedUnivariate<T, P, 6>,
    pub(crate) r3: SharedUnivariate<T, P, 6>,
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> Default for DeltaRangeConstraintRelationAcc<T, P> {
    fn default() -> Self {
        Self {
            r0: Default::default(),
            r1: Default::default(),
            r2: Default::default(),
            r3: Default::default(),
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> DeltaRangeConstraintRelationAcc<T, P> {
    pub(crate) fn scale(&mut self, driver: &mut T, elements: &[P::ScalarField]) {
        assert!(elements.len() == DeltaRangeConstraintRelation::NUM_RELATIONS);
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

pub(crate) struct DeltaRangeConstraintRelation {}

impl DeltaRangeConstraintRelation {
    pub(crate) const NUM_RELATIONS: usize = 4;
    pub(crate) const CRAND_PAIRS_FACTOR: usize = 12;
}

impl<T: NoirUltraHonkProver<P>, P: HonkCurve<TranscriptFieldType>> Relation<T, P>
    for DeltaRangeConstraintRelation
{
    type Acc = DeltaRangeConstraintRelationAcc<T, P>;
    const SKIPPABLE: bool = true;

    fn skip(input: &ProverUnivariates<T, P>) -> bool {
        <Self as Relation<T, P>>::check_skippable();
        input.precomputed.q_delta_range().is_zero()
    }

    /**
     * @brief Expression for the generalized permutation sort gate.
     * @details The relation is defined as C(in(X)...) =
     *    q_delta_range * \sum{ i = [0, 3]} \alpha^i D_i(D_i - 1)(D_i - 2)(D_i - 3)
     *      where
     *      D_0 = w_2 - w_1
     *      D_1 = w_3 - w_2
     *      D_2 = w_4 - w_3
     *      D_3 = w_1_shift - w_4
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
        tracing::trace!("Accumulate DeltaRangeConstraintRelation");

        let w_1 = input.witness.w_l();
        let w_2 = input.witness.w_r();
        let w_3 = input.witness.w_o();
        let w_4 = input.witness.w_4();
        let w_1_shift = input.shifted_witness.w_l();
        let q_delta_range = input.precomputed.q_delta_range();
        let minus_one = -P::ScalarField::one();
        let minus_two = -P::ScalarField::from(2u64);

        // Compute wire differences
        let delta_1 = w_2.sub(driver, w_1);
        let delta_2 = w_3.sub(driver, w_2);
        let delta_3 = w_4.sub(driver, w_3);
        let delta_4 = w_1_shift.sub(driver, w_4);

        let tmp_1 = delta_1.add_scalar(driver, minus_one);
        let tmp_2 = delta_2.add_scalar(driver, minus_one);
        let tmp_3 = delta_3.add_scalar(driver, minus_one);
        let tmp_4 = delta_4.add_scalar(driver, minus_one);
        let tmp_1_2 = delta_1.add_scalar(driver, minus_two);
        let tmp_2_2 = delta_2.add_scalar(driver, minus_two);
        let tmp_3_2 = delta_3.add_scalar(driver, minus_two);
        let tmp_4_2 = delta_4.add_scalar(driver, minus_two);

        let lhs = SharedUnivariate::univariates_to_vec(&[
            tmp_1, tmp_2, tmp_3, tmp_4, tmp_1_2, tmp_2_2, tmp_3_2, tmp_4_2,
        ]);
        let mut sqr = driver.mul_many(&lhs, &lhs).await?;

        for el in sqr.iter_mut() {
            *el = driver.add_with_public(minus_one, *el);
        }

        let (lhs, rhs) = sqr.split_at(sqr.len() >> 1);
        let mul = driver.mul_many(lhs, rhs).await?;
        let mul = SharedUnivariate::<T, P, MAX_PARTIAL_RELATION_LENGTH>::vec_to_univariates(&mul);

        // Contribution (1)
        let mut tmp = mul[0].mul_public(driver, q_delta_range);
        tmp.scale_inplace(driver, *scaling_factor);

        for i in 0..univariate_accumulator.r0.evaluations.len() {
            univariate_accumulator.r0.evaluations[i] =
                driver.add(univariate_accumulator.r0.evaluations[i], tmp.evaluations[i]);
        }

        ///////////////////////////////////////////////////////////////////////
        // Contribution (2)
        let mut tmp = mul[1].mul_public(driver, q_delta_range);
        tmp.scale_inplace(driver, *scaling_factor);

        for i in 0..univariate_accumulator.r1.evaluations.len() {
            univariate_accumulator.r1.evaluations[i] =
                driver.add(univariate_accumulator.r1.evaluations[i], tmp.evaluations[i]);
        }

        ///////////////////////////////////////////////////////////////////////
        // Contribution (3)
        let mut tmp = mul[2].mul_public(driver, q_delta_range);
        tmp.scale_inplace(driver, *scaling_factor);

        for i in 0..univariate_accumulator.r2.evaluations.len() {
            univariate_accumulator.r2.evaluations[i] =
                driver.add(univariate_accumulator.r2.evaluations[i], tmp.evaluations[i]);
        }

        ///////////////////////////////////////////////////////////////////////
        // Contribution (4)
        let mut tmp = mul[3].mul_public(driver, q_delta_range);
        tmp.scale_inplace(driver, *scaling_factor);

        for i in 0..univariate_accumulator.r3.evaluations.len() {
            univariate_accumulator.r3.evaluations[i] =
                driver.add(univariate_accumulator.r3.evaluations[i], tmp.evaluations[i]);
        }

        Ok(())
    }
}
