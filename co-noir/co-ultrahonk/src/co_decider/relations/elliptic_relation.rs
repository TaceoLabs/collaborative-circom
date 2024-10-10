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
use ultrahonk::prelude::{HonkCurve, HonkProofResult, TranscriptFieldType, Univariate};

#[derive(Clone, Debug)]
pub(crate) struct EllipticRelationAcc<T: NoirUltraHonkProver<P>, P: Pairing> {
    pub(crate) r0: SharedUnivariate<T, P, 6>,
    pub(crate) r1: SharedUnivariate<T, P, 6>,
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> Default for EllipticRelationAcc<T, P> {
    fn default() -> Self {
        Self {
            r0: Default::default(),
            r1: Default::default(),
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> EllipticRelationAcc<T, P> {
    pub(crate) fn scale(&mut self, driver: &mut T, elements: &[P::ScalarField]) {
        assert!(elements.len() == EllipticRelation::NUM_RELATIONS);
        self.r0.scale_inplace(driver, elements[0]);
        self.r1.scale_inplace(driver, elements[1]);
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
    }
}

pub(crate) struct EllipticRelation {}

impl EllipticRelation {
    pub(crate) const NUM_RELATIONS: usize = 2;
    pub(crate) const CRAND_PAIRS_FACTOR: usize = 12;
}

impl<T: NoirUltraHonkProver<P>, P: HonkCurve<TranscriptFieldType>> Relation<T, P>
    for EllipticRelation
{
    type Acc = EllipticRelationAcc<T, P>;
    const SKIPPABLE: bool = true;

    fn skip(input: &ProverUnivariates<T, P>) -> bool {
        <Self as Relation<T, P>>::check_skippable();
        input.precomputed.q_elliptic().is_zero()
    }

    /**
     * @brief Expression for the Ultra Arithmetic gate.
     * @details The relation is defined as C(in(X)...) =
     *   AZTEC TODO(#429): steal description from elliptic_widget.hpp
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
        tracing::trace!("Accumulate EllipticRelation");

        // AZTEC TODO(@zac - williamson #2608 when Pedersen refactor is completed,
        // replace old addition relations with these ones and
        // remove endomorphism coefficient in ecc add gate(not used))

        let x_1 = input.witness.w_r();
        let y_1 = input.witness.w_o();

        let x_2 = input.shifted_witness.w_l();
        let y_2 = input.shifted_witness.w_4();
        let y_3 = input.shifted_witness.w_o();
        let x_3 = input.shifted_witness.w_r();

        let q_sign = input.precomputed.q_l();
        let q_elliptic = input.precomputed.q_elliptic();
        let q_is_double = input.precomputed.q_m();

        // First round of multiplications
        let x_diff = x_2.sub(driver, x_1);
        let y1_plus_y3 = y_1.add(driver, y_3);
        let y_diff = y_2.mul_public(driver, q_sign).sub(driver, y_1);
        let x1_mul_3 = x_1.add(driver, x_1).add(driver, x_1);

        let lhs = SharedUnivariate::univariates_to_vec(&[
            y_1.to_owned(),
            y_2.to_owned(),
            y_1.to_owned(),
            x_diff.to_owned(),
            y1_plus_y3.to_owned(),
            y_diff,
            x1_mul_3.to_owned(),
        ]);
        let rhs = SharedUnivariate::univariates_to_vec(&[
            y_1.to_owned(),
            y_2.to_owned(),
            y_2.to_owned(),
            x_diff.to_owned(),
            x_diff,
            x_3.sub(driver, x_1),
            x_1.to_owned(),
        ]);
        let mul1 = driver.mul_many(&lhs, &rhs).await?;
        let mul1 = SharedUnivariate::vec_to_univariates(&mul1);

        // Second round of multiplications
        let curve_b = P::get_curve_b(); // here we need the extra constraint on the Curve
        let y1_sqr = &mul1[0];
        let y1_sqr_mul_4 = y1_sqr.add(driver, y1_sqr);
        let y1_sqr_mul_4 = y1_sqr_mul_4.add(driver, &y1_sqr_mul_4);
        let x1_sqr_mul_3 = &mul1[6];

        let lhs = SharedUnivariate::univariates_to_vec(&[
            x_3.add(driver, x_2).add(driver, x_1),
            y1_sqr.sub_scalar(driver, curve_b),
            x_3.add(driver, x_1).add(driver, x_1),
            x1_sqr_mul_3.to_owned(),
            y_1.add(driver, y_1),
        ]);
        let rhs = SharedUnivariate::univariates_to_vec(&[
            mul1[3].to_owned(),
            x1_mul_3,
            y1_sqr_mul_4,
            x_1.sub(driver, x_3),
            y1_plus_y3,
        ]);
        let mul2 = driver.mul_many(&lhs, &rhs).await?;
        let mul2 = SharedUnivariate::vec_to_univariates(&mul2);

        // Contribution (1) point addition, x-coordinate check
        // q_elliptic * (x3 + x2 + x1)(x2 - x1)(x2 - x1) - y2^2 - y1^2 + 2(y2y1)*q_sign = 0
        let y2_sqr = &mul1[1];
        let y1y2 = mul1[2].mul_public(driver, q_sign);
        let x_add_identity = mul2[0]
            .sub(driver, y2_sqr)
            .sub(driver, y1_sqr)
            .add(driver, &y1y2)
            .add(driver, &y1y2);

        let q_elliptic_by_scaling = q_elliptic.to_owned() * scaling_factor;
        let q_elliptic_q_double_scaling = q_elliptic_by_scaling.to_owned() * q_is_double;
        let q_elliptic_not_double_scaling = q_elliptic_by_scaling - &q_elliptic_q_double_scaling;
        let tmp_1 = x_add_identity.mul_public(driver, &q_elliptic_not_double_scaling);

        ///////////////////////////////////////////////////////////////////////
        // Contribution (2) point addition, x-coordinate check
        // q_elliptic * (q_sign * y1 + y3)(x2 - x1) + (x3 - x1)(y2 - q_sign * y1) = 0
        let y_add_identity = &mul1[4].add(driver, &mul1[5]);
        let tmp_2 = y_add_identity.mul_public(driver, &q_elliptic_not_double_scaling);

        ///////////////////////////////////////////////////////////////////////
        // Contribution (3) point doubling, x-coordinate check
        // (x3 + x1 + x1) (4y1*y1) - 9 * x1 * x1 * x1 * x1 = 0
        // N.B. we're using the equivalence x1*x1*x1 === y1*y1 - curve_b to reduce degree by 1

        let x_pow_4_mul_3 = &mul2[1];
        let x1_pow_4_mul_9 = x_pow_4_mul_3
            .add(driver, x_pow_4_mul_3)
            .add(driver, x_pow_4_mul_3);
        let x_double_identity = mul2[2].sub(driver, &x1_pow_4_mul_9);

        let tmp = x_double_identity.mul_public(driver, &q_elliptic_q_double_scaling);
        let tmp_1 = tmp_1.add(driver, &tmp);

        ///////////////////////////////////////////////////////////////////////
        // Contribution (4) point doubling, y-coordinate check
        // (y1 + y1) (2y1) - (3 * x1 * x1)(x1 - x3) = 0
        let y_double_identity = mul2[3].sub(driver, &mul2[4]);
        let tmp = y_double_identity.mul_public(driver, &q_elliptic_q_double_scaling);
        let tmp_2 = tmp_2.add(driver, &tmp);

        ///////////////////////////////////////////////////////////////////////

        for i in 0..univariate_accumulator.r0.evaluations.len() {
            univariate_accumulator.r0.evaluations[i] = driver.add(
                univariate_accumulator.r0.evaluations[i],
                tmp_1.evaluations[i],
            );
        }

        for i in 0..univariate_accumulator.r1.evaluations.len() {
            univariate_accumulator.r1.evaluations[i] = driver.add(
                univariate_accumulator.r1.evaluations[i],
                tmp_2.evaluations[i],
            );
        }

        Ok(())
    }
}
