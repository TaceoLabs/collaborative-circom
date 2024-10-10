use super::Relation;
use crate::{
    co_decider::{
        types::{ProverUnivariates, RelationParameters, MAX_PARTIAL_RELATION_LENGTH},
        univariates::SharedUnivariate,
    },
    mpc::NoirUltraHonkProver,
};
use ark_ec::pairing::Pairing;
use ultrahonk::prelude::{HonkCurve, HonkProofResult, TranscriptFieldType, Univariate};

#[derive(Clone, Debug)]
pub(crate) struct UltraPermutationRelationAcc<T: NoirUltraHonkProver<P>, P: Pairing> {
    pub(crate) r0: SharedUnivariate<T, P, 6>,
    pub(crate) r1: SharedUnivariate<T, P, 3>,
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> Default for UltraPermutationRelationAcc<T, P> {
    fn default() -> Self {
        Self {
            r0: Default::default(),
            r1: Default::default(),
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> UltraPermutationRelationAcc<T, P> {
    pub(crate) fn scale(&mut self, driver: &mut T, elements: &[P::ScalarField]) {
        assert!(elements.len() == UltraPermutationRelation::NUM_RELATIONS);
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

pub(crate) struct UltraPermutationRelation {}

impl UltraPermutationRelation {
    pub(crate) const NUM_RELATIONS: usize = 2;
    pub(crate) const CRAND_PAIRS_FACTOR: usize = 8;
}

impl UltraPermutationRelation {
    async fn compute_grand_product_numerator_and_denominator<
        T: NoirUltraHonkProver<P>,
        P: Pairing,
    >(
        driver: &mut T,
        input: &ProverUnivariates<T, P>,
        relation_parameters: &RelationParameters<P::ScalarField>,
    ) -> HonkProofResult<Vec<T::ArithmeticShare>> {
        let w_1 = input.witness.w_l();
        let w_2 = input.witness.w_r();
        let w_3 = input.witness.w_o();
        let w_4 = input.witness.w_4();
        let id_1 = input.precomputed.id_1();
        let id_2 = input.precomputed.id_2();
        let id_3 = input.precomputed.id_3();
        let id_4 = input.precomputed.id_4();
        let sigma_1 = input.precomputed.sigma_1();
        let sigma_2 = input.precomputed.sigma_2();
        let sigma_3 = input.precomputed.sigma_3();
        let sigma_4 = input.precomputed.sigma_4();

        let beta = &relation_parameters.beta;
        let gamma = &relation_parameters.gamma;

        // witness degree 4; full degree 8
        let wid1 = w_1.add_public(driver, &(id_1.to_owned() * beta + gamma));
        let wid2 = w_2.add_public(driver, &(id_2.to_owned() * beta + gamma));
        let wid3 = w_3.add_public(driver, &(id_3.to_owned() * beta + gamma));
        let wid4 = w_4.add_public(driver, &(id_4.to_owned() * beta + gamma));

        let wsigma1 = w_1.add_public(driver, &(sigma_1.to_owned() * beta + gamma));
        let wsigma2 = w_2.add_public(driver, &(sigma_2.to_owned() * beta + gamma));
        let wsigma3 = w_3.add_public(driver, &(sigma_3.to_owned() * beta + gamma));
        let wsigma4 = w_4.add_public(driver, &(sigma_4.to_owned() * beta + gamma));

        let lhs = SharedUnivariate::univariates_to_vec(&[wid1, wsigma1, wid3, wsigma3]);
        let rhs = SharedUnivariate::univariates_to_vec(&[wid2, wsigma2, wid4, wsigma4]);
        let mul1 = driver.mul_many(&lhs, &rhs).await?;
        let (lhs, rhs) = mul1.split_at(mul1.len() >> 1);
        let mul2 = driver.mul_many(lhs, rhs).await?;
        // We need the result as input to the mul operations
        // let (num, den) = mul2.split_at(mul2.len() >> 1);
        // let num = SharedUnivariate::from_vec(num);
        // let den = SharedUnivariate::from_vec(den);
        // Ok((num, den))
        Ok(mul2)
    }
}

impl<T: NoirUltraHonkProver<P>, P: HonkCurve<TranscriptFieldType>> Relation<T, P>
    for UltraPermutationRelation
{
    type Acc = UltraPermutationRelationAcc<T, P>;
    const SKIPPABLE: bool = false;

    fn skip(_input: &ProverUnivariates<T, P>) -> bool {
        <Self as Relation<T, P>>::check_skippable();
        // Cannot skip because z_perm and z_perm_shift are secret-shared
        false
    }

    /**
    * @brief Compute contribution of the permutation relation for a given edge (internal function)
    *
    * @details This relation confirms faithful calculation of the grand
    * product polynomial \f$ Z_{\text{perm}}\f$.
    * In Sumcheck Prover Round, this method adds to accumulators evaluations of subrelations at the point
       \f$(u_0,\ldots, u_{i-1}, k, \vec\ell)\f$ for \f$ k=0,\ldots, D\f$, where \f$ \vec \ell\f$ is a point  on the
       Boolean hypercube \f$\{0,1\}^{d-1-i}\f$ and \f$ D \f$ is specified by the calling class. It does so by taking as
       input an array of Prover Polynomials partially evaluated at the points \f$(u_0,\ldots, u_{i-1}, k, \vec\ell)\f$ and
       computing point-wise evaluations of the sub-relations. \todo Protogalaxy Accumulation
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
        relation_parameters: &RelationParameters<P::ScalarField>,
        scaling_factor: &P::ScalarField,
    ) -> HonkProofResult<()> {
        tracing::trace!("Accumulate UltraPermutationRelation");

        let public_input_delta = &relation_parameters.public_input_delta;
        let z_perm = input.witness.z_perm();
        let z_perm_shift = input.shifted_witness.z_perm();
        let lagrange_first = input.precomputed.lagrange_first();
        let lagrange_last = input.precomputed.lagrange_last();

        // witness degree: deg 5 - deg 5 = deg 5
        // total degree: deg 9 - deg 10 = deg 10

        let num_den = Self::compute_grand_product_numerator_and_denominator(
            driver,
            input,
            relation_parameters,
        )
        .await?;

        let tmp_lhs = z_perm.add_public(driver, lagrange_first);
        let tmp_rhs =
            z_perm_shift.add_public(driver, &(lagrange_last.to_owned() * public_input_delta));

        let lhs = num_den;
        let rhs = SharedUnivariate::univariates_to_vec(&[tmp_lhs, tmp_rhs]);
        let mul1 = driver.mul_many(&lhs, &rhs).await?;
        let (lhs, rhs) = mul1.split_at(mul1.len() >> 1);
        let lhs = SharedUnivariate::<T, P, MAX_PARTIAL_RELATION_LENGTH>::from_vec(lhs);
        let rhs = SharedUnivariate::<T, P, MAX_PARTIAL_RELATION_LENGTH>::from_vec(rhs);

        let tmp = lhs.sub(driver, &rhs).scale(driver, *scaling_factor);

        for i in 0..univariate_accumulator.r0.evaluations.len() {
            univariate_accumulator.r0.evaluations[i] =
                driver.add(univariate_accumulator.r0.evaluations[i], tmp.evaluations[i]);
        }

        ///////////////////////////////////////////////////////////////////////

        let tmp = z_perm_shift
            .mul_public(driver, lagrange_last)
            .scale(driver, *scaling_factor);

        for i in 0..univariate_accumulator.r1.evaluations.len() {
            univariate_accumulator.r1.evaluations[i] =
                driver.add(univariate_accumulator.r1.evaluations[i], tmp.evaluations[i]);
        }

        Ok(())
    }
}
