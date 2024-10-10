use crate::{
    co_decider::{
        relations::{
            auxiliary_relation::AuxiliaryRelation,
            delta_range_constraint_relation::DeltaRangeConstraintRelation,
            elliptic_relation::EllipticRelation, logderiv_lookup_relation::LogDerivLookupRelation,
            permutation_relation::UltraPermutationRelation,
            poseidon2_external_relation::Poseidon2ExternalRelation,
            poseidon2_internal_relation::Poseidon2InternalRelation,
            ultra_arithmetic_relation::UltraArithmeticRelation, AllRelationAcc, Relation,
        },
        types::{ProverUnivariates, RelationParameters, MAX_PARTIAL_RELATION_LENGTH},
        univariates::SharedUnivariate,
    },
    mpc::NoirUltraHonkProver,
    types::AllEntities,
};
use ark_ec::pairing::Pairing;
use ark_ff::One;
use ultrahonk::prelude::{
    GateSeparatorPolynomial, HonkCurve, HonkProofResult, TranscriptFieldType, Univariate,
};

pub(crate) type SumcheckRoundOutput<T, P> =
    SharedUnivariate<T, P, { MAX_PARTIAL_RELATION_LENGTH + 1 }>;

pub(crate) struct SumcheckRound {
    pub(crate) round_size: usize,
}

impl SumcheckRound {
    pub(crate) fn new(initial_round_size: usize) -> Self {
        SumcheckRound {
            round_size: initial_round_size,
        }
    }

    fn extend_edges<T: NoirUltraHonkProver<P>, P: HonkCurve<TranscriptFieldType>>(
        driver: &mut T,
        extended_edges: &mut ProverUnivariates<T, P>,
        multivariates: &AllEntities<Vec<T::ArithmeticShare>, Vec<P::ScalarField>>,
        edge_index: usize,
    ) {
        tracing::trace!("Extend edges");
        for (src, des) in multivariates
            .public_iter()
            .zip(extended_edges.public_iter_mut())
        {
            des.extend_from(&src[edge_index..edge_index + 2]);
        }
        for (src, des) in multivariates
            .shared_iter()
            .zip(extended_edges.shared_iter_mut())
        {
            des.extend_from(driver, &src[edge_index..edge_index + 2]);
        }
    }

    /**
     * @brief Extend Univariates then sum them multiplying by the current \f$ pow_{\beta} \f$-contributions.
     * @details Since the sub-relations comprising full Honk relation are of different degrees, the computation of the
     * evaluations of round univariate \f$ \tilde{S}_{i}(X_{i}) \f$ at points \f$ X_{i} = 0,\ldots, D \f$ requires to
     * extend evaluations of individual relations to the domain \f$ 0,\ldots, D\f$. Moreover, linearly independent
     * sub-relations, i.e. whose validity is being checked at every point of the hypercube, are multiplied by the
     * constant \f$ c_i = pow_\beta(u_0,\ldots, u_{i-1}) \f$ and the current \f$pow_{\beta}\f$-factor \f$ ( (1−X_i) +
     * X_i\cdot \beta_i ) \vert_{X_i = k} \f$ for \f$ k = 0,\ldots, D\f$.
     * @tparam extended_size Size after extension
     * @param tuple A tuple of tuples of Univariates
     * @param result Round univariate \f$ \tilde{S}^i\f$ represented by its evaluations over \f$ \{0,\ldots, D\} \f$.
     * @param gate_sparators Round \f$pow_{\beta}\f$-factor  \f$ ( (1−X_i) + X_i\cdot \beta_i )\f$.
     */
    fn extend_and_batch_univariates<T: NoirUltraHonkProver<P>, P: Pairing>(
        driver: &mut T,
        result: &mut SumcheckRoundOutput<T, P>,
        univariate_accumulators: AllRelationAcc<T, P>,
        gate_sparators: &GateSeparatorPolynomial<P::ScalarField>,
    ) {
        // Pow-Factor  \f$ (1-X) + X\beta_i \f$
        let random_polynomial = [P::ScalarField::one(), gate_sparators.current_element()];
        let mut extended_random_polynomial = Univariate::default();
        extended_random_polynomial.extend_from(&random_polynomial);

        univariate_accumulators.extend_and_batch_univariates(
            driver,
            result,
            &extended_random_polynomial,
            &gate_sparators.partial_evaluation_result,
        )
    }

    /**
     * @brief Given a tuple of tuples of extended per-relation contributions,  \f$ (t_0, t_1, \ldots,
     * t_{\text{NUM_SUBRELATIONS}-1}) \f$ and a challenge \f$ \alpha \f$, scale them by the relation separator
     * \f$\alpha\f$, extend to the correct degree, and take the sum multiplying by \f$pow_{\beta}\f$-contributions.
     *
     * @details This method receives as input the univariate accumulators computed by \ref
     * accumulate_relation_univariates "accumulate relation univariates" after passing through the entire hypercube and
     * applying \ref bb::RelationUtils::add_nested_tuples "add_nested_tuples" method to join the threads. The
     * accumulators are scaled using the method \ref bb::RelationUtils< Flavor >::scale_univariates "scale univariates",
     * extended to the degree \f$ D \f$ and summed with appropriate  \f$pow_{\beta}\f$-factors using \ref
     * extend_and_batch_univariates "extend and batch univariates method" to return a vector \f$(\tilde{S}^i(0), \ldots,
     * \tilde{S}^i(D))\f$.
     *
     * @param challenge Challenge \f$\alpha\f$.
     * @param gate_sparators Round \f$pow_{\beta}\f$-factor given by  \f$ ( (1−u_i) + u_i\cdot \beta_i )\f$.
     */
    fn batch_over_relations_univariates<T: NoirUltraHonkProver<P>, P: Pairing>(
        driver: &mut T,
        mut univariate_accumulators: AllRelationAcc<T, P>,
        alphas: &[P::ScalarField; crate::NUM_ALPHAS],
        gate_sparators: &GateSeparatorPolynomial<P::ScalarField>,
    ) -> SumcheckRoundOutput<T, P> {
        tracing::trace!("batch over relations");

        let running_challenge = P::ScalarField::one();
        univariate_accumulators.scale(driver, running_challenge, alphas);

        let mut res = SumcheckRoundOutput::default();
        Self::extend_and_batch_univariates(
            driver,
            &mut res,
            univariate_accumulators,
            gate_sparators,
        );
        res
    }

    async fn accumulate_one_relation_univariates<
        T: NoirUltraHonkProver<P>,
        P: HonkCurve<TranscriptFieldType>,
        R: Relation<T, P>,
    >(
        driver: &mut T,
        univariate_accumulator: &mut R::Acc,
        extended_edges: &ProverUnivariates<T, P>,
        relation_parameters: &RelationParameters<P::ScalarField>,
        scaling_factor: &P::ScalarField,
    ) -> HonkProofResult<()> {
        if R::SKIPPABLE && R::skip(extended_edges) {
            return Ok(());
        }

        R::accumulate(
            driver,
            univariate_accumulator,
            extended_edges,
            relation_parameters,
            scaling_factor,
        )
        .await
    }

    async fn accumulate_relation_univariates<
        T: NoirUltraHonkProver<P>,
        P: HonkCurve<TranscriptFieldType>,
    >(
        driver: &mut T,
        univariate_accumulators: &mut AllRelationAcc<T, P>,
        extended_edges: &ProverUnivariates<T, P>,
        relation_parameters: &RelationParameters<P::ScalarField>,
        scaling_factor: &P::ScalarField,
    ) -> HonkProofResult<()> {
        tracing::trace!("Accumulate relations");
        Self::accumulate_one_relation_univariates::<_, _, UltraArithmeticRelation>(
            driver,
            &mut univariate_accumulators.r_arith,
            extended_edges,
            relation_parameters,
            scaling_factor,
        )
        .await?;
        Self::accumulate_one_relation_univariates::<_, _, UltraPermutationRelation>(
            driver,
            &mut univariate_accumulators.r_perm,
            extended_edges,
            relation_parameters,
            scaling_factor,
        )
        .await?;
        Self::accumulate_one_relation_univariates::<_, _, DeltaRangeConstraintRelation>(
            driver,
            &mut univariate_accumulators.r_delta,
            extended_edges,
            relation_parameters,
            scaling_factor,
        )
        .await?;
        Self::accumulate_one_relation_univariates::<_, _, EllipticRelation>(
            driver,
            &mut univariate_accumulators.r_elliptic,
            extended_edges,
            relation_parameters,
            scaling_factor,
        )
        .await?;
        Self::accumulate_one_relation_univariates::<_, _, AuxiliaryRelation>(
            driver,
            &mut univariate_accumulators.r_aux,
            extended_edges,
            relation_parameters,
            scaling_factor,
        )
        .await?;
        Self::accumulate_one_relation_univariates::<_, _, LogDerivLookupRelation>(
            driver,
            &mut univariate_accumulators.r_lookup,
            extended_edges,
            relation_parameters,
            scaling_factor,
        )
        .await?;
        Self::accumulate_one_relation_univariates::<_, _, Poseidon2ExternalRelation>(
            driver,
            &mut univariate_accumulators.r_pos_ext,
            extended_edges,
            relation_parameters,
            scaling_factor,
        )
        .await?;
        Self::accumulate_one_relation_univariates::<_, _, Poseidon2InternalRelation>(
            driver,
            &mut univariate_accumulators.r_pos_int,
            extended_edges,
            relation_parameters,
            scaling_factor,
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn compute_univariate<
        T: NoirUltraHonkProver<P>,
        P: HonkCurve<TranscriptFieldType>,
    >(
        &self,
        driver: &mut T,
        round_index: usize,
        relation_parameters: &RelationParameters<P::ScalarField>,
        gate_sparators: &GateSeparatorPolynomial<P::ScalarField>,
        polynomials: &AllEntities<Vec<T::ArithmeticShare>, Vec<P::ScalarField>>,
    ) -> HonkProofResult<SumcheckRoundOutput<T, P>> {
        tracing::trace!("Sumcheck round {}", round_index);

        // Barretenberg uses multithreading here

        // Construct extended edge containers
        let mut extended_edge = ProverUnivariates::default();

        let mut univariate_accumulators = AllRelationAcc::<T, P>::default();

        // Accumulate the contribution from each sub-relation accross each edge of the hyper-cube
        for edge_idx in (0..self.round_size).step_by(2) {
            Self::extend_edges(driver, &mut extended_edge, polynomials, edge_idx);
            // Compute the \f$ \ell \f$-th edge's univariate contribution,
            // scale it by the corresponding \f$ pow_{\beta} \f$ contribution and add it to the accumulators for \f$
            // \tilde{S}^i(X_i) \f$. If \f$ \ell \f$'s binary representation is given by \f$ (\ell_{i+1},\ldots,
            // \ell_{d-1})\f$, the \f$ pow_{\beta}\f$-contribution is \f$\beta_{i+1}^{\ell_{i+1}} \cdot \ldots \cdot
            // \beta_{d-1}^{\ell_{d-1}}\f$.
            Self::accumulate_relation_univariates(
                driver,
                &mut univariate_accumulators,
                &extended_edge,
                relation_parameters,
                &gate_sparators.beta_products[(edge_idx >> 1) * gate_sparators.periodicity],
            )
            .await?;
        }
        let res = Self::batch_over_relations_univariates(
            driver,
            univariate_accumulators,
            &relation_parameters.alphas,
            gate_sparators,
        );
        Ok(res)
    }
}
