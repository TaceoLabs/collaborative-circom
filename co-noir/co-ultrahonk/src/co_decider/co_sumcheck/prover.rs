use super::SumcheckOutput;
use crate::{
    co_decider::{
        co_sumcheck::round::SumcheckRound,
        prover::CoDecider,
        types::{ClaimedEvaluations, PartiallyEvaluatePolys, MAX_PARTIAL_RELATION_LENGTH},
    },
    types::AllEntities,
    FieldShare, CONST_PROOF_SIZE_LOG_N,
};
use mpc_core::traits::{MSMProvider, PrimeFieldMpcProtocol};
use ultrahonk::{
    prelude::{
        GateSeparatorPolynomial, HonkCurve, HonkProofResult, Transcript, TranscriptFieldType,
        TranscriptHasher, Univariate,
    },
    Utils,
};

// Keep in mind, the UltraHonk protocol (UltraFlavor) does not per default have ZK
impl<T, P: HonkCurve<TranscriptFieldType>, H: TranscriptHasher<TranscriptFieldType>>
    CoDecider<T, P, H>
where
    T: PrimeFieldMpcProtocol<P::ScalarField> + MSMProvider<P::G1>,
{
    pub(crate) fn partially_evaluate_init(
        driver: &mut T,
        partially_evaluated_poly: &mut PartiallyEvaluatePolys<T, P>,
        polys: &AllEntities<Vec<FieldShare<T, P>>, Vec<P::ScalarField>>,
        round_size: usize,
        round_challenge: &P::ScalarField,
    ) {
        tracing::trace!("Partially_evaluate init");

        // Barretenberg uses multithreading here

        for (poly_src, poly_des) in polys
            .public_iter()
            .zip(partially_evaluated_poly.public_iter_mut())
        {
            for i in (0..round_size).step_by(2) {
                poly_des[i >> 1] = poly_src[i] + (poly_src[i + 1] - poly_src[i]) * round_challenge;
            }
        }

        for (poly_src, poly_des) in polys
            .shared_iter()
            .zip(partially_evaluated_poly.shared_iter_mut())
        {
            for i in (0..round_size).step_by(2) {
                let tmp = driver.sub(&poly_src[i + 1], &poly_src[i]);
                let tmp = driver.mul_with_public(round_challenge, &tmp);
                poly_des[i >> 1] = driver.add(&poly_src[i], &tmp);
            }
        }
    }

    pub(crate) fn partially_evaluate_inplace(
        driver: &mut T,
        partially_evaluated_poly: &mut PartiallyEvaluatePolys<T, P>,
        round_size: usize,
        round_challenge: &P::ScalarField,
    ) {
        tracing::trace!("Partially_evaluate inplace");

        // Barretenberg uses multithreading here

        for poly in partially_evaluated_poly.public_iter_mut() {
            for i in (0..round_size).step_by(2) {
                poly[i >> 1] = poly[i] + (poly[i + 1] - poly[i]) * round_challenge;
            }
        }

        for poly in partially_evaluated_poly.shared_iter_mut() {
            for i in (0..round_size).step_by(2) {
                let tmp = driver.sub(&poly[i + 1], &poly[i]);
                let tmp = driver.mul_with_public(round_challenge, &tmp);
                poly[i >> 1] = driver.add(&poly[i], &tmp);
            }
        }
    }

    fn add_evals_to_transcript(
        transcript: &mut Transcript<TranscriptFieldType, H>,
        evaluations: &ClaimedEvaluations<P::ScalarField>,
    ) {
        tracing::trace!("Add Evals to Transcript");

        transcript.send_fr_iter_to_verifier::<P, _>(
            "Sumcheck:evaluations".to_string(),
            evaluations.iter(),
        );
    }

    fn extract_claimed_evaluations(
        driver: &mut T,
        partially_evaluated_polynomials: PartiallyEvaluatePolys<T, P>,
    ) -> HonkProofResult<ClaimedEvaluations<P::ScalarField>> {
        let mut multivariate_evaluations = ClaimedEvaluations::default();

        for (src, des) in partially_evaluated_polynomials
            .public_iter()
            .zip(multivariate_evaluations.public_iter_mut())
        {
            *des = src[0];
        }

        let shared = partially_evaluated_polynomials
            .into_shared_iter()
            .map(|x| x[0].to_owned())
            .collect::<Vec<_>>();

        let opened = driver.open_many(&shared).unwrap();

        for (src, des) in opened
            .into_iter()
            .zip(multivariate_evaluations.shared_iter_mut())
        {
            *des = src;
        }

        Ok(multivariate_evaluations)
    }

    pub(crate) fn sumcheck_prove(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        circuit_size: u32,
    ) -> HonkProofResult<SumcheckOutput<P::ScalarField>> {
        tracing::trace!("Sumcheck prove");

        let multivariate_n = circuit_size;
        let multivariate_d = Utils::get_msb64(multivariate_n as u64);

        let mut sum_check_round = SumcheckRound::new(multivariate_n as usize);

        let mut gate_separators = GateSeparatorPolynomial::new(
            self.memory.relation_parameters.gate_challenges.to_owned(),
            multivariate_d as usize,
        );

        let mut multivariate_challenge = Vec::with_capacity(multivariate_d as usize);
        let round_idx = 0;

        tracing::trace!("Sumcheck prove round {}", round_idx);

        // In the first round, we compute the first univariate polynomial and populate the book-keeping table of
        // #partially_evaluated_polynomials, which has \f$ n/2 \f$ rows and \f$ N \f$ columns. When the Flavor has ZK,
        // compute_univariate also takes into account the zk_sumcheck_data.
        let round_univariate = sum_check_round.compute_univariate::<T, P>(
            &mut self.driver,
            round_idx,
            &self.memory.relation_parameters,
            &gate_separators,
            &self.memory.polys,
        )?;
        let round_univariate = self.driver.open_many(&round_univariate.evaluations)?;

        // Place the evaluations of the round univariate into transcript.
        transcript.send_fr_iter_to_verifier::<P, _>(
            "Sumcheck:univariate_0".to_string(),
            &round_univariate,
        );
        let round_challenge = transcript.get_challenge::<P>("Sumcheck:u_0".to_string());
        multivariate_challenge.push(round_challenge);

        // Prepare sumcheck book-keeping table for the next round
        let mut partially_evaluated_polys =
            PartiallyEvaluatePolys::<T, P>::new(multivariate_n as usize >> 1);
        Self::partially_evaluate_init(
            &mut self.driver,
            &mut partially_evaluated_polys,
            &self.memory.polys,
            multivariate_n as usize,
            &round_challenge,
        );
        gate_separators.partially_evaluate(round_challenge);
        sum_check_round.round_size >>= 1; // AZTEC TODO(#224)(Cody): Maybe partially_evaluate should do this and
                                          // release memory?        // All but final round
                                          // We operate on partially_evaluated_polynomials in place.

        for round_idx in 1..multivariate_d as usize {
            tracing::trace!("Sumcheck prove round {}", round_idx);
            // Write the round univariate to the transcript

            let round_univariate = sum_check_round.compute_univariate::<T, P>(
                &mut self.driver,
                round_idx,
                &self.memory.relation_parameters,
                &gate_separators,
                &partially_evaluated_polys,
            )?;
            let round_univariate = self.driver.open_many(&round_univariate.evaluations)?;

            // Place the evaluations of the round univariate into transcript.
            transcript.send_fr_iter_to_verifier::<P, _>(
                format!("Sumcheck:univariate_{}", round_idx),
                &round_univariate,
            );
            let round_challenge =
                transcript.get_challenge::<P>(format!("Sumcheck:u_{}", round_idx));
            multivariate_challenge.push(round_challenge);
            // Prepare sumcheck book-keeping table for the next round
            Self::partially_evaluate_inplace(
                &mut self.driver,
                &mut partially_evaluated_polys,
                sum_check_round.round_size,
                &round_challenge,
            );
            gate_separators.partially_evaluate(round_challenge);
            sum_check_round.round_size >>= 1;
        }

        // Zero univariates are used to pad the proof to the fixed size CONST_PROOF_SIZE_LOG_N.
        let zero_univariate =
            Univariate::<P::ScalarField, { MAX_PARTIAL_RELATION_LENGTH + 1 }>::default();
        for idx in multivariate_d as usize..CONST_PROOF_SIZE_LOG_N {
            transcript.send_fr_iter_to_verifier::<P, _>(
                format!("Sumcheck:univariate_{}", idx),
                &zero_univariate.evaluations,
            );
            let round_challenge = transcript.get_challenge::<P>(format!("Sumcheck:u_{}", idx));
            multivariate_challenge.push(round_challenge);
        }

        // Claimed evaluations of Prover polynomials are extracted and added to the transcript. When Flavor has ZK, the
        // evaluations of all witnesses are masked.
        let multivariate_evaluations =
            Self::extract_claimed_evaluations(&mut self.driver, partially_evaluated_polys)?;
        Self::add_evals_to_transcript(transcript, &multivariate_evaluations);

        let res = SumcheckOutput {
            claimed_evaluations: multivariate_evaluations,
            challenges: multivariate_challenge,
        };
        Ok(res)
    }
}
