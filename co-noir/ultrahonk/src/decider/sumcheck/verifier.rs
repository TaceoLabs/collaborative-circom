use super::SumcheckVerifierOutput;
use crate::{
    decider::{
        sumcheck::{round_prover::SumcheckRoundOutput, round_verifier::SumcheckVerifierRound},
        types::MAX_PARTIAL_RELATION_LENGTH,
        verifier::DeciderVerifier,
    },
    prelude::{GateSeparatorPolynomial, HonkCurve, TranscriptFieldType},
    transcript::{Transcript, TranscriptHasher},
    types::NUM_ALL_ENTITIES,
    verifier::HonkVerifyResult,
    Utils, CONST_PROOF_SIZE_LOG_N,
};

// Keep in mind, the UltraHonk protocol (UltraFlavor) does not per default have ZK
impl<P: HonkCurve<TranscriptFieldType>, H: TranscriptHasher<TranscriptFieldType>>
    DeciderVerifier<P, H>
{
    pub(crate) fn sumcheck_verify(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        circuit_size: u32,
    ) -> HonkVerifyResult<SumcheckVerifierOutput<P::ScalarField>> {
        tracing::trace!("Sumcheck verify");

        let mut verified: bool = true;

        let multivariate_n = circuit_size;
        let multivariate_d = Utils::get_msb64(multivariate_n as u64);

        let mut gate_separators = GateSeparatorPolynomial::new_without_products(
            self.memory.relation_parameters.gate_challenges.to_owned(),
        );

        if multivariate_d == 0 {
            return Err(eyre::eyre!("Number of variables in multivariate is 0"));
        }

        let mut sum_check_round = SumcheckVerifierRound::<P>::default();

        let mut multivariate_challenge = Vec::with_capacity(multivariate_d as usize);

        for round_idx in 0..CONST_PROOF_SIZE_LOG_N {
            tracing::trace!("Sumcheck verify round {}", round_idx);
            let round_univariate_label = format!("Sumcheck:univariate_{}", round_idx);

            let evaluations = transcript
                .receive_fr_array_from_verifier::<P, { MAX_PARTIAL_RELATION_LENGTH + 1 }>(
                    round_univariate_label,
                )?;
            let round_univariate = SumcheckRoundOutput { evaluations };

            let round_challenge =
                transcript.get_challenge::<P>(format!("Sumcheck:u_{}", round_idx));

            // No recursive flavor, otherwise we need to make some modifications to the following
            if round_idx < multivariate_d as usize {
                let checked = sum_check_round.check_sum(&round_univariate);
                verified = verified && checked;

                multivariate_challenge.push(round_challenge);

                sum_check_round.compute_next_target_sum(&round_univariate, round_challenge);
                gate_separators.partially_evaluate(round_challenge);
            } else {
                multivariate_challenge.push(round_challenge);
            }
        }

        // Final round
        let transcript_evaluations = transcript.receive_fr_vec_from_verifier::<P>(
            "Sumcheck:evaluations".to_string(),
            NUM_ALL_ENTITIES,
        )?;

        for (eval, &transcript_eval) in self
            .memory
            .claimed_evaluations
            .iter_mut()
            .zip(transcript_evaluations.iter())
        {
            *eval = transcript_eval;
        }

        // Evaluate the Honk relation at the point (u_0, ..., u_{d-1}) using claimed evaluations of prover polynomials.

        let full_honk_purported_value =
            SumcheckVerifierRound::<P>::compute_full_relation_purported_value(
                &self.memory.claimed_evaluations,
                &self.memory.relation_parameters,
                gate_separators,
            );

        let checked = full_honk_purported_value == sum_check_round.target_total_sum;
        verified = verified && checked;

        Ok(SumcheckVerifierOutput {
            multivariate_challenge,
            verified,
        })
    }
}
