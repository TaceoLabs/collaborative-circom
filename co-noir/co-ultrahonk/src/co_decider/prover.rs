use super::{
    co_sumcheck::SumcheckOutput, co_zeromorph::ZeroMorphOpeningClaim, types::ProverMemory,
};
use crate::{mpc::NoirUltraHonkProver, CoUtils};
use std::marker::PhantomData;
use ultrahonk::prelude::{
    HonkCurve, HonkProof, HonkProofResult, ProverCrs, TranscriptFieldType, TranscriptType,
};

pub(crate) struct CoDecider<T: NoirUltraHonkProver<P>, P: HonkCurve<TranscriptFieldType>> {
    pub(crate) driver: T,
    pub(super) memory: ProverMemory<T, P>,
    phantom_data: PhantomData<P>,
}

impl<T: NoirUltraHonkProver<P>, P: HonkCurve<TranscriptFieldType>> CoDecider<T, P> {
    pub fn new(driver: T, memory: ProverMemory<T, P>) -> Self {
        Self {
            driver,
            memory,
            phantom_data: PhantomData,
        }
    }

    async fn compute_opening_proof(
        driver: &mut T,
        opening_claim: ZeroMorphOpeningClaim<T, P>,
        transcript: &mut TranscriptType,
        crs: &ProverCrs<P>,
    ) -> HonkProofResult<()> {
        let mut quotient = opening_claim.polynomial;
        let pair = opening_claim.opening_pair;

        quotient[0] = driver.add_with_public(-pair.evaluation, quotient[0]);
        // Computes the coefficients for the quotient polynomial q(X) = (p(X) - v) / (X - r) through an FFT
        quotient.factor_roots(driver, &pair.challenge);
        let quotient_commitment = CoUtils::commit::<T, P>(&quotient.coefficients, crs);
        // AZTEC TODO(#479): for now we compute the KZG commitment directly to unify the KZG and IPA interfaces but in the
        // future we might need to adjust this to use the incoming alternative to work queue (i.e. variation of
        // pthreads) or even the work queue itself
        let quotient_commitment = driver.open_point(quotient_commitment).await?;
        transcript.send_point_to_verifier::<P>("KZG:W".to_string(), quotient_commitment.into());
        Ok(())
    }

    /**
     * @brief Run Sumcheck to establish that ∑_i pow(\vec{β*})f_i(ω) = e*. This results in u = (u_1,...,u_d) sumcheck round
     * challenges and all evaluations at u being calculated.
     *
     */
    async fn execute_relation_check_rounds(
        &mut self,
        transcript: &mut TranscriptType,
        circuit_size: u32,
    ) -> HonkProofResult<SumcheckOutput<P::ScalarField>> {
        // This is just Sumcheck.prove
        self.sumcheck_prove(transcript, circuit_size).await
    }

    /**
     * @brief Execute the ZeroMorph protocol to produce an opening claim for the multilinear evaluations produced by
     * Sumcheck and then produce an opening proof with a univariate PCS.
     * @details See https://hackmd.io/dlf9xEwhTQyE3hiGbq4FsA?view for a complete description of the unrolled protocol.
     *
     * */
    async fn execute_pcs_rounds(
        &mut self,
        transcript: &mut TranscriptType,
        circuit_size: u32,
        crs: &ProverCrs<P>,
        sumcheck_output: SumcheckOutput<P::ScalarField>,
    ) -> HonkProofResult<()> {
        let prover_opening_claim = self
            .zeromorph_prove(transcript, circuit_size, crs, sumcheck_output)
            .await?;
        Self::compute_opening_proof(&mut self.driver, prover_opening_claim, transcript, crs).await
    }

    pub(crate) async fn prove(
        mut self,
        circuit_size: u32,
        crs: &ProverCrs<P>,
        mut transcript: TranscriptType,
    ) -> HonkProofResult<HonkProof<TranscriptFieldType>> {
        tracing::trace!("Decider prove");

        // Run sumcheck subprotocol.
        let sumcheck_output = self
            .execute_relation_check_rounds(&mut transcript, circuit_size)
            .await?;

        // Fiat-Shamir: rho, y, x, z
        // Execute Zeromorph multilinear PCS
        self.execute_pcs_rounds(&mut transcript, circuit_size, crs, sumcheck_output)
            .await?;

        Ok(transcript.get_proof())
    }
}
