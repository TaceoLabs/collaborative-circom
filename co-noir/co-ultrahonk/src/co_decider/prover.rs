use super::{
    co_sumcheck::SumcheckOutput, co_zeromorph::ZeroMorphOpeningClaim, types::ProverMemory,
};
use crate::CoUtils;
use mpc_core::traits::{MSMProvider, PrimeFieldMpcProtocol};
use std::marker::PhantomData;
use ultrahonk::prelude::{
    HonkCurve, HonkProof, HonkProofResult, ProverCrs, Transcript, TranscriptFieldType,
    TranscriptHasher,
};

pub(crate) struct CoDecider<
    T,
    P: HonkCurve<TranscriptFieldType>,
    H: TranscriptHasher<TranscriptFieldType>,
> where
    T: PrimeFieldMpcProtocol<P::ScalarField> + MSMProvider<P::G1>,
{
    pub(crate) driver: T,
    pub(super) memory: ProverMemory<T, P>,
    phantom_data: PhantomData<P>,
    phantom_hasher: PhantomData<H>,
}

impl<T, P: HonkCurve<TranscriptFieldType>, H: TranscriptHasher<TranscriptFieldType>>
    CoDecider<T, P, H>
where
    T: PrimeFieldMpcProtocol<P::ScalarField> + MSMProvider<P::G1>,
{
    pub fn new(driver: T, memory: ProverMemory<T, P>) -> Self {
        Self {
            driver,
            memory,
            phantom_data: PhantomData,
            phantom_hasher: PhantomData,
        }
    }

    fn compute_opening_proof(
        driver: &mut T,
        opening_claim: ZeroMorphOpeningClaim<T, P>,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        crs: &ProverCrs<P>,
    ) -> HonkProofResult<()> {
        let mut quotient = opening_claim.polynomial;
        let pair = opening_claim.opening_pair;

        quotient[0] = driver.add_with_public(&-pair.evaluation, &quotient[0]);
        // Computes the coefficients for the quotient polynomial q(X) = (p(X) - v) / (X - r) through an FFT
        quotient.factor_roots(driver, &pair.challenge);
        let quotient_commitment = CoUtils::commit(driver, &quotient.coefficients, crs);
        // AZTEC TODO(#479): for now we compute the KZG commitment directly to unify the KZG and IPA interfaces but in the
        // future we might need to adjust this to use the incoming alternative to work queue (i.e. variation of
        // pthreads) or even the work queue itself
        let quotient_commitment = driver.open_point(&quotient_commitment)?;
        transcript.send_point_to_verifier::<P>("KZG:W".to_string(), quotient_commitment.into());
        Ok(())
    }

    /**
     * @brief Run Sumcheck to establish that ∑_i pow(\vec{β*})f_i(ω) = e*. This results in u = (u_1,...,u_d) sumcheck round
     * challenges and all evaluations at u being calculated.
     *
     */
    fn execute_relation_check_rounds(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        circuit_size: u32,
    ) -> HonkProofResult<SumcheckOutput<P::ScalarField>> {
        // This is just Sumcheck.prove
        self.sumcheck_prove(transcript, circuit_size)
    }

    /**
     * @brief Execute the ZeroMorph protocol to produce an opening claim for the multilinear evaluations produced by
     * Sumcheck and then produce an opening proof with a univariate PCS.
     * @details See https://hackmd.io/dlf9xEwhTQyE3hiGbq4FsA?view for a complete description of the unrolled protocol.
     *
     * */
    fn execute_pcs_rounds(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        circuit_size: u32,
        crs: &ProverCrs<P>,
        sumcheck_output: SumcheckOutput<P::ScalarField>,
    ) -> HonkProofResult<()> {
        let prover_opening_claim =
            self.zeromorph_prove(transcript, circuit_size, crs, sumcheck_output)?;
        Self::compute_opening_proof(&mut self.driver, prover_opening_claim, transcript, crs)
    }

    pub(crate) fn prove(
        mut self,
        circuit_size: u32,
        crs: &ProverCrs<P>,
        mut transcript: Transcript<TranscriptFieldType, H>,
    ) -> HonkProofResult<HonkProof<TranscriptFieldType>> {
        tracing::trace!("Decider prove");

        // Run sumcheck subprotocol.
        let sumcheck_output = self.execute_relation_check_rounds(&mut transcript, circuit_size)?;

        // Fiat-Shamir: rho, y, x, z
        // Execute Zeromorph multilinear PCS
        self.execute_pcs_rounds(&mut transcript, circuit_size, crs, sumcheck_output)?;

        Ok(transcript.get_proof())
    }
}
