use crate::{
    co_decider::{prover::CoDecider, types::ProverMemory},
    co_oink::prover::CoOink,
    types::ProvingKey,
    CONST_PROOF_SIZE_LOG_N,
};
use ark_ec::pairing::Pairing;
use mpc_core::traits::{MSMProvider, PrimeFieldMpcProtocol};
use std::marker::PhantomData;
use ultrahonk::prelude::{
    HonkCurve, HonkProof, HonkProofResult, TranscriptFieldType, TranscriptType,
};

pub struct CoUltraHonk<T, P: HonkCurve<TranscriptFieldType>>
where
    T: PrimeFieldMpcProtocol<P::ScalarField> + MSMProvider<P::G1>,
{
    pub(crate) driver: T,
    phantom_data: PhantomData<P>,
}

impl<T, P: HonkCurve<TranscriptFieldType>> CoUltraHonk<T, P>
where
    T: PrimeFieldMpcProtocol<P::ScalarField> + MSMProvider<P::G1>,
{
    pub fn new(driver: T) -> Self {
        Self {
            driver,
            phantom_data: PhantomData,
        }
    }

    fn generate_gate_challenges(transcript: &mut TranscriptType) -> Vec<P::ScalarField> {
        tracing::trace!("generate gate challenges");

        let mut gate_challenges: Vec<<P as Pairing>::ScalarField> =
            Vec::with_capacity(CONST_PROOF_SIZE_LOG_N);

        for idx in 0..CONST_PROOF_SIZE_LOG_N {
            let chall = transcript.get_challenge::<P>(format!("Sumcheck:gate_challenge_{}", idx));
            gate_challenges.push(chall);
        }
        gate_challenges
    }

    pub fn prove(
        mut self,
        proving_key: ProvingKey<T, P>,
    ) -> HonkProofResult<HonkProof<TranscriptFieldType>> {
        tracing::trace!("CoUltraHonk prove");

        let mut transcript = TranscriptType::new();

        let oink = CoOink::new(&mut self.driver);
        let oink_result = oink.prove(&proving_key, &mut transcript)?;

        let cicruit_size = proving_key.circuit_size;
        let crs = proving_key.crs;

        let mut memory =
            ProverMemory::from_memory_and_polynomials(oink_result, proving_key.polynomials);
        memory.relation_parameters.gate_challenges =
            Self::generate_gate_challenges(&mut transcript);

        let decider = CoDecider::new(self.driver, memory);
        decider.prove(cicruit_size, &crs, transcript)
    }
}
