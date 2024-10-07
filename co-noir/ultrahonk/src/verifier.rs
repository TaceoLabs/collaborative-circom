use crate::{
    decider::{types::VerifierMemory, verifier::DeciderVerifier},
    oink::verifier::OinkVerifier,
    prelude::{HonkCurve, TranscriptFieldType, TranscriptType},
    prover::UltraHonk,
    types::{HonkProof, VerifyingKey},
};

pub(crate) type HonkVerifyResult<T> = std::result::Result<T, eyre::Report>;

impl<P: HonkCurve<TranscriptFieldType>> UltraHonk<P> {
    pub fn verify(
        honk_proof: HonkProof<TranscriptFieldType>,
        verifying_key: VerifyingKey<P>,
    ) -> HonkVerifyResult<bool> {
        tracing::trace!("UltraHonk verification");

        let mut transcript = TranscriptType::new_verifier(honk_proof);

        let oink_verifier = OinkVerifier::default();
        let oink_result = oink_verifier.verify(&verifying_key, &mut transcript)?;

        let cicruit_size = verifying_key.circuit_size;
        let crs = verifying_key.crs;

        let mut memory = VerifierMemory::from_memory_and_key(oink_result, verifying_key);
        memory.relation_parameters.gate_challenges =
            Self::generate_gate_challenges(&mut transcript);

        let decider_verifier = DeciderVerifier::new(memory);
        decider_verifier.verify(cicruit_size, &crs, transcript)
    }
}
