use super::types::VerifierMemory;
use crate::{
    oink::prover::Oink,
    prelude::{HonkCurve, TranscriptFieldType},
    transcript::{Transcript, TranscriptHasher},
    types::VerifyingKey,
    verifier::HonkVerifyResult,
};
use eyre::Context;

pub(crate) struct OinkVerifier<
    P: HonkCurve<TranscriptFieldType>,
    H: TranscriptHasher<TranscriptFieldType>,
> {
    memory: VerifierMemory<P>,
    pub public_inputs: Vec<P::ScalarField>,
    phantom_hasher: std::marker::PhantomData<H>,
}

impl<P: HonkCurve<TranscriptFieldType>, H: TranscriptHasher<TranscriptFieldType>> Default
    for OinkVerifier<P, H>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<P: HonkCurve<TranscriptFieldType>, H: TranscriptHasher<TranscriptFieldType>>
    OinkVerifier<P, H>
{
    pub(crate) fn new() -> Self {
        Self {
            memory: VerifierMemory::default(),
            public_inputs: Default::default(),
            phantom_hasher: Default::default(),
        }
    }

    fn execute_preamble_round(
        &mut self,
        verifying_key: &VerifyingKey<P>,
        transcript: &mut Transcript<TranscriptFieldType, H>,
    ) -> HonkVerifyResult<()> {
        tracing::trace!("executing (verifying) preamble round");

        let circuit_size = transcript
            .receive_u64_from_prover("circuit_size".to_string())
            .context("Failed to receive circuit_size")?;
        let public_input_size = transcript
            .receive_u64_from_prover("public_input_size".to_string())
            .context("Failed to receive public_input_size")?;
        let pub_inputs_offset = transcript
            .receive_u64_from_prover("pub_inputs_offset".to_string())
            .context("Failed to receive pub_inputs_offset")?;

        if circuit_size != verifying_key.circuit_size as u64 {
            return Err(eyre::eyre!("OinkVerifier::execute_preamble_round: proof circuit size does not match verification key!"));
        }

        if public_input_size != verifying_key.num_public_inputs as u64 {
            return Err(eyre::eyre!("OinkVerifier::execute_preamble_round: public inputs size does not match verification key!"));
        }

        if pub_inputs_offset != verifying_key.pub_inputs_offset as u64 {
            return Err(eyre::eyre!("OinkVerifier::execute_preamble_round: public inputs offset does not match verification key!"));
        }

        self.public_inputs = Vec::with_capacity(public_input_size as usize);

        for i in 0..public_input_size {
            let public_input =
                transcript.receive_fr_from_prover::<P>(format!("public_input_{}", i))?;
            self.public_inputs.push(public_input);
        }

        Ok(())
    }

    fn execute_wire_commitments_round(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
    ) -> HonkVerifyResult<()> {
        tracing::trace!("executing (verifying) wire commitments round");

        *self.memory.witness_commitments.w_l_mut() =
            transcript.receive_point_from_prover::<P>("W_L".to_string())?;
        *self.memory.witness_commitments.w_r_mut() =
            transcript.receive_point_from_prover::<P>("W_R".to_string())?;
        *self.memory.witness_commitments.w_o_mut() =
            transcript.receive_point_from_prover::<P>("W_O".to_string())?;

        // Round is done since ultra_honk is no goblin flavor
        Ok(())
    }

    fn execute_sorted_list_accumulator_round(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
    ) -> HonkVerifyResult<()> {
        tracing::trace!("executing (verifying) sorted list accumulator round");

        let challs = transcript.get_challenges::<P>(&[
            "eta".to_string(),
            "eta_two".to_string(),
            "eta_three".to_string(),
        ]);
        self.memory.challenges.eta_1 = challs[0];
        self.memory.challenges.eta_2 = challs[1];
        self.memory.challenges.eta_3 = challs[2];

        *self.memory.witness_commitments.lookup_read_counts_mut() =
            transcript.receive_point_from_prover::<P>("lookup_read_counts".to_string())?;

        *self.memory.witness_commitments.lookup_read_tags_mut() =
            transcript.receive_point_from_prover::<P>("lookup_read_tags".to_string())?;

        *self.memory.witness_commitments.w_4_mut() =
            transcript.receive_point_from_prover::<P>("w_4".to_string())?;

        Ok(())
    }

    fn execute_log_derivative_inverse_round(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
    ) -> HonkVerifyResult<()> {
        tracing::trace!("executing (verifying) log derivative inverse round");

        let challs = transcript.get_challenges::<P>(&["beta".to_string(), "gamma".to_string()]);
        self.memory.challenges.beta = challs[0];
        self.memory.challenges.gamma = challs[1];

        *self.memory.witness_commitments.lookup_inverses_mut() =
            transcript.receive_point_from_prover::<P>("lookup_inverses".to_string())?;

        // Round is done since ultra_honk is no goblin flavor
        Ok(())
    }

    fn execute_grand_product_computation_round(
        &mut self,
        verifying_key: &VerifyingKey<P>,
        transcript: &mut Transcript<TranscriptFieldType, H>,
    ) -> HonkVerifyResult<()> {
        tracing::trace!("executing (verifying) grand product computation round");
        self.memory.public_input_delta = Oink::<P, H>::compute_public_input_delta(
            &self.memory.challenges.beta,
            &self.memory.challenges.gamma,
            &self.public_inputs,
            verifying_key.circuit_size,
            verifying_key.pub_inputs_offset,
        );
        *self.memory.witness_commitments.z_perm_mut() =
            transcript.receive_point_from_prover::<P>("z_perm".to_string())?;
        Ok(())
    }

    pub(crate) fn verify(
        mut self,
        verifying_key: &VerifyingKey<P>,
        transcript: &mut Transcript<TranscriptFieldType, H>,
    ) -> HonkVerifyResult<VerifierMemory<P>> {
        tracing::trace!("Oink verify");
        self.execute_preamble_round(verifying_key, transcript)?;
        self.execute_wire_commitments_round(transcript)?;
        self.execute_sorted_list_accumulator_round(transcript)?;
        self.execute_log_derivative_inverse_round(transcript)?;
        self.execute_grand_product_computation_round(verifying_key, transcript)?;
        Oink::<P, H>::generate_alphas_round(&mut self.memory.challenges.alphas, transcript);
        Ok(self.memory)
    }
}
