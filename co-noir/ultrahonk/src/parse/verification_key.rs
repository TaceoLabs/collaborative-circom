use super::{
    builder::{GenericUltraCircuitBuilder, UltraCircuitBuilder, UltraCircuitVariable},
    types::{AggregationObjectPubInputIndices, AGGREGATION_OBJECT_SIZE},
};
use crate::{
    prelude::{CrsParser, HonkCurve, TranscriptFieldType},
    prover::{HonkProofError, HonkProofResult},
    types::{
        Crs, HonkProof, PrecomputedEntities, ProverCrs, ProvingKey, VerifyingKey,
        PRECOMPUTED_ENTITIES_SIZE,
    },
    Serialize, Utils,
};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ff::PrimeField;
use eyre::Result;

impl<P: Pairing> VerifyingKey<P> {
    pub fn create(circuit: UltraCircuitBuilder<P>, crs: Crs<P>) -> HonkProofResult<Self> {
        let (_, vk) = circuit.create_keys(crs)?;
        Ok(vk)
    }

    pub fn from_barrettenberg_and_crs(
        barretenberg_vk: VerifyingKeyBarretenberg<P>,
        crs: P::G2Affine,
    ) -> Self {
        Self {
            crs,
            circuit_size: barretenberg_vk.circuit_size as u32,
            num_public_inputs: barretenberg_vk.num_public_inputs as u32,
            pub_inputs_offset: barretenberg_vk.pub_inputs_offset as u32,
            commitments: barretenberg_vk.commitments,
        }
    }

    pub fn get_crs<S: UltraCircuitVariable<P::ScalarField>>(
        circuit: &GenericUltraCircuitBuilder<P, S>,
        path_g1: &str,
        path_g2: &str,
    ) -> Result<Crs<P>> {
        tracing::info!("Getting crs");
        ProvingKey::get_crs(circuit, path_g1, path_g2)
    }

    pub fn get_prover_crs<S: UltraCircuitVariable<P::ScalarField>>(
        circuit: &GenericUltraCircuitBuilder<P, S>,
        path_g1: &str,
    ) -> Result<ProverCrs<P>> {
        tracing::info!("Getting crs");
        ProvingKey::get_prover_crs(circuit, path_g1)
    }

    pub fn get_verifier_crs(path_g2: &str) -> Result<P::G2Affine> {
        tracing::info!("Getting verifier crs");
        CrsParser::<P>::get_crs_g2(path_g2)
    }
}

pub struct VerifyingKeyBarretenberg<P: Pairing> {
    pub(crate) circuit_size: u64,
    pub(crate) log_circuit_size: u64,
    pub(crate) num_public_inputs: u64,
    pub(crate) pub_inputs_offset: u64,
    pub(crate) contains_recursive_proof: bool,
    pub(crate) recursive_proof_public_input_indices: AggregationObjectPubInputIndices,
    pub(crate) commitments: PrecomputedEntities<P::G1Affine>,
}

impl<P: HonkCurve<TranscriptFieldType>> VerifyingKeyBarretenberg<P> {
    const NUM_64_LIMBS: u32 = P::BaseField::MODULUS_BIT_SIZE.div_ceil(64);
    const FIELDSIZE_BYTES: u32 = Self::NUM_64_LIMBS * 8;
    const SER_SIZE: usize = 4 * 8
        + 1
        + AGGREGATION_OBJECT_SIZE * 4
        + PRECOMPUTED_ENTITIES_SIZE * 2 * Self::FIELDSIZE_BYTES as usize;

    fn write_g1_element(buf: &mut Vec<u8>, el: &P::G1Affine, write_x_first: bool) {
        let prev_len = buf.len();

        if el.is_zero() {
            for _ in 0..Self::FIELDSIZE_BYTES * 2 {
                buf.push(255);
            }
        } else {
            let (x, y) = P::g1_affine_to_xy(el);
            if write_x_first {
                Serialize::<P::BaseField>::write_field_element(buf, x);
                Serialize::<P::BaseField>::write_field_element(buf, y);
            } else {
                Serialize::<P::BaseField>::write_field_element(buf, y);
                Serialize::<P::BaseField>::write_field_element(buf, x);
            }
        }

        debug_assert_eq!(buf.len() - prev_len, Self::FIELDSIZE_BYTES as usize * 2);
    }

    fn read_g1_element(buf: &[u8], offset: &mut usize, read_x_first: bool) -> P::G1Affine {
        if buf.iter().all(|&x| x == 255) {
            *offset += Self::FIELDSIZE_BYTES as usize * 2;
            return P::G1Affine::zero();
        }

        let first = Serialize::<P::BaseField>::read_field_element(buf, offset);
        let second = Serialize::<P::BaseField>::read_field_element(buf, offset);

        if read_x_first {
            P::g1_affine_from_xy(first, second)
        } else {
            P::g1_affine_from_xy(second, first)
        }
    }

    pub fn to_buffer(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(Self::SER_SIZE);

        Serialize::<P::ScalarField>::write_u64(&mut buffer, self.circuit_size);
        Serialize::<P::ScalarField>::write_u64(&mut buffer, self.log_circuit_size);
        Serialize::<P::ScalarField>::write_u64(&mut buffer, self.num_public_inputs);
        Serialize::<P::ScalarField>::write_u64(&mut buffer, self.pub_inputs_offset);
        Serialize::<P::ScalarField>::write_u8(&mut buffer, self.contains_recursive_proof as u8);

        for val in self.recursive_proof_public_input_indices.iter() {
            Serialize::<P::ScalarField>::write_u32(&mut buffer, *val);
        }

        for el in self.commitments.iter() {
            Self::write_g1_element(&mut buffer, el, true);
        }

        debug_assert_eq!(buffer.len(), Self::SER_SIZE);
        buffer
    }

    pub fn from_buffer(buf: &[u8]) -> HonkProofResult<Self> {
        let size = buf.len();
        let mut offset = 0;

        if size != Self::SER_SIZE {
            return Err(HonkProofError::InvalidKeyLength);
        }

        // Read data
        let circuit_size = Serialize::<P::ScalarField>::read_u64(buf, &mut offset);
        let log_circuit_size = Serialize::<P::ScalarField>::read_u64(buf, &mut offset);
        if log_circuit_size != Utils::get_msb64(circuit_size) as u64 {
            return Err(HonkProofError::CorruptedKey);
        }
        let num_public_inputs = Serialize::<P::ScalarField>::read_u64(buf, &mut offset);
        let pub_inputs_offset = Serialize::<P::ScalarField>::read_u64(buf, &mut offset);
        let contains_recursive_proof_u8 = Serialize::<P::ScalarField>::read_u8(buf, &mut offset);
        if contains_recursive_proof_u8 > 1 {
            return Err(HonkProofError::CorruptedKey);
        }
        let contains_recursive_proof = contains_recursive_proof_u8 == 1;

        let mut recursive_proof_public_input_indices = AggregationObjectPubInputIndices::default();
        for val in recursive_proof_public_input_indices.iter_mut() {
            *val = Serialize::<P::ScalarField>::read_u32(buf, &mut offset);
        }

        let mut commitments = PrecomputedEntities::default();

        for el in commitments.iter_mut() {
            *el = Self::read_g1_element(buf, &mut offset, true);
        }

        debug_assert_eq!(offset, Self::SER_SIZE);

        Ok(Self {
            circuit_size,
            log_circuit_size,
            num_public_inputs,
            pub_inputs_offset,
            contains_recursive_proof,
            recursive_proof_public_input_indices,
            commitments,
        })
    }
}
