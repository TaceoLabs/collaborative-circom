use super::{types::VerifierMemory, zeromorph::ZeroMorphVerifierOpeningClaim};
use crate::{
    prelude::{HonkCurve, TranscriptFieldType},
    transcript::{Transcript, TranscriptHasher},
    verifier::HonkVerifyResult,
};
use ark_ec::AffineRepr;
use ark_ff::One;
use std::marker::PhantomData;

pub(crate) struct DeciderVerifier<
    P: HonkCurve<TranscriptFieldType>,
    H: TranscriptHasher<TranscriptFieldType>,
> {
    pub(super) memory: VerifierMemory<P>,
    phantom_data: PhantomData<P>,
    phantom_hasher: PhantomData<H>,
}

impl<P: HonkCurve<TranscriptFieldType>, H: TranscriptHasher<TranscriptFieldType>>
    DeciderVerifier<P, H>
{
    pub(crate) fn new(memory: VerifierMemory<P>) -> Self {
        Self {
            memory,
            phantom_data: PhantomData,
            phantom_hasher: PhantomData,
        }
    }

    // this is the KZG one:
    // Note: The pairing check can be expressed naturally as
    // e(C - v * [1]_1, [1]_2) = e([W]_1, [X - r]_2) where C =[p(X)]_1. This can be rearranged (e.g. see the plonk
    // paper) as e(C + r*[W]_1 - v*[1]_1, [1]_2) * e(-[W]_1, [X]_2) = 1, or e(P_0, [1]_2) * e(P_1, [X]_2) = 1
    pub(crate) fn reduce_verify(
        opening_pair: ZeroMorphVerifierOpeningClaim<P>,
        mut transcript: Transcript<TranscriptFieldType, H>,
    ) -> HonkVerifyResult<(P::G1Affine, P::G1Affine)> {
        tracing::trace!("Reduce and verify opening pair");

        let g1_affine = P::G1Affine::generator();
        let g1_projective: P::G1 = g1_affine.into_group();

        let quotient_commitment = transcript.receive_point_from_prover::<P>("KZG:W".to_string())?;

        let p_1 = -P::G1::from(quotient_commitment);
        let p_0 = opening_pair.commitment;
        let first = quotient_commitment.into_group() * opening_pair.challenge;
        let second = g1_projective * opening_pair.evaluation;
        let p_0 = p_0 + first;
        let p_0 = p_0 - second;
        Ok((p_0.into(), p_1.into()))
    }

    pub fn pairing_check(
        p0: P::G1Affine,
        p1: P::G1Affine,
        g2_x: P::G2Affine,
        g2_gen: P::G2Affine,
    ) -> bool {
        tracing::trace!("Pairing check");
        let p = [g2_gen, g2_x];
        let g1_prepared = [P::G1Prepared::from(p0), P::G1Prepared::from(p1)];
        P::multi_pairing(g1_prepared, p).0 == P::TargetField::one()
    }

    pub(crate) fn verify(
        mut self,
        circuit_size: u32,
        crs: &P::G2Affine,
        mut transcript: Transcript<TranscriptFieldType, H>,
    ) -> HonkVerifyResult<bool> {
        tracing::trace!("Decider verification");

        let sumcheck_output = self.sumcheck_verify(&mut transcript, circuit_size)?;
        if !sumcheck_output.verified {
            tracing::trace!("Sumcheck failed");
            return Ok(false);
        }

        let opening_claim = self.zeromorph_verify(
            &mut transcript,
            circuit_size,
            sumcheck_output.multivariate_challenge,
        )?;
        let pairing_points = Self::reduce_verify(opening_claim, transcript)?;
        let pcs_verified = Self::pairing_check(
            pairing_points.0,
            pairing_points.1,
            *crs,
            P::G2Affine::generator(),
        );
        Ok(sumcheck_output.verified && pcs_verified)
    }
}
