use super::types::{PolyF, PolyG, PolyGShift};
use crate::{
    decider::{
        types::{ClaimedEvaluations, VerifierCommitments},
        verifier::DeciderVerifier,
        zeromorph::ZeroMorphVerifierOpeningClaim,
    },
    prelude::{HonkCurve, TranscriptFieldType},
    transcript::{Transcript, TranscriptHasher},
    verifier::HonkVerifyResult,
    Utils, CONST_PROOF_SIZE_LOG_N,
};
use ark_ec::AffineRepr;
use ark_ff::{Field, One, Zero};

impl<P: HonkCurve<TranscriptFieldType>, H: TranscriptHasher<TranscriptFieldType>>
    DeciderVerifier<P, H>
{
    pub fn get_g_shift_evaluations(
        evaluations: &ClaimedEvaluations<P::ScalarField>,
    ) -> PolyGShift<P::ScalarField> {
        PolyGShift {
            tables: &evaluations.shifted_tables,
            wires: &evaluations.shifted_witness,
        }
    }

    pub fn get_g_shift_comms(evaluations: &VerifierCommitments<P::G1Affine>) -> PolyG<P::G1Affine> {
        PolyG {
            tables: evaluations
                .precomputed
                .get_table_polynomials()
                .try_into()
                .unwrap(),
            wires: evaluations.witness.to_be_shifted().try_into().unwrap(),
        }
    }

    pub fn get_f_evaluations(
        evaluations: &ClaimedEvaluations<P::ScalarField>,
    ) -> PolyF<P::ScalarField> {
        PolyF {
            precomputed: &evaluations.precomputed,
            witness: &evaluations.witness,
        }
    }
    pub fn get_f_comms(evaluations: &ClaimedEvaluations<P::G1Affine>) -> PolyF<P::G1Affine> {
        PolyF {
            precomputed: &evaluations.precomputed,
            witness: &evaluations.witness,
        }
    }

    pub(crate) fn zeromorph_verify(
        &self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        circuit_size: u32,
        multivariate_challenge: Vec<P::ScalarField>,
    ) -> HonkVerifyResult<ZeroMorphVerifierOpeningClaim<P>> {
        tracing::trace!("Zeromorph verify");

        let unshifted_evaluations: PolyF<P::ScalarField> =
            Self::get_f_evaluations(&self.memory.claimed_evaluations);
        let shifted_evaluations: PolyGShift<P::ScalarField> =
            Self::get_g_shift_evaluations(&self.memory.claimed_evaluations);
        let rho = transcript.get_challenge::<P>("rho".to_string());

        let mut batched_evaluation = P::ScalarField::zero();
        let mut batching_scalar = P::ScalarField::one();

        for &value in unshifted_evaluations
            .iter()
            .chain(shifted_evaluations.iter())
        {
            batched_evaluation += value * batching_scalar;
            batching_scalar *= rho;
        }

        let mut c_q_k = Vec::with_capacity(CONST_PROOF_SIZE_LOG_N);
        for i in 0..CONST_PROOF_SIZE_LOG_N {
            c_q_k.push(transcript.receive_point_from_prover::<P>(format!("ZM:C_q_{}", i))?);
        }

        let y_challenge = transcript.get_challenge::<P>("y_challenge".to_string());

        // Receive commitment C_{q}
        //  auto c_q = transcript->template receive_from_prover<Commitment>("ZM:C_q");
        let c_q = transcript.receive_point_from_prover::<P>("ZM:C_q".to_string())?;

        let x_challenge = transcript.get_challenge::<P>("x_challenge".to_string());

        let z_challenge = transcript.get_challenge::<P>("z_challenge".to_string());

        let c_zeta_x = Self::compute_c_zeta_x(c_q, &c_q_k, y_challenge, x_challenge, circuit_size)?;

        let c_z_x = self.compute_c_z_x(
            c_q_k,
            rho,
            batched_evaluation,
            x_challenge,
            multivariate_challenge,
            circuit_size,
        )?;

        let c_zeta_z = c_zeta_x + c_z_x * z_challenge;

        Ok(ZeroMorphVerifierOpeningClaim {
            challenge: x_challenge,
            evaluation: P::ScalarField::ZERO,
            commitment: c_zeta_z,
        })
    }

    // (compare cpp/src/barretenberg/commitment_schemes/zeromorph/zeromorph.hpp or https://hackmd.io/dlf9xEwhTQyE3hiGbq4FsA?view)

    fn compute_c_zeta_x(
        c_q: P::G1Affine,
        c_q_k: &[P::G1Affine],
        y_challenge: P::ScalarField,
        x_challenge: P::ScalarField,
        circuit_size: u32,
    ) -> HonkVerifyResult<P::G1> {
        let log_circuit_size = Utils::get_msb32(circuit_size);

        let mut scalars = Vec::with_capacity(c_q_k.len() + 1);
        scalars.push(P::ScalarField::one());
        let mut commitments = Vec::new();
        commitments.push(c_q);

        // Contribution from C_q_k, k = 0,...,log_N-1
        c_q_k.iter().enumerate().for_each(|(k, &c_q_k_item)| {
            // Utilize dummy rounds in order to make verifier circuit independent of proof size
            let is_dummy_round = k >= log_circuit_size as usize;
            let deg_k = (1 << k) - 1;
            // Compute scalar y^k * x^{N - deg_k - 1}

            let scalar = if is_dummy_round {
                P::ScalarField::ZERO
            } else {
                let mut scalar = y_challenge.pow([k as u64]);
                let x_exponent = circuit_size - deg_k as u32 - 1;
                scalar *= x_challenge.pow([x_exponent as u64]);
                scalar *= P::ScalarField::ZERO - P::ScalarField::ONE;
                scalar
            };

            scalars.push(scalar);
            commitments.push(c_q_k_item);
        });

        Ok(Utils::msm::<P>(&scalars, &commitments)?)
    }

    fn compute_c_z_x(
        &self,
        c_q_k: Vec<P::G1Affine>,
        rho: P::ScalarField,
        batched_evaluation: P::ScalarField,
        x_challenge: P::ScalarField,
        u_challenge: Vec<P::ScalarField>,
        circuit_size: u32,
    ) -> HonkVerifyResult<P::G1> {
        let unshifted_commitments = Self::get_f_comms(&self.memory.verifier_commitments);
        let to_be_shifted_commitments = Self::get_g_shift_comms(&self.memory.verifier_commitments);

        let log_circuit_size = Utils::get_msb32(circuit_size);
        let mut scalars = Vec::with_capacity(
            1 + unshifted_commitments.len()
                + to_be_shifted_commitments.len()
                + CONST_PROOF_SIZE_LOG_N,
        );
        let mut commitments = Vec::with_capacity(
            1 + unshifted_commitments.len()
                + to_be_shifted_commitments.len()
                + CONST_PROOF_SIZE_LOG_N,
        );

        let phi_numerator = x_challenge.pow([circuit_size as u64]) - P::ScalarField::ONE;
        let minus_one = -P::ScalarField::ONE;
        let phi_n_x = phi_numerator / (x_challenge - P::ScalarField::ONE);

        scalars.push(batched_evaluation * x_challenge * phi_n_x * minus_one);
        commitments.push(P::G1Affine::generator());
        let mut rho_pow = P::ScalarField::ONE;
        for &value in unshifted_commitments.iter() {
            scalars.push(x_challenge * rho_pow);
            commitments.push(value);
            rho_pow *= rho;
        }
        for &value in to_be_shifted_commitments.iter() {
            scalars.push(rho_pow);
            commitments.push(value);
            rho_pow *= rho;
        }
        let mut x_pow_2k = x_challenge; // x^{2^k}
        let mut x_pow_2kp1 = x_challenge * x_challenge;

        for k in 0..CONST_PROOF_SIZE_LOG_N {
            let is_dummy_round = k >= log_circuit_size as usize;
            if is_dummy_round {
                scalars.push(P::ScalarField::ZERO);
                commitments.push(c_q_k[k]);
            } else {
                let phi_term_1 = phi_numerator / (x_pow_2kp1 - P::ScalarField::ONE); // \Phi_{n-k-1}(x^{2^{k + 1}})
                let phi_term_2 = phi_numerator / (x_pow_2k - P::ScalarField::ONE); // \Phi_{n-k}(x^{2^k})

                let scalar = ((x_pow_2k * phi_term_1) - (u_challenge[k] * phi_term_2))
                    * x_challenge
                    * minus_one;

                scalars.push(scalar);
                commitments.push(c_q_k[k]);

                // Update powers of challenge x
                x_pow_2k = x_pow_2kp1;
                x_pow_2kp1 *= x_pow_2kp1;
            }
        }

        Ok(Utils::msm::<P>(&scalars, &commitments)?)
    }
}
