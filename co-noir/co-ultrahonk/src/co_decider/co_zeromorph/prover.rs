use super::types::{PolyF, PolyG, PolyGShift};
use crate::{
    co_decider::{
        co_sumcheck::SumcheckOutput, polynomial::SharedPolynomial, prover::CoDecider,
        types::ClaimedEvaluations,
    },
    types::AllEntities,
    CoUtils, FieldShare, CONST_PROOF_SIZE_LOG_N,
};
use ark_ec::Group;
use ark_ff::{Field, One};
use itertools::izip;
use mpc_core::traits::{MSMProvider, PrimeFieldMpcProtocol};
use ultrahonk::{
    prelude::{
        HonkCurve, HonkProofResult, Polynomial, ProverCrs, TranscriptFieldType, TranscriptType,
    },
    Utils,
};

impl<T, P: HonkCurve<TranscriptFieldType>> CoDecider<T, P>
where
    T: PrimeFieldMpcProtocol<P::ScalarField> + MSMProvider<P::G1>,
{
    // /**
    //  * @brief Compute multivariate quotients q_k(X_0, ..., X_{k-1}) for f(X_0, ..., X_{n-1})
    //  * @details Starting from the coefficients of f, compute q_k inductively from k = n - 1, to k = 0.
    //  *          f needs to be updated at each step.
    //  *
    //  *          First, compute q_{n-1} of size N/2 by
    //  *          q_{n-1}[l] = f[N/2 + l ] - f[l].
    //  *
    //  *          Update f by f[l] <- f[l] + u_{n-1} * q_{n-1}[l]; f now has size N/2.
    //  *          Compute q_{n-2} of size N/(2^2) by
    //  *          q_{n-2}[l] = f[N/2^2 + l] - f[l].
    //  *
    //  *          Update f by f[l] <- f[l] + u_{n-2} * q_{n-2}[l]; f now has size N/(2^2).
    //  *          Compute q_{n-3} of size N/(2^3) by
    //  *          q_{n-3}[l] = f[N/2^3 + l] - f[l]. Repeat similarly until you reach q_0.
    //  *
    //  * @param polynomial Multilinear polynomial f(X_0, ..., X_{d-1})
    //  * @param u_challenge Multivariate challenge u = (u_0, ..., u_{d-1})
    //  * @return std::vector<Polynomial> The quotients q_k
    //  */
    fn compute_multilinear_quotients(
        driver: &mut T,
        polynomial: &SharedPolynomial<T, P>,
        u_challenge: &[P::ScalarField],
    ) -> Vec<SharedPolynomial<T, P>> {
        let log_n = Utils::get_msb64(polynomial.len() as u64);
        // Define the vector of quotients q_k, k = 0, ..., log_N-1
        // let mut quotients = Vec::with_capacity(log_n as usize);
        let mut quotients = vec![SharedPolynomial::default(); log_n as usize];

        // Compute the coefficients of q_{n-1}
        let mut size_q = 1 << (log_n - 1);
        let mut q = Vec::with_capacity(size_q);
        let (half_a, half_b) = polynomial.coefficients.split_at(size_q);
        for (a, b) in half_a.iter().zip(half_b.iter()) {
            q.push(driver.sub(b, a));
        }

        quotients[log_n as usize - 1].coefficients = q;

        let mut g = half_a.to_owned();

        // Compute q_k in reverse order from k= n-2, i.e. q_{n-2}, ..., q_0
        for k in 1..log_n {
            // Compute f_k
            let mut f_k = Vec::with_capacity(size_q);
            let index = log_n as usize - k as usize;
            for (g, q) in izip!(g, quotients[index].iter()) {
                let tmp = driver.mul_with_public(&u_challenge[index], q);
                f_k.push(driver.add(&g, &tmp));
            }
            size_q >>= 1;
            let mut q = Vec::with_capacity(size_q);
            let (half_a, half_b) = f_k.split_at(size_q);
            for (a, b) in half_a.iter().zip(half_b.iter()) {
                q.push(driver.sub(b, a));
            }

            quotients[index - 1].coefficients = q;
            g = f_k;
        }

        quotients
    }

    /**
     * @brief Construct batched, lifted-degree univariate quotient \hat{q} = \sum_k y^k * X^{N - d_k - 1} * q_k
     * @details The purpose of the batched lifted-degree quotient is to reduce the individual degree checks
     * deg(q_k) <= 2^k - 1 to a single degree check on \hat{q}. This is done by first shifting each of the q_k to the
     * right (i.e. multiplying by an appropriate power of X) so that each is degree N-1, then batching them all together
     * using powers of the provided challenge. Note: In practice, we do not actually compute the shifted q_k, we simply
     * accumulate them into \hat{q} at the appropriate offset.
     *
     * @param quotients Polynomials q_k, interpreted as univariates; deg(q_k) = 2^k - 1
     * @param N circuit size
     * @return Polynomial
     */
    fn compute_batched_lifted_degree_quotient(
        driver: &mut T,
        quotients: &[SharedPolynomial<T, P>],
        y_challenge: &P::ScalarField,
        n: usize,
    ) -> SharedPolynomial<T, P> {
        // Batched lifted degree quotient polynomial
        let mut result = vec![FieldShare::<T, P>::default(); n];

        // Compute \hat{q} = \sum_k y^k * X^{N - d_k - 1} * q_k
        let mut scalar = P::ScalarField::one();
        for (k, quotient) in quotients.iter().enumerate() {
            // Rather than explicitly computing the shifts of q_k by N - d_k - 1 (i.e. multiplying q_k by X^{N - d_k -
            // 1}) then accumulating them, we simply accumulate y^k*q_k into \hat{q} at the index offset N - d_k - 1
            let deg_k = (1 << k) - 1;
            let offset = n - deg_k - 1;

            for (r, q) in result
                .iter_mut()
                .skip(offset)
                .take(deg_k + 1)
                .zip(quotient.iter())
            {
                let tmp = driver.mul_with_public(&scalar, q);
                *r = driver.add(r, &tmp);
            }

            scalar *= y_challenge; // update batching scalar y^k
        }

        SharedPolynomial::new(result)
    }

    fn get_f_polyomials(
        polys: &AllEntities<Vec<FieldShare<T, P>>, Vec<P::ScalarField>>,
    ) -> PolyF<Vec<FieldShare<T, P>>, Vec<P::ScalarField>> {
        PolyF {
            precomputed: &polys.precomputed,
            witness: &polys.witness,
        }
    }

    fn get_g_shift_evaluations(
        evaluations: &ClaimedEvaluations<P::ScalarField>,
    ) -> PolyGShift<P::ScalarField> {
        PolyGShift {
            tables: &evaluations.shifted_tables,
            wires: &evaluations.shifted_witness,
        }
    }

    fn get_g_polyomials(
        polys: &AllEntities<Vec<FieldShare<T, P>>, Vec<P::ScalarField>>,
    ) -> PolyG<Vec<FieldShare<T, P>>, Vec<P::ScalarField>> {
        let tables = [
            polys.precomputed.table_1(),
            polys.precomputed.table_2(),
            polys.precomputed.table_3(),
            polys.precomputed.table_4(),
        ];

        let wires = [
            polys.witness.w_l(),
            polys.witness.w_r(),
            polys.witness.w_o(),
            polys.witness.w_4(),
        ];

        PolyG {
            tables,
            wires,
            z_perm: polys.witness.z_perm(),
        }
    }

    fn get_f_evaluations(
        evaluations: &ClaimedEvaluations<P::ScalarField>,
    ) -> PolyF<P::ScalarField, P::ScalarField> {
        PolyF {
            precomputed: &evaluations.precomputed,
            witness: &evaluations.witness,
        }
    }

    fn compute_batched_polys(
        &mut self,
        transcript: &mut TranscriptType,
        claimed_evaluations: AllEntities<P::ScalarField, P::ScalarField>,
        n: usize,
    ) -> (
        SharedPolynomial<T, P>,
        SharedPolynomial<T, P>,
        P::ScalarField,
    ) {
        let f_polynomials = Self::get_f_polyomials(&self.memory.polys);
        let g_polynomials = Self::get_g_polyomials(&self.memory.polys);
        let f_evaluations = Self::get_f_evaluations(&claimed_evaluations);
        let g_shift_evaluations = Self::get_g_shift_evaluations(&claimed_evaluations);

        // Generate batching challenge \rho and powers 1,...,\rho^{m-1}
        let rho = transcript.get_challenge::<P>("rho".to_string());

        // Compute batching of unshifted polynomials f_i and to-be-shifted polynomials g_i:
        // f_batched = sum_{i=0}^{m-1}\rho^i*f_i and g_batched = sum_{i=0}^{l-1}\rho^{m+i}*g_i,
        // and also batched evaluation
        // v = sum_{i=0}^{m-1}\rho^i*f_i(u) + sum_{i=0}^{l-1}\rho^{m+i}*h_i(u).
        // Note: g_batched is formed from the to-be-shifted polynomials, but the batched evaluation incorporates the
        // evaluations produced by sumcheck of h_i = g_i_shifted.
        let mut batched_evaluation = P::ScalarField::ZERO;
        let mut batching_scalar = P::ScalarField::ONE;
        let mut f_batched = Polynomial::new_zero(n); // batched unshifted polynomials

        // Precomputed part of f_batched
        for (f_poly, f_eval) in f_polynomials
            .precomputed
            .iter()
            .zip(f_evaluations.precomputed.iter())
        {
            f_batched.add_scaled_slice(f_poly, &batching_scalar);
            batched_evaluation += batching_scalar * f_eval;
            batching_scalar *= rho;
        }

        // Shared part of f_batched
        let mut f_batched = SharedPolynomial::<T, P>::promote_poly(&self.driver, f_batched);
        for (f_poly, f_eval) in f_polynomials
            .witness
            .shared_iter()
            .zip(f_evaluations.witness.shared_iter())
        {
            f_batched.add_scaled_slice(&mut self.driver, f_poly, &batching_scalar);
            batched_evaluation += batching_scalar * f_eval;
            batching_scalar *= rho;
        }

        // Final public part of f_batched
        for (f_poly, f_eval) in f_polynomials
            .witness
            .public_iter()
            .zip(f_evaluations.witness.public_iter())
        {
            f_batched.add_scaled_slice_public(&mut self.driver, f_poly, &batching_scalar);
            batched_evaluation += batching_scalar * f_eval;
            batching_scalar *= rho;
        }

        // For g_batched the order of public first and shared later is ok
        let mut g_batched = Polynomial::new_zero(n); // batched to-be-shifted polynomials

        // Public part of g_batched
        for (g_poly, g_shift_eval) in g_polynomials
            .public_iter()
            .zip(g_shift_evaluations.public_iter())
        {
            g_batched.add_scaled_slice(g_poly, &batching_scalar);
            batched_evaluation += batching_scalar * g_shift_eval;
            batching_scalar *= rho;
        }

        // Shared part of g_batched
        let mut g_batched = SharedPolynomial::<T, P>::promote_poly(&self.driver, g_batched);
        for (g_poly, g_shift_eval) in g_polynomials
            .shared_iter()
            .zip(g_shift_evaluations.shared_iter())
        {
            g_batched.add_scaled_slice(&mut self.driver, g_poly, &batching_scalar);
            batched_evaluation += batching_scalar * g_shift_eval;
            batching_scalar *= rho;
        }

        (f_batched, g_batched, batched_evaluation)
    }

    /**
     * @brief  * @brief Returns a univariate opening claim equivalent to a set of multilinear evaluation claims for
     * unshifted polynomials f_i and to-be-shifted polynomials g_i to be subsequently proved with a univariate PCS
     *
     * @param f_polynomials Unshifted polynomials
     * @param g_polynomials To-be-shifted polynomials (of which the shifts h_i were evaluated by sumcheck)
     * @param evaluations Set of evaluations v_i = f_i(u), w_i = h_i(u) = g_i_shifted(u)
     * @param multilinear_challenge Multilinear challenge point u
     * @param commitment_key
     * @param transcript
     *
     * @todo https://github.com/AztecProtocol/barretenberg/issues/1030: document concatenation trick
     */
    pub(crate) fn zeromorph_prove(
        &mut self,
        transcript: &mut TranscriptType,
        circuit_size: u32,
        crs: &ProverCrs<P>,
        sumcheck_output: SumcheckOutput<P::ScalarField>,
        // ) -> HonkProofResult<ZeroMorphOpeningClaim<P::ScalarField>> {
    ) -> HonkProofResult<()> {
        tracing::trace!("Zeromorph prove");

        let multilinear_challenge = &sumcheck_output.challenges;
        let commitment_key = crs;

        // Extract multilinear challenge u and claimed multilinear evaluations from Sumcheck output
        let u_challenge = multilinear_challenge;
        let log_n = Utils::get_msb32(circuit_size);
        let n = 1 << log_n;

        let (f_batched, g_batched, batched_evaluation) =
            self.compute_batched_polys(transcript, sumcheck_output.claimed_evaluations, n);

        // We don't have groups, so we skip a lot now

        // Compute the full batched polynomial f = f_batched + g_batched.shifted() = f_batched + h_batched. This is the
        // polynomial for which we compute the quotients q_k and prove f(u) = v_batched.
        let mut f_polynomial = f_batched.to_owned();

        f_polynomial.add_assign_slice(&mut self.driver, g_batched.shifted());
        // f_polynomial += concatenated_batched; // No groups

        // Compute the multilinear quotients q_k = q_k(X_0, ..., X_{k-1})
        let quotients =
            Self::compute_multilinear_quotients(&mut self.driver, &f_polynomial, u_challenge);
        debug_assert_eq!(quotients.len(), log_n as usize);
        // Compute and send commitments C_{q_k} = [q_k], k = 0,...,d-1
        let mut commitments = Vec::with_capacity(log_n as usize);
        for q in quotients.iter() {
            let commitment = CoUtils::commit(&mut self.driver, q.as_ref(), commitment_key);
            commitments.push(commitment);
        }
        let commitments = self.driver.open_point_many(&commitments)?;
        for (idx, val) in commitments.into_iter().enumerate() {
            let label = format!("ZM:C_q_{}", idx);
            transcript.send_point_to_verifier::<P>(label, val.into());
        }
        // Add buffer elements to remove log_N dependence in proof
        for idx in log_n as usize..CONST_PROOF_SIZE_LOG_N {
            let res = P::G1::generator(); // TODO Is this one?
            let label = format!("ZM:C_q_{}", idx);
            transcript.send_point_to_verifier::<P>(label, res.into());
        }

        // Get challenge y
        let y_challenge = transcript.get_challenge::<P>("ZM:y".to_string());

        // Compute the batched, lifted-degree quotient \hat{q}
        let batched_quotient = Self::compute_batched_lifted_degree_quotient(
            &mut self.driver,
            &quotients,
            &y_challenge,
            n,
        );

        // Compute and send the commitment C_q = [\hat{q}]
        let q_commitment = CoUtils::commit(
            &mut self.driver,
            &batched_quotient.coefficients,
            commitment_key,
        );
        let q_commitment = self.driver.open_point(&q_commitment)?;
        transcript.send_point_to_verifier::<P>("ZM:C_q".to_string(), q_commitment.into());

        // Get challenges x and z
        let challs = transcript.get_challenges::<P>(&["ZM:x".to_string(), "ZM:z".to_string()]);
        let x_challenge = challs[0];
        let z_challenge = challs[1];

        // Compute degree check polynomial \zeta partially evaluated at x
        let zeta_x = Self::compute_partially_evaluated_degree_check_polynomial(
            &batched_quotient,
            &quotients,
            &y_challenge,
            &x_challenge,
        );

        // Compute ZeroMorph identity polynomial Z partially evaluated at x
        // let z_x = Self::compute_partially_evaluated_zeromorph_identity_polynomial(
        //     f_batched,
        //     g_batched,
        //     quotients,
        //     batched_evaluation,
        //     u_challenge,
        //     x_challenge,
        // );

        // // Compute batched degree-check and ZM-identity quotient polynomial pi
        // let pi_polynomial =
        //     Self::compute_batched_evaluation_and_degree_check_polynomial(zeta_x, z_x, z_challenge);

        // let res = ZeroMorphOpeningClaim {
        //     polynomial: pi_polynomial,
        //     opening_pair: OpeningPair {
        //         challenge: x_challenge,
        //         evaluation: P::ScalarField::zero(),
        //     },
        // };
        // Ok(res)

        todo!("ZeroMorph prove")
    }
}
