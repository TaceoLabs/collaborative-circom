use super::types::{PolyF, PolyG, PolyGShift};
use crate::{
    co_decider::{
        co_sumcheck::SumcheckOutput,
        co_zeromorph::{OpeningPair, ZeroMorphOpeningClaim},
        polynomial::SharedPolynomial,
        prover::CoDecider,
        types::ClaimedEvaluations,
    },
    types::AllEntities,
    CoUtils, FieldShare, CONST_PROOF_SIZE_LOG_N, N_MAX,
};
use ark_ec::Group;
use ark_ff::{Field, One, Zero};
use itertools::izip;
use mpc_core::traits::{MSMProvider, PrimeFieldMpcProtocol};
use ultrahonk::{
    prelude::{
        HonkCurve, HonkProofResult, Polynomial, ProverCrs, Transcript, TranscriptFieldType,
        TranscriptHasher,
    },
    Utils,
};

impl<T, P: HonkCurve<TranscriptFieldType>, H: TranscriptHasher<TranscriptFieldType>>
    CoDecider<T, P, H>
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

    // /**
    //  * @brief Compute partially evaluated degree check polynomial \zeta_x = q - \sum_k y^k * x^{N - d_k - 1} * q_k
    //  * @details Compute \zeta_x, where
    //  *
    //  *                          \zeta_x = q - \sum_k y^k * x^{N - d_k - 1} * q_k
    //  *
    //  * @param batched_quotient
    //  * @param quotients
    //  * @param y_challenge
    //  * @param x_challenge
    //  * @return Polynomial Degree check polynomial \zeta_x such that \zeta_x(x) = 0
    //  */
    fn compute_partially_evaluated_degree_check_polynomial(
        driver: &mut T,
        batched_quotient: &SharedPolynomial<T, P>,
        quotients: &[SharedPolynomial<T, P>],
        y_challenge: &P::ScalarField,
        x_challenge: &P::ScalarField,
    ) -> SharedPolynomial<T, P> {
        let n = batched_quotient.len();

        // Initialize partially evaluated degree check polynomial \zeta_x to \hat{q}
        let mut result = batched_quotient.clone();

        let mut y_power = P::ScalarField::ONE; // y^k
        for (k, q) in quotients.iter().enumerate() {
            // Accumulate y^k * x^{N - d_k - 1} * q_k into \hat{q}
            let deg_k = (1 << k) - 1;
            let exponent = (n - deg_k - 1) as u64;
            let x_power = x_challenge.pow([exponent]); // x^{N - d_k - 1}

            result.add_scaled(driver, q, &(-y_power * x_power));

            y_power *= y_challenge; // update batching scalar y^k
        }

        result
    }

    /**
     * @brief Compute partially evaluated zeromorph identity polynomial Z_x
     * @details Compute Z_x, where
     *
     *  Z_x = x * f_batched + g_batched - v * x * \Phi_n(x)
     *           - x * \sum_k (x^{2^k}\Phi_{n-k-1}(x^{2^{k-1}}) - u_k\Phi_{n-k}(x^{2^k})) * q_k
     *           + concatentation_term
     *
     * where f_batched = \sum_{i=0}^{m-1}\rho^i*f_i, g_batched = \sum_{i=0}^{l-1}\rho^{m+i}*g_i
     *
     * and concatenation_term = \sum_{i=0}^{num_chunks_per_group}(x^{i * min_N + 1}concatenation_groups_batched_{i})
     *
     * @note The concatenation term arises from an implementation detail in the Translator and is not part of the
     * conventional ZM protocol
     * @param input_polynomial
     * @param quotients
     * @param v_evaluation
     * @param x_challenge
     * @return Polynomial
     */
    fn compute_partially_evaluated_zeromorph_identity_polynomial(
        driver: &mut T,
        f_batched: SharedPolynomial<T, P>,
        g_batched: SharedPolynomial<T, P>,
        quotients: Vec<SharedPolynomial<T, P>>,
        v_evaluation: P::ScalarField,
        u_challenge: &[P::ScalarField],
        x_challenge: P::ScalarField,
    ) -> SharedPolynomial<T, P> {
        let n = f_batched.len();

        // Initialize Z_x with x * \sum_{i=0}^{m-1} f_i + \sum_{i=0}^{l-1} g_i
        let mut result = g_batched;
        result.add_scaled(driver, &f_batched, &x_challenge);

        // Compute Z_x -= v * x * \Phi_n(x)
        let phi_numerator = x_challenge.pow([n as u64]) - P::ScalarField::ONE; // x^N - 1
        let phi_n_x = phi_numerator / (x_challenge - P::ScalarField::ONE);
        let rhs = v_evaluation * x_challenge * phi_n_x;
        result[0] = driver.add_with_public(&-rhs, &result[0]);

        // Add contribution from q_k polynomials
        for (k, (q, u)) in izip!(quotients.iter(), u_challenge.iter()).enumerate() {
            let exp_1 = 1 << k;
            let x_power = x_challenge.pow([exp_1]); // x^{2^k}

            // \Phi_{n-k-1}(x^{2^{k + 1}})
            let exp_2 = 1 << (k + 1);
            let phi_term_1 = phi_numerator / (x_challenge.pow([exp_2]) - P::ScalarField::ONE);

            // \Phi_{n-k}(x^{2^k})
            let phi_term_2 = phi_numerator / (x_challenge.pow([exp_1]) - P::ScalarField::ONE);

            // x^{2^k} * \Phi_{n-k-1}(x^{2^{k+1}}) - u_k *  \Phi_{n-k}(x^{2^k})
            let mut scalar = x_power * phi_term_1 - phi_term_2 * u;

            scalar *= x_challenge;
            scalar *= -P::ScalarField::ONE;

            result.add_scaled(driver, q, &scalar);
        }

        // We don't have groups, so we are done already

        result
    }

    /**
     * @brief Compute combined evaluation and degree-check polynomial pi
     * @details Compute univariate polynomial pi, where
     *
     *  pi = (\zeta_c + z*Z_x) X^{N_{max}-(N-1)}
     *
     * The proof that pi(x) = 0 for some verifier challenge x will then be computed as part of the univariate PCS
     * opening. If this is instantiated with KZG, the PCS is going to compute the quotient
     * q_pi = (q_\zeta + z*q_Z)X^{N_{max}-(N-1)}, with q_\zeta = \zeta_x/(X-x), q_Z = Z_x/(X-x),
     *
     * @param Z_x
     * @param zeta_x
     * @param x_challenge
     * @param z_challenge
     * @param N_max
     * @return Polynomial
     */
    fn compute_batched_evaluation_and_degree_check_polynomial(
        driver: &mut T,
        zeta_x: SharedPolynomial<T, P>,
        z_x: SharedPolynomial<T, P>,
        z_challenge: P::ScalarField,
    ) -> SharedPolynomial<T, P> {
        // We cannot commit to polynomials with size > N_max
        let n = zeta_x.len();
        assert!(n <= N_MAX);
        let mut batched_polynomial = zeta_x;
        batched_polynomial.add_scaled(driver, &z_x, &z_challenge);

        // AZTEC TODO(#742): To complete the degree check, we need to do an opening proof for x_challenge with a univariate
        // PCS for the degree-lifted polynomial (\zeta_c + z*Z_x)*X^{N_max - N - 1}. If this PCS is KZG, verification
        // then requires a pairing check similar to the standard KZG check but with [1]_2 replaced by [X^{N_max - N
        // -1}]_2. Two issues: A) we do not have an SRS with these G2 elements (so need to generate a fake setup until
        // we can do the real thing), and B) its not clear to me how to update our pairing algorithms to do this type of
        // pairing. For now, simply construct pi without the shift and do a standard KZG pairing check if the PCS is
        // KZG. When we're ready, all we have to do to make this fully legit is commit to the shift here and update the
        // pairing check accordingly. Note: When this is implemented properly, it doesnt make sense to store the
        // (massive) shifted polynomial of size N_max. Ideally would only store the unshifted version and just compute
        // the shifted commitment directly via a new method.
        batched_polynomial
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
        transcript: &mut Transcript<TranscriptFieldType, H>,
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
     * @AZTEC todo https://github.com/AztecProtocol/barretenberg/issues/1030: document concatenation trick
     */
    pub(crate) fn zeromorph_prove(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        circuit_size: u32,
        crs: &ProverCrs<P>,
        sumcheck_output: SumcheckOutput<P::ScalarField>,
    ) -> HonkProofResult<ZeroMorphOpeningClaim<T, P>> {
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
            let res = P::G1::generator();
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
            &mut self.driver,
            &batched_quotient,
            &quotients,
            &y_challenge,
            &x_challenge,
        );

        // Compute ZeroMorph identity polynomial Z partially evaluated at x
        let z_x = Self::compute_partially_evaluated_zeromorph_identity_polynomial(
            &mut self.driver,
            f_batched,
            g_batched,
            quotients,
            batched_evaluation,
            u_challenge,
            x_challenge,
        );

        // Compute batched degree-check and ZM-identity quotient polynomial pi
        let pi_polynomial = Self::compute_batched_evaluation_and_degree_check_polynomial(
            &mut self.driver,
            zeta_x,
            z_x,
            z_challenge,
        );

        let res = ZeroMorphOpeningClaim {
            polynomial: pi_polynomial,
            opening_pair: OpeningPair {
                challenge: x_challenge,
                evaluation: P::ScalarField::zero(),
            },
        };
        Ok(res)
    }
}
