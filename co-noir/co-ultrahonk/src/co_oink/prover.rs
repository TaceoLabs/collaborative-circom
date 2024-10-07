// clang-format off
/*                                            )\   /|
*                                          .-/'-|_/ |
*                       __            __,-' (   / \/
*                   .-'"  "'-..__,-'""          -o.`-._
*                  /                                   '/
*          *--._ ./                                 _.--
*                |                              _.-'
*                :                           .-/
*                 \                       )_ /
*                  \                _)   / \(
*                    `.   /-.___.---'(  /   \\
*                     (  /   \\       \(     L\
*                      \(     L\       \\
*                       \\              \\
*                        L\              L\
*/
// clang-format on

use super::types::ProverMemory;
use crate::{types::ProvingKey, CoUtils, FieldShare};
use ark_ff::One;
use itertools::izip;
use mpc_core::traits::{MSMProvider, PrimeFieldMpcProtocol};
use std::{array, marker::PhantomData};
use ultrahonk::{
    prelude::{
        HonkCurve, HonkProofError, HonkProofResult, Polynomial, Transcript, TranscriptFieldType,
        TranscriptHasher,
    },
    Utils, NUM_ALPHAS,
};

pub(crate) struct CoOink<
    'a,
    T,
    P: HonkCurve<TranscriptFieldType>,
    H: TranscriptHasher<TranscriptFieldType>,
> where
    T: PrimeFieldMpcProtocol<P::ScalarField> + MSMProvider<P::G1>,
{
    driver: &'a mut T,
    memory: ProverMemory<T, P>,
    phantom_data: PhantomData<P>,
    phantom_hasher: PhantomData<H>,
}

impl<'a, T, P: HonkCurve<TranscriptFieldType>, H: TranscriptHasher<TranscriptFieldType>>
    CoOink<'a, T, P, H>
where
    T: PrimeFieldMpcProtocol<P::ScalarField> + MSMProvider<P::G1>,
{
    pub(crate) fn new(driver: &'a mut T) -> Self {
        Self {
            driver,
            memory: ProverMemory::default(),
            phantom_data: PhantomData,
            phantom_hasher: PhantomData,
        }
    }

    fn compute_w4(&mut self, proving_key: &ProvingKey<T, P>) {
        tracing::trace!("compute w4");
        // The memory record values are computed at the indicated indices as
        // w4 = w3 * eta^3 + w2 * eta^2 + w1 * eta + read_write_flag;

        debug_assert_eq!(
            proving_key.polynomials.witness.w_l().len(),
            proving_key.polynomials.witness.w_r().len()
        );
        debug_assert_eq!(
            proving_key.polynomials.witness.w_l().len(),
            proving_key.polynomials.witness.w_o().len()
        );
        self.memory.w_4 = proving_key.polynomials.witness.w_4().clone();
        self.memory.w_4.resize(
            proving_key.polynomials.witness.w_l().len(),
            FieldShare::<T, P>::default(),
        );

        // Compute read record values
        for gate_idx in proving_key.memory_read_records.iter() {
            let gate_idx = *gate_idx as usize;
            let target = &mut self.memory.w_4[gate_idx];

            let mul1 = self.driver.mul_with_public(
                &self.memory.challenges.eta_1,
                &proving_key.polynomials.witness.w_l()[gate_idx],
            );
            let mul2 = self.driver.mul_with_public(
                &self.memory.challenges.eta_2,
                &proving_key.polynomials.witness.w_r()[gate_idx],
            );
            let mul3 = self.driver.mul_with_public(
                &self.memory.challenges.eta_3,
                &proving_key.polynomials.witness.w_o()[gate_idx],
            );
            // TACEO TODO add_assign?
            *target = self.driver.add(target, &mul1);
            *target = self.driver.add(target, &mul2);
            *target = self.driver.add(target, &mul3);
        }

        // Compute write record values
        for gate_idx in proving_key.memory_write_records.iter() {
            let gate_idx = *gate_idx as usize;
            let target = &mut self.memory.w_4[gate_idx];

            let mul1 = self.driver.mul_with_public(
                &self.memory.challenges.eta_1,
                &proving_key.polynomials.witness.w_l()[gate_idx],
            );
            let mul2 = self.driver.mul_with_public(
                &self.memory.challenges.eta_2,
                &proving_key.polynomials.witness.w_r()[gate_idx],
            );
            let mul3 = self.driver.mul_with_public(
                &self.memory.challenges.eta_3,
                &proving_key.polynomials.witness.w_o()[gate_idx],
            );
            // TACEO TODO add_assign?
            *target = self.driver.add(target, &mul1);
            *target = self.driver.add(target, &mul2);
            *target = self.driver.add(target, &mul3);
            *target = self.driver.add_with_public(&P::ScalarField::one(), target);
        }
    }

    fn compute_read_term(&mut self, proving_key: &ProvingKey<T, P>, i: usize) -> FieldShare<T, P> {
        tracing::trace!("compute read term");

        let gamma = &self.memory.challenges.gamma;
        let eta_1 = &self.memory.challenges.eta_1;
        let eta_2 = &self.memory.challenges.eta_2;
        let eta_3 = &self.memory.challenges.eta_3;
        let w_1 = &proving_key.polynomials.witness.w_l()[i];
        let w_2 = &proving_key.polynomials.witness.w_r()[i];
        let w_3 = &proving_key.polynomials.witness.w_o()[i];
        let w_1_shift = &proving_key.polynomials.witness.w_l().shifted()[i];
        let w_2_shift = &proving_key.polynomials.witness.w_r().shifted()[i];
        let w_3_shift = &proving_key.polynomials.witness.w_o().shifted()[i];
        let table_index = &proving_key.polynomials.precomputed.q_o()[i];
        let negative_column_1_step_size = &proving_key.polynomials.precomputed.q_r()[i];
        let negative_column_2_step_size = &proving_key.polynomials.precomputed.q_m()[i];
        let negative_column_3_step_size = &proving_key.polynomials.precomputed.q_c()[i];

        // The wire values for lookup gates are accumulators structured in such a way that the differences w_i -
        // step_size*w_i_shift result in values present in column i of a corresponding table. See the documentation in
        // method get_lookup_accumulators() in  for a detailed explanation.

        let mul = &self
            .driver
            .mul_with_public(negative_column_1_step_size, w_1_shift);
        let add = &self.driver.add_with_public(gamma, mul);
        let derived_table_entry_1 = self.driver.add(w_1, add);

        let mul = self
            .driver
            .mul_with_public(negative_column_2_step_size, w_2_shift);
        let derived_table_entry_2 = self.driver.add(w_2, &mul);

        let mul = self
            .driver
            .mul_with_public(negative_column_3_step_size, w_3_shift);
        let derived_table_entry_3 = self.driver.add(w_3, &mul);

        // (w_1 + \gamma q_2*w_1_shift) + η(w_2 + q_m*w_2_shift) + η₂(w_3 + q_c*w_3_shift) + η₃q_index.
        // deg 2 or 3
        // TACEO TODO add_assign?
        let mul = self.driver.mul_with_public(eta_1, &derived_table_entry_2);
        let res = self.driver.add(&derived_table_entry_1, &mul);
        let mul = &self.driver.mul_with_public(eta_2, &derived_table_entry_3);
        let res = self.driver.add(&res, mul);
        self.driver.add_with_public(&(*table_index * eta_3), &res)
    }

    // Compute table_1 + gamma + table_2 * eta + table_3 * eta_2 + table_4 * eta_3
    fn compute_write_term(&self, proving_key: &ProvingKey<T, P>, i: usize) -> P::ScalarField {
        tracing::trace!("compute write term");

        let gamma = &self.memory.challenges.gamma;
        let eta_1 = &self.memory.challenges.eta_1;
        let eta_2 = &self.memory.challenges.eta_2;
        let eta_3 = &self.memory.challenges.eta_3;
        let table_1 = &proving_key.polynomials.precomputed.table_1()[i];
        let table_2 = &proving_key.polynomials.precomputed.table_2()[i];
        let table_3 = &proving_key.polynomials.precomputed.table_3()[i];
        let table_4 = &proving_key.polynomials.precomputed.table_4()[i];

        *table_1 + gamma + *table_2 * eta_1 + *table_3 * eta_2 + *table_4 * eta_3
    }

    fn compute_logderivative_inverses(
        &mut self,
        proving_key: &ProvingKey<T, P>,
    ) -> HonkProofResult<()> {
        tracing::trace!("compute logderivative inverse");

        debug_assert_eq!(
            proving_key.polynomials.precomputed.q_lookup().len(),
            proving_key.circuit_size as usize
        );
        debug_assert_eq!(
            proving_key.polynomials.witness.lookup_read_tags().len(),
            proving_key.circuit_size as usize
        );
        self.memory
            .lookup_inverses
            .resize(proving_key.circuit_size as usize, Default::default());

        // const READ_TERMS: usize = 1;
        // const WRITE_TERMS: usize = 1;
        // // 1 + polynomial degree of this relation
        // const LENGTH: usize = 5; // both subrelations are degree 4

        for (i, (q_lookup, lookup_read_tag)) in izip!(
            proving_key.polynomials.precomputed.q_lookup().iter(),
            proving_key.polynomials.witness.lookup_read_tags().iter(),
        )
        .enumerate()
        {
            if !(q_lookup.is_one() || lookup_read_tag.is_one()) {
                continue;
            }

            // READ_TERMS and WRITE_TERMS are 1, so we skip the loop
            let read_term = self.compute_read_term(proving_key, i);
            let write_term = self.compute_write_term(proving_key, i);
            self.memory.lookup_inverses[i] = self.driver.mul_with_public(&write_term, &read_term);
        }

        // Compute inverse polynomial I in place by inverting the product at each row
        // Note: zeroes are ignored as they are not used anyway
        CoUtils::batch_invert::<T, P>(self.driver, self.memory.lookup_inverses.as_mut())?;
        Ok(())
    }

    fn compute_public_input_delta(&self, proving_key: &ProvingKey<T, P>) -> P::ScalarField {
        tracing::trace!("compute public input delta");

        // Let m be the number of public inputs x₀,…, xₘ₋₁.
        // Recall that we broke the permutation σ⁰ by changing the mapping
        //  (i) -> (n+i)   to   (i) -> (-(i+1))   i.e. σ⁰ᵢ = −(i+1)
        //
        // Therefore, the term in the numerator with ID¹ᵢ = n+i does not cancel out with any term in the denominator.
        // Similarly, the denominator contains an extra σ⁰ᵢ = −(i+1) term that does not appear in the numerator.
        // We expect the values of W⁰ᵢ and W¹ᵢ to be equal to xᵢ.
        // The expected accumulated product would therefore be equal to

        //   ∏ᵢ (γ + W¹ᵢ + β⋅ID¹ᵢ)        ∏ᵢ (γ + xᵢ + β⋅(n+i) )
        //  -----------------------  =  ------------------------
        //   ∏ᵢ (γ + W⁰ᵢ + β⋅σ⁰ᵢ )        ∏ᵢ (γ + xᵢ - β⋅(i+1) )

        // At the start of the loop for each xᵢ where i = 0, 1, …, m-1,
        // we have
        //      numerator_acc   = γ + β⋅(n+i) = γ + β⋅n + β⋅i
        //      denominator_acc = γ - β⋅(1+i) = γ - β   - β⋅i
        // at the end of the loop, add and subtract β to each term respectively to
        // set the expected value for the start of iteration i+1.
        // Note: The public inputs may be offset from the 0th index of the wires, for example due to the inclusion of an
        // initial zero row or Goblin-stlye ECC op gates. Accordingly, the indices i in the above formulas are given by i =
        // [0, m-1] + offset, i.e. i = offset, 1 + offset, …, m - 1 + offset.

        let mut num = P::ScalarField::one();
        let mut denom = P::ScalarField::one();
        let mut num_acc = self.memory.challenges.gamma
            + self.memory.challenges.beta
                * P::ScalarField::from(
                    (proving_key.circuit_size + proving_key.pub_inputs_offset) as u64,
                );
        let mut denom_acc = self.memory.challenges.gamma
            - self.memory.challenges.beta
                * P::ScalarField::from((1 + proving_key.pub_inputs_offset) as u64);

        for x_i in proving_key.public_inputs.iter() {
            num *= num_acc + x_i;
            denom *= denom_acc + x_i;
            num_acc += self.memory.challenges.beta;
            denom_acc -= self.memory.challenges.beta;
        }
        num / denom
    }

    fn batched_grand_product_num_denom(
        driver: &mut T,
        shared1: &Polynomial<FieldShare<T, P>>,
        shared2: &Polynomial<FieldShare<T, P>>,
        pub1: &Polynomial<P::ScalarField>,
        pub2: &Polynomial<P::ScalarField>,
        beta: &P::ScalarField,
        gamma: &P::ScalarField,
    ) -> HonkProofResult<Vec<FieldShare<T, P>>> {
        let len = shared1.len();
        debug_assert_eq!(len, shared2.len());

        // We drop the last element since it is not needed for the grand product
        let mut mul1 = Vec::with_capacity(len - 1);
        let mut mul2 = Vec::with_capacity(len - 1);

        for (s1, s2, p1, p2) in
            izip!(shared1.iter(), shared2.iter(), pub1.iter(), pub2.iter()).take(len - 1)
        {
            let m1 = driver.add_with_public(&(*p1 * beta + gamma), s1);
            let m2 = driver.add_with_public(&(*p2 * beta + gamma), s2);
            mul1.push(m1);
            mul2.push(m2);
        }

        Ok(driver.mul_many(&mul1, &mul2)?)
    }

    // To reduce the number of communication rounds, we implement the array_prod_mul macro according to https://www.usenix.org/system/files/sec22-ozdemir.pdf, p11 first paragraph.
    fn array_prod_mul(
        &mut self,
        inp: &[FieldShare<T, P>],
    ) -> HonkProofResult<Vec<FieldShare<T, P>>> {
        // Do the multiplications of inp[i] * inp[i-1] in constant rounds
        let len = inp.len();

        let r = (0..=len)
            .map(|_| self.driver.rand())
            .collect::<Result<Vec<_>, _>>()?;
        let r_inv = self.driver.inv_many(&r)?;
        let r_inv0 = vec![r_inv[0].clone(); len];

        let mut unblind = self.driver.mul_many(&r_inv0, &r[1..])?;

        let mul = self.driver.mul_many(&r[..len], inp)?;
        let mut open = self.driver.mul_open_many(&mul, &r_inv[1..])?;

        for i in 1..open.len() {
            open[i] = open[i] * open[i - 1];
        }

        for (unblind, open) in unblind.iter_mut().zip(open.iter()) {
            *unblind = self.driver.mul_with_public(open, unblind);
        }
        Ok(unblind)
    }

    fn compute_grand_product(&mut self, proving_key: &ProvingKey<T, P>) -> HonkProofResult<()> {
        tracing::trace!("compute grand product");
        // Barratenberg uses multithreading here

        // In Barretenberg circuit size is taken from the q_c polynomial
        // Step (1)
        // Populate `numerator` and `denominator` with the algebra described by Relation

        // TACEO TODO could batch those 4 as well
        let denom1 = Self::batched_grand_product_num_denom(
            self.driver,
            proving_key.polynomials.witness.w_l(),
            proving_key.polynomials.witness.w_r(),
            proving_key.polynomials.precomputed.sigma_1(),
            proving_key.polynomials.precomputed.sigma_2(),
            &self.memory.challenges.beta,
            &self.memory.challenges.gamma,
        )?;
        let denom2 = Self::batched_grand_product_num_denom(
            self.driver,
            proving_key.polynomials.witness.w_o(),
            &self.memory.w_4,
            proving_key.polynomials.precomputed.sigma_3(),
            proving_key.polynomials.precomputed.sigma_4(),
            &self.memory.challenges.beta,
            &self.memory.challenges.gamma,
        )?;
        let num1 = Self::batched_grand_product_num_denom(
            self.driver,
            proving_key.polynomials.witness.w_l(),
            proving_key.polynomials.witness.w_r(),
            proving_key.polynomials.precomputed.id_1(),
            proving_key.polynomials.precomputed.id_2(),
            &self.memory.challenges.beta,
            &self.memory.challenges.gamma,
        )?;
        let num2 = Self::batched_grand_product_num_denom(
            self.driver,
            proving_key.polynomials.witness.w_o(),
            &self.memory.w_4,
            proving_key.polynomials.precomputed.id_3(),
            proving_key.polynomials.precomputed.id_4(),
            &self.memory.challenges.beta,
            &self.memory.challenges.gamma,
        )?;

        // TACEO TODO could batch here as well
        let numerator = self.driver.mul_many(&num1, &num2)?;
        let denominator = self.driver.mul_many(&denom1, &denom2)?;

        // Step (2)
        // Compute the accumulating product of the numerator and denominator terms.

        // TACEO TODO could batch here as well
        // Do the multiplications of num[i] * num[i-1] and den[i] * den[i-1] in constant rounds
        let numerator = self.array_prod_mul(&numerator)?;
        let mut denominator = self.array_prod_mul(&denominator)?;

        // invert denominator
        CoUtils::batch_invert::<T, P>(self.driver, &mut denominator)?;

        // Step (3) Compute z_perm[i] = numerator[i] / denominator[i]
        let mut z_perm = self.driver.mul_many(&numerator, &denominator)?;
        z_perm.insert(0, FieldShare::<T, P>::default()); // insert a default element at the beginning
        self.memory.z_perm = Polynomial::new(z_perm);
        Ok(())
    }

    // Generate relation separators alphas for sumcheck/combiner computation
    fn generate_alphas_round(&mut self, transcript: &mut Transcript<TranscriptFieldType, H>) {
        tracing::trace!("generate alpha round");

        let args: [String; NUM_ALPHAS] = array::from_fn(|i| format!("alpha_{}", i));
        self.memory
            .challenges
            .alphas
            .copy_from_slice(&transcript.get_challenges::<P>(&args));
    }

    // Add circuit size public input size and public inputs to transcript
    fn execute_preamble_round(
        transcript: &mut Transcript<TranscriptFieldType, H>,
        proving_key: &ProvingKey<T, P>,
    ) -> HonkProofResult<()> {
        tracing::trace!("executing preamble round");

        transcript
            .send_u64_to_verifier("circuit_size".to_string(), proving_key.circuit_size as u64);
        transcript.send_u64_to_verifier(
            "public_input_size".to_string(),
            proving_key.num_public_inputs as u64,
        );
        transcript.send_u64_to_verifier(
            "pub_inputs_offset".to_string(),
            proving_key.pub_inputs_offset as u64,
        );

        if proving_key.num_public_inputs as usize != proving_key.public_inputs.len() {
            return Err(HonkProofError::CorruptedWitness(
                proving_key.public_inputs.len(),
            ));
        }

        for (i, public_input) in proving_key.public_inputs.iter().enumerate() {
            // transcript.add_scalar(*public_input);
            transcript.send_fr_to_verifier::<P>(format!("public_input_{}", i), *public_input);
        }
        Ok(())
    }

    // Compute first three wire commitments
    fn execute_wire_commitments_round(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        proving_key: &ProvingKey<T, P>,
    ) -> HonkProofResult<()> {
        tracing::trace!("executing wire commitments round");

        // Commit to the first three wire polynomials of the instance
        // We only commit to the fourth wire polynomial after adding memory records

        let w_l = CoUtils::commit(
            self.driver,
            proving_key.polynomials.witness.w_l().as_ref(),
            &proving_key.crs,
        );
        let w_r = CoUtils::commit(
            self.driver,
            proving_key.polynomials.witness.w_r().as_ref(),
            &proving_key.crs,
        );
        let w_o = CoUtils::commit(
            self.driver,
            proving_key.polynomials.witness.w_o().as_ref(),
            &proving_key.crs,
        );

        let open = self.driver.open_point_many(&[w_l, w_r, w_o])?;

        transcript.send_point_to_verifier::<P>("W_L".to_string(), open[0].into());
        transcript.send_point_to_verifier::<P>("W_R".to_string(), open[1].into());
        transcript.send_point_to_verifier::<P>("W_O".to_string(), open[2].into());

        // Round is done since ultra_honk is no goblin flavor
        Ok(())
    }

    // Compute sorted list accumulator and commitment
    fn execute_sorted_list_accumulator_round(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        proving_key: &ProvingKey<T, P>,
    ) -> HonkProofResult<()> {
        tracing::trace!("executing sorted list accumulator round");

        let challs = transcript.get_challenges::<P>(&[
            "eta".to_string(),
            "eta_two".to_string(),
            "eta_three".to_string(),
        ]);
        self.memory.challenges.eta_1 = challs[0];
        self.memory.challenges.eta_2 = challs[1];
        self.memory.challenges.eta_3 = challs[2];
        self.compute_w4(proving_key);

        // Commit to lookup argument polynomials and the finalized (i.e. with memory records) fourth wire polynomial
        let lookup_read_counts = Utils::commit(
            proving_key
                .polynomials
                .witness
                .lookup_read_counts()
                .as_ref(),
            &proving_key.crs,
        )?;
        let lookup_read_tags = Utils::commit(
            proving_key.polynomials.witness.lookup_read_tags().as_ref(),
            &proving_key.crs,
        )?;
        let w_4 = CoUtils::commit(self.driver, self.memory.w_4.as_ref(), &proving_key.crs);
        let w_4 = self.driver.open_point(&w_4)?;

        transcript.send_point_to_verifier::<P>(
            "LOOKUP_READ_COUNTS".to_string(),
            lookup_read_counts.into(),
        );
        transcript
            .send_point_to_verifier::<P>("LOOKUP_READ_TAGS".to_string(), lookup_read_tags.into());
        transcript.send_point_to_verifier::<P>("W_4".to_string(), w_4.into());

        Ok(())
    }

    // Fiat-Shamir: beta & gamma
    fn execute_log_derivative_inverse_round(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        proving_key: &ProvingKey<T, P>,
    ) -> HonkProofResult<()> {
        tracing::trace!("executing log derivative inverse round");

        let challs = transcript.get_challenges::<P>(&["beta".to_string(), "gamma".to_string()]);
        self.memory.challenges.beta = challs[0];
        self.memory.challenges.gamma = challs[1];

        self.compute_logderivative_inverses(proving_key)?;

        // We moved the commiting and opening of the lookup inverses to be at the same time as z_perm

        // Round is done since ultra_honk is no goblin flavor
        Ok(())
    }

    // Compute grand product(s) and commitments.
    fn execute_grand_product_computation_round(
        &mut self,
        transcript: &mut Transcript<TranscriptFieldType, H>,
        proving_key: &ProvingKey<T, P>,
    ) -> HonkProofResult<()> {
        tracing::trace!("executing grand product computation round");

        self.memory.public_input_delta = self.compute_public_input_delta(proving_key);
        self.compute_grand_product(proving_key)?;

        // This is from the previous round, but we open it here with z_perm
        let lookup_inverses = CoUtils::commit(
            self.driver,
            self.memory.lookup_inverses.as_ref(),
            &proving_key.crs,
        );

        let z_perm = CoUtils::commit(self.driver, self.memory.z_perm.as_ref(), &proving_key.crs);

        let open = self.driver.open_point_many(&[lookup_inverses, z_perm])?;

        transcript.send_point_to_verifier::<P>("LOOKUP_INVERSES".to_string(), open[0].into());
        transcript.send_point_to_verifier::<P>("Z_PERM".to_string(), open[1].into());
        Ok(())
    }

    pub(crate) fn prove(
        mut self,
        proving_key: &ProvingKey<T, P>,
        transcript: &mut Transcript<TranscriptFieldType, H>,
    ) -> HonkProofResult<ProverMemory<T, P>> {
        tracing::trace!("Oink prove");

        // Add circuit size public input size and public inputs to transcript
        Self::execute_preamble_round(transcript, proving_key)?;
        // Compute first three wire commitments
        self.execute_wire_commitments_round(transcript, proving_key)?;
        // Compute sorted list accumulator and commitment
        self.execute_sorted_list_accumulator_round(transcript, proving_key)?;

        // Fiat-Shamir: beta & gamma
        self.execute_log_derivative_inverse_round(transcript, proving_key)?;
        // Compute grand product(s) and commitments.
        self.execute_grand_product_computation_round(transcript, proving_key)?;

        // Generate relation separators alphas for sumcheck/combiner computation
        self.generate_alphas_round(transcript);

        Ok(self.memory)
    }
}
