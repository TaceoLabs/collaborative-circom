use super::{
    acir_format::AcirFormat,
    plookup::BasicTableId,
    types::{
        AddQuad, AddTriple, AggregationObjectIndices, AggregationObjectPubInputIndices,
        AuxSelectors, BlockConstraint, BlockType, CachedPartialNonNativeFieldMultiplication,
        ColumnIdx, MulQuad, PlookupBasicTable, PolyTriple, RamTranscript, RangeList, ReadData,
        RomTranscript, UltraTraceBlock, UltraTraceBlocks,
    },
    verification_key::VerifyingKeyBarretenberg,
};
use crate::{
    decider::sumcheck::verifier,
    parse::{
        plookup::{MultiTableId, Plookup},
        types::{FieldCT, GateCounter, RomRecord, RomTable, NUM_WIRES},
    },
    prover::HonkProofResult,
    types::{Crs, PrecomputedEntities, ProverCrs, ProvingKey, VerifyingKey},
    Utils,
};
use ark_ec::pairing::Pairing;
use ark_ff::{One, PrimeField, Zero};
use num_bigint::BigUint;
use std::{collections::HashMap, fmt::Debug};

type GateBlocks<F> = UltraTraceBlocks<UltraTraceBlock<F>>;

pub trait UltraCircuitVariable<F>: Clone + PartialEq + Debug {
    type Shared;

    fn from_public(value: F) -> Self;
    fn from_shared(value: Self::Shared) -> Self;
    fn is_public(&self) -> bool;
    fn public_into_field(self) -> F;
}

impl<F: PrimeField> UltraCircuitVariable<F> for F {
    type Shared = F;

    fn from_public(value: F) -> Self {
        value
    }

    fn from_shared(value: Self::Shared) -> Self {
        value
    }

    fn is_public(&self) -> bool {
        true
    }

    fn public_into_field(self) -> F {
        self
    }
}

pub type UltraCircuitBuilder<P> = GenericUltraCircuitBuilder<P, <P as Pairing>::ScalarField>;

impl<P: Pairing> UltraCircuitBuilder<P> {
    pub fn create_vk_barretenberg(
        self,
        crs: ProverCrs<P>,
    ) -> HonkProofResult<VerifyingKeyBarretenberg<P>> {
        let contains_recursive_proof = self.contains_recursive_proof;
        let recursive_proof_public_input_indices = self.recursive_proof_public_input_indices;

        let pk = ProvingKey::create(self, crs);
        let circuit_size = pk.circuit_size;

        let mut commitments = PrecomputedEntities::default();
        for (des, src) in commitments
            .iter_mut()
            .zip(pk.polynomials.precomputed.iter())
        {
            let comm = Utils::commit(src.as_ref(), &pk.crs)?;
            *des = P::G1Affine::from(comm);
        }

        let vk = VerifyingKeyBarretenberg {
            circuit_size: circuit_size as u64,
            log_circuit_size: Utils::get_msb64(circuit_size as u64) as u64,
            num_public_inputs: pk.num_public_inputs as u64,
            pub_inputs_offset: pk.pub_inputs_offset as u64,
            contains_recursive_proof,
            recursive_proof_public_input_indices,
            commitments,
        };

        Ok(vk)
    }

    pub fn create_keys(self, crs: Crs<P>) -> HonkProofResult<(ProvingKey<P>, VerifyingKey<P>)> {
        let prover_crs = ProverCrs {
            monomials: crs.monomials,
        };
        let verifier_crs = crs.g2_x;

        let pk = ProvingKey::create(self, prover_crs);
        let circuit_size = pk.circuit_size;

        let mut commitments = PrecomputedEntities::default();
        for (des, src) in commitments
            .iter_mut()
            .zip(pk.polynomials.precomputed.iter())
        {
            let comm = Utils::commit(src.as_ref(), &pk.crs)?;
            *des = P::G1Affine::from(comm);
        }

        // Create and return the VerifyingKey instance
        let vk = VerifyingKey {
            crs: verifier_crs,
            circuit_size,
            num_public_inputs: pk.num_public_inputs,
            pub_inputs_offset: pk.pub_inputs_offset,
            commitments,
        };

        Ok((pk, vk))
    }
}

pub struct GenericUltraCircuitBuilder<P: Pairing, S: UltraCircuitVariable<P::ScalarField>> {
    pub variables: Vec<S>,
    variable_names: HashMap<u32, String>,
    next_var_index: Vec<u32>,
    prev_var_index: Vec<u32>,
    pub real_variable_index: Vec<u32>,
    pub(crate) real_variable_tags: Vec<u32>,
    pub public_inputs: Vec<u32>,
    is_recursive_circuit: bool,
    pub(crate) tau: HashMap<u32, u32>,
    constant_variable_indices: HashMap<P::ScalarField, u32>,
    pub(crate) zero_idx: u32,
    one_idx: u32,
    pub blocks: GateBlocks<P::ScalarField>, // Storage for wires and selectors for all gate types
    num_gates: usize,
    circuit_finalized: bool,
    pub contains_recursive_proof: bool,
    pub recursive_proof_public_input_indices: AggregationObjectPubInputIndices,
    rom_arrays: Vec<RomTranscript>,
    ram_arrays: Vec<RamTranscript>,
    pub(crate) lookup_tables: Vec<PlookupBasicTable<P::ScalarField>>,
    plookup: Plookup<P::ScalarField>,
    range_lists: HashMap<u64, RangeList>,
    cached_partial_non_native_field_multiplications:
        Vec<CachedPartialNonNativeFieldMultiplication<P::ScalarField>>,
    // Stores gate index of ROM and RAM reads (required by proving key)
    pub(crate) memory_read_records: Vec<u32>,
    // Stores gate index of RAM writes (required by proving key)
    pub(crate) memory_write_records: Vec<u32>,
}

impl<P: Pairing, S: UltraCircuitVariable<P::ScalarField>> GenericUltraCircuitBuilder<P, S> {
    pub(crate) const DUMMY_TAG: u32 = 0;
    pub(crate) const REAL_VARIABLE: u32 = u32::MAX - 1;
    pub(crate) const FIRST_VARIABLE_IN_CLASS: u32 = u32::MAX - 2;
    pub(crate) const UNINITIALIZED_MEMORY_RECORD: u32 = u32::MAX;
    pub(crate) const NUMBER_OF_GATES_PER_RAM_ACCESS: usize = 2;
    pub(crate) const NUMBER_OF_ARITHMETIC_GATES_PER_RAM_ARRAY: usize = 1;
    pub(crate) const NUM_RESERVED_GATES: usize = 4;
    // number of gates created per non-native field operation in process_non_native_field_multiplications
    pub(crate) const GATES_PER_NON_NATIVE_FIELD_MULTIPLICATION_ARITHMETIC: usize = 7;

    pub fn create_circuit(
        constraint_system: AcirFormat<P::ScalarField>,
        size_hint: usize,
        witness: Vec<S>,
        honk_recursion: bool,           // true for ultrahonk
        collect_gates_per_opcode: bool, // false for ultrahonk
    ) -> Self {
        tracing::info!("Builder create circuit");

        let has_valid_witness_assignments = !witness.is_empty();

        let mut builder = Self::init(
            size_hint,
            witness,
            constraint_system.public_inputs.to_owned(),
            constraint_system.varnum as usize,
            constraint_system.recursive,
        );

        builder.build_constraints(
            constraint_system,
            has_valid_witness_assignments,
            honk_recursion,
            collect_gates_per_opcode,
        );

        builder
    }

    fn new(size_hint: usize) -> Self {
        tracing::info!("Builder new");
        let variables = Vec::with_capacity(size_hint * 3);
        let variable_names = HashMap::with_capacity(size_hint * 3);
        let next_var_index = Vec::with_capacity(size_hint * 3);
        let prev_var_index = Vec::with_capacity(size_hint * 3);
        let real_variable_index = Vec::with_capacity(size_hint * 3);
        let real_variable_tags = Vec::with_capacity(size_hint * 3);

        Self {
            variables,
            variable_names,
            next_var_index,
            prev_var_index,
            real_variable_index,
            real_variable_tags,
            public_inputs: Vec::new(),
            is_recursive_circuit: false,
            tau: HashMap::new(),
            constant_variable_indices: HashMap::new(),
            zero_idx: 0,
            one_idx: 1,
            blocks: GateBlocks::default(),
            num_gates: 0,
            circuit_finalized: false,
            contains_recursive_proof: false,
            recursive_proof_public_input_indices: Default::default(),
            rom_arrays: Vec::new(),
            ram_arrays: Vec::new(),
            lookup_tables: Vec::new(),
            plookup: Default::default(),
            range_lists: HashMap::new(),
            cached_partial_non_native_field_multiplications: Vec::new(),
            memory_read_records: Vec::new(),
            memory_write_records: Vec::new(),
        }
    }

    /**
     * @brief Constructor from data generated from ACIR
     *
     * @param size_hint
     * @param witness_values witnesses values known to acir
     * @param public_inputs indices of public inputs in witness array
     * @param varnum number of known witness
     *
     * @note The size of witness_values may be less than varnum. The former is the set of actual witness values known at
     * the time of acir generation. The latter may be larger and essentially acounts for placeholders for witnesses that
     * we know will exist but whose values are not known during acir generation. Both are in general less than the total
     * number of variables/witnesses that might be present for a circuit generated from acir, since many gates will
     * depend on the details of the bberg implementation (or more generally on the backend used to process acir).
     */
    fn init(
        size_hint: usize,
        witness_values: Vec<S>,
        public_inputs: Vec<u32>,
        varnum: usize,
        recursive: bool,
    ) -> Self {
        tracing::info!("Builder init");
        let mut builder = Self::new(size_hint);

        // AZTEC TODO(https://github.com/AztecProtocol/barretenberg/issues/870): reserve space in blocks here somehow?

        let len = witness_values.len();
        for witness in witness_values.into_iter().take(varnum) {
            builder.add_variable(witness);
        }
        // Zeros are added for variables whose existence is known but whose values are not yet known. The values may
        // be "set" later on via the assert_equal mechanism.
        for _ in len..varnum {
            builder.add_variable(S::from_public(P::ScalarField::zero()));
        }

        // Add the public_inputs from acir
        builder.public_inputs = public_inputs;

        // Add the const zero variable after the acir witness has been
        // incorporated into variables.
        builder.zero_idx = builder.put_constant_variable(P::ScalarField::zero());
        builder.tau.insert(Self::DUMMY_TAG, Self::DUMMY_TAG); // AZTEC TODO(luke): explain this

        builder.is_recursive_circuit = recursive;
        builder
    }

    pub(crate) fn add_variable(&mut self, value: S) -> u32 {
        let idx = self.variables.len() as u32;
        self.variables.push(value);
        self.real_variable_index.push(idx);
        self.next_var_index.push(Self::REAL_VARIABLE);
        self.prev_var_index.push(Self::FIRST_VARIABLE_IN_CLASS);
        self.real_variable_tags.push(Self::DUMMY_TAG);
        idx
    }

    pub(crate) fn put_constant_variable(&mut self, variable: P::ScalarField) -> u32 {
        if let Some(val) = self.constant_variable_indices.get(&variable) {
            *val
        } else {
            let variable_index = self.add_variable(S::from_public(variable));
            self.fix_witness(variable_index, variable);
            self.constant_variable_indices
                .insert(variable, variable_index);
            variable_index
        }
    }

    fn create_poly_gate(&mut self, inp: &PolyTriple<P::ScalarField>) {
        self.assert_valid_variables(&[inp.a, inp.b, inp.c]);

        self.blocks
            .arithmetic
            .populate_wires(inp.a, inp.b, inp.c, self.zero_idx);
        self.blocks.arithmetic.q_m().push(inp.q_m);
        self.blocks.arithmetic.q_1().push(inp.q_l);
        self.blocks.arithmetic.q_2().push(inp.q_r);
        self.blocks.arithmetic.q_3().push(inp.q_o);
        self.blocks.arithmetic.q_c().push(inp.q_c);
        self.blocks
            .arithmetic
            .q_delta_range()
            .push(P::ScalarField::zero());

        self.blocks.arithmetic.q_arith().push(P::ScalarField::one());
        self.blocks.arithmetic.q_4().push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_lookup_type()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_elliptic()
            .push(P::ScalarField::zero());
        self.blocks.arithmetic.q_aux().push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_external()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_internal()
            .push(P::ScalarField::zero());

        self.check_selector_length_consistency();
        self.num_gates += 1;
    }

    fn create_big_mul_gate(&mut self, inp: &MulQuad<P::ScalarField>) {
        self.assert_valid_variables(&[inp.a, inp.b, inp.c, inp.d]);

        self.blocks
            .arithmetic
            .populate_wires(inp.a, inp.b, inp.c, inp.d);
        self.blocks.arithmetic.q_m().push(inp.mul_scaling);
        self.blocks.arithmetic.q_1().push(inp.a_scaling);
        self.blocks.arithmetic.q_2().push(inp.b_scaling);
        self.blocks.arithmetic.q_3().push(inp.c_scaling);
        self.blocks.arithmetic.q_c().push(inp.const_scaling);
        self.blocks.arithmetic.q_arith().push(P::ScalarField::one());
        self.blocks.arithmetic.q_4().push(inp.d_scaling);
        self.blocks
            .arithmetic
            .q_delta_range()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_lookup_type()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_elliptic()
            .push(P::ScalarField::zero());
        self.blocks.arithmetic.q_aux().push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_external()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_internal()
            .push(P::ScalarField::zero());

        self.check_selector_length_consistency();
        self.num_gates += 1;
    }

    pub(crate) fn create_add_gate(&mut self, inp: &AddTriple<P::ScalarField>) {
        self.assert_valid_variables(&[inp.a, inp.b, inp.c]);

        self.blocks
            .arithmetic
            .populate_wires(inp.a, inp.b, inp.c, self.zero_idx);
        self.blocks.arithmetic.q_m().push(P::ScalarField::zero());
        self.blocks.arithmetic.q_1().push(inp.a_scaling);
        self.blocks.arithmetic.q_2().push(inp.b_scaling);
        self.blocks.arithmetic.q_3().push(inp.c_scaling);
        self.blocks.arithmetic.q_c().push(inp.const_scaling);
        self.blocks.arithmetic.q_arith().push(P::ScalarField::one());
        self.blocks.arithmetic.q_4().push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_delta_range()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_lookup_type()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_elliptic()
            .push(P::ScalarField::zero());
        self.blocks.arithmetic.q_aux().push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_external()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_internal()
            .push(P::ScalarField::zero());

        self.check_selector_length_consistency();
        self.num_gates += 1;
    }

    pub(crate) fn create_big_add_gate(
        &mut self,
        inp: &AddQuad<P::ScalarField>,
        include_next_gate_w_4: bool,
    ) {
        self.assert_valid_variables(&[inp.a, inp.b, inp.c, inp.d]);

        self.blocks
            .arithmetic
            .populate_wires(inp.a, inp.b, inp.c, inp.d);
        self.blocks.arithmetic.q_m().push(P::ScalarField::zero());
        self.blocks.arithmetic.q_1().push(inp.a_scaling);
        self.blocks.arithmetic.q_2().push(inp.b_scaling);
        self.blocks.arithmetic.q_3().push(inp.c_scaling);
        self.blocks.arithmetic.q_c().push(inp.const_scaling);
        self.blocks
            .arithmetic
            .q_arith()
            .push(if include_next_gate_w_4 {
                P::ScalarField::from(2u64)
            } else {
                P::ScalarField::one()
            });
        self.blocks.arithmetic.q_4().push(inp.d_scaling);
        self.blocks
            .arithmetic
            .q_delta_range()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_lookup_type()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_elliptic()
            .push(P::ScalarField::zero());
        self.blocks.arithmetic.q_aux().push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_external()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_internal()
            .push(P::ScalarField::zero());

        self.check_selector_length_consistency();
        self.num_gates += 1;
    }

    fn create_block_constraints(
        &mut self,
        constraint: &BlockConstraint<P::ScalarField>,
        has_valid_witness_assignments: bool,
    ) {
        let mut init = Vec::with_capacity(constraint.init.len());
        for inp in constraint.init.iter() {
            let value = self.poly_to_field_ct(inp);
            init.push(value);
        }

        // Note: CallData/ReturnData not supported by Ultra; interpreted as ROM ops instead
        match constraint.type_ {
            BlockType::CallData | BlockType::ReturnData | BlockType::ROM => {
                self.process_rom_operations(constraint, has_valid_witness_assignments, init)
            }
            BlockType::RAM => todo!("BLOCK RAM constraint"),
        }
    }

    fn fix_witness(&mut self, witness_index: u32, witness_value: P::ScalarField) {
        self.assert_valid_variables(&[witness_index]);

        self.blocks.arithmetic.populate_wires(
            witness_index,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.blocks.arithmetic.q_m().push(P::ScalarField::zero());
        self.blocks.arithmetic.q_1().push(P::ScalarField::one());
        self.blocks.arithmetic.q_2().push(P::ScalarField::zero());
        self.blocks.arithmetic.q_3().push(P::ScalarField::zero());
        self.blocks.arithmetic.q_c().push(-witness_value);
        self.blocks.arithmetic.q_arith().push(P::ScalarField::one());
        self.blocks.arithmetic.q_4().push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_delta_range()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_lookup_type()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_elliptic()
            .push(P::ScalarField::zero());
        self.blocks.arithmetic.q_aux().push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_external()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_internal()
            .push(P::ScalarField::zero());

        self.check_selector_length_consistency();
        self.num_gates += 1;
    }

    fn assert_valid_variables(&self, variable_indices: &[u32]) {
        for variable_index in variable_indices.iter().cloned() {
            assert!(self.is_valid_variable(variable_index as usize));
        }
    }

    fn is_valid_variable(&self, variable_index: usize) -> bool {
        variable_index < self.variables.len()
    }

    fn check_selector_length_consistency(&self) {
        for block in self.blocks.get() {
            let nominal_size = block.selectors[0].len();
            for selector in block.selectors.iter().skip(1) {
                debug_assert_eq!(selector.len(), nominal_size);
            }
        }
    }

    fn build_constraints(
        &mut self,
        mut constraint_system: AcirFormat<P::ScalarField>,
        has_valid_witness_assignments: bool,
        honk_recursion: bool,
        collect_gates_per_opcode: bool,
    ) {
        tracing::info!("Builder build constraints");
        if collect_gates_per_opcode {
            constraint_system
                .gates_per_opcode
                .resize(constraint_system.num_acir_opcodes as usize, 0);
        }

        let mut gate_counter = GateCounter::new(collect_gates_per_opcode);

        // Add arithmetic gates
        for (i, constraint) in constraint_system.poly_triple_constraints.iter().enumerate() {
            self.create_poly_gate(constraint);
            gate_counter.track_diff(
                self,
                &mut constraint_system.gates_per_opcode,
                constraint_system
                    .original_opcode_indices
                    .poly_triple_constraints[i],
            );
        }
        for (i, constraint) in constraint_system.quad_constraints.iter().enumerate() {
            self.create_big_mul_gate(constraint);
            gate_counter.track_diff(
                self,
                &mut constraint_system.gates_per_opcode,
                constraint_system.original_opcode_indices.quad_constraints[i],
            );
        }

        // Add logic constraint
        // for (i, constraint) in constraint_system.logic_constraints.iter().enumerate() {
        //     todo!("Logic gates");
        // }

        // Add range constraint
        // for (i, constraint) in constraint_system.range_constraints.iter().enumerate() {
        //     todo!("rage gates");
        // }

        // Add aes128 constraints
        // for (i, constraint) in constraint_system.aes128_constraints.iter().enumerate() {
        //     todo!("aes128 gates");
        // }

        // Add sha256 constraints
        // for (i, constraint) in constraint_system.sha256_constraints.iter().enumerate() {
        //     todo!("sha256 gates");
        // }

        // for (i, constraint) in constraint_system.sha256_compression.iter().enumerate() {
        //     todo!("sha256 compression gates");
        // }

        // Add schnorr constraints
        // for (i, constraint) in constraint_system.schnorr_constraints.iter().enumerate() {
        //     todo!("schnorr gates");
        // }

        // Add ECDSA k1 constraints
        // for (i, constraint) in constraint_system.ecdsa_k1_constraints.iter().enumerate() {
        //     todo!("ecdsa k1 gates");
        // }

        // Add ECDSA r1 constraints
        // for (i, constraint) in constraint_system.ecdsa_r1_constraints.iter().enumerate() {
        //     todo!("ecdsa r1 gates");
        // }

        // Add blake2s constraints
        // for (i, constraint) in constraint_system.blake2s_constraints.iter().enumerate() {
        //     todo!("blake2s gates");
        // }

        // Add blake3 constraints
        // for (i, constraint) in constraint_system.blake3_constraints.iter().enumerate() {
        //     todo!("blake3 gates");
        // }

        // Add keccak constraints
        // for (i, constraint) in constraint_system.keccak_constraints.iter().enumerate() {
        //     todo!("keccak gates");
        // }

        // for (i, constraint) in constraint_system.keccak_permutations.iter().enumerate() {
        //     todo!("keccak permutation gates");
        // }

        // Add pedersen constraints
        // for (i, constraint) in constraint_system.pedersen_constraints.iter().enumerate() {
        //     todo!("pederson gates");
        // }

        // for (i, constraint) in constraint_system.pedersen_hash_constraints.iter().enumerate() {
        //     todo!("pedersen hash gates");
        // }

        // Add poseidon2 constraints
        // for (i, constraint) in constraint_system.poseidon2_constraints.iter().enumerate() {
        //     todo!("poseidon2 gates");
        // }

        // Add multi scalar mul constraints
        // for (i, constraint) in constraint_system.multi_scalar_mul_constraints.iter().enumerate() {
        //     todo!("multi scalar mul gates");
        // }

        // Add ec add constraints
        // for (i, constraint) in constraint_system.ec_add_constraints.iter().enumerate() {
        //     todo!("ec add gates");
        // }

        // Add block constraints
        for (i, constraint) in constraint_system.block_constraints.iter().enumerate() {
            self.create_block_constraints(constraint, has_valid_witness_assignments);
            if collect_gates_per_opcode {
                let avg_gates_per_opcode = gate_counter.compute_diff(self)
                    / constraint_system.original_opcode_indices.block_constraints[i].len();
                for opcode_index in constraint_system.original_opcode_indices.block_constraints[i]
                    .iter()
                    .cloned()
                {
                    constraint_system.gates_per_opcode[opcode_index] = avg_gates_per_opcode;
                }
            }
        }

        // Add big_int constraints
        // for (i, constraint) in constraint_system.bigint_from_le_bytes_constraints.iter().enumerate() {
        //     todo!("bigint from le bytes gates");
        // }

        // for (i, constraint) in constraint_system.bigint_operations.iter().enumerate() {
        //     todo!("bigint operations gates");
        // }

        // for (i, constraint) in constraint_system.bigint_to_le_bytes_constraints.iter().enumerate() {
        //     todo!("bigint to le bytes gates");
        // }

        // assert equals
        for (i, constraint) in constraint_system.assert_equalities.iter().enumerate() {
            todo!("assert equalities gates");
        }

        // RecursionConstraints
        self.process_plonk_recursion_constraints(
            &constraint_system,
            has_valid_witness_assignments,
            &mut gate_counter,
        );
        self.process_honk_recursion_constraints(
            &constraint_system,
            has_valid_witness_assignments,
            &mut gate_counter,
        );
        self.process_avm_recursion_constraints(
            &constraint_system,
            has_valid_witness_assignments,
            &mut gate_counter,
        );

        // If the circuit does not itself contain honk recursion constraints but is going to be
        // proven with honk then recursively verified, add a default aggregation object
        if constraint_system.honk_recursion_constraints.is_empty()
            && honk_recursion
            && self.is_recursive_circuit
        {
            // Set a default aggregation object if we don't have one.
            let current_aggregation_object = self.init_default_agg_obj_indices();
            // Make sure the verification key records the public input indices of the
            // final recursion output.
            self.add_recursive_proof(current_aggregation_object);
        }
    }

    fn process_plonk_recursion_constraints(
        &mut self,
        constraint_system: &AcirFormat<P::ScalarField>,
        has_valid_witness_assignments: bool,
        gate_counter: &mut GateCounter,
    ) {
        for (i, constraint) in constraint_system.recursion_constraints.iter().enumerate() {
            todo!("Plonk recursion");
        }
    }

    fn process_honk_recursion_constraints(
        &mut self,
        constraint_system: &AcirFormat<P::ScalarField>,
        has_valid_witness_assignments: bool,
        gate_counter: &mut GateCounter,
    ) {
        {
            for (i, constraint) in constraint_system
                .honk_recursion_constraints
                .iter()
                .enumerate()
            {
                todo!("Honk recursion");
            }
        }
    }

    fn process_avm_recursion_constraints(
        &mut self,
        constraint_system: &AcirFormat<P::ScalarField>,
        has_valid_witness_assignments: bool,
        gate_counter: &mut GateCounter,
    ) {
        let current_aggregation_object = self.init_default_agg_obj_indices();

        for (i, constraint) in constraint_system
            .avm_recursion_constraints
            .iter()
            .enumerate()
        {
            todo!("avm recursion");
        }
    }

    pub(crate) fn get_num_gates(&self) -> usize {
        // if circuit finalized already added extra gates
        if self.circuit_finalized {
            return self.num_gates;
        }
        let mut count = 0;
        let mut rangecount = 0;
        let mut romcount = 0;
        let mut ramcount = 0;
        let mut nnfcount = 0;
        self.get_num_gates_split_into_components(
            &mut count,
            &mut rangecount,
            &mut romcount,
            &mut ramcount,
            &mut nnfcount,
        );
        count + romcount + ramcount + rangecount + nnfcount
    }

    pub(crate) fn get_tables_size(&self) -> usize {
        let mut tables_size = 0;
        for table in self.lookup_tables.iter() {
            tables_size += table.len();
        }

        tables_size
    }

    fn get_lookups_size(&self) -> usize {
        let mut lookups_size = 0;
        for table in self.lookup_tables.iter() {
            lookups_size += table.lookup_gates.len();
        }
        lookups_size
    }

    fn get_num_gates_split_into_components(
        &self,
        count: &mut usize,
        rangecount: &mut usize,
        romcount: &mut usize,
        ramcount: &mut usize,
        nnfcount: &mut usize,
    ) {
        *count = self.num_gates;

        // each ROM gate adds +1 extra gate due to the rom reads being copied to a sorted list set
        for rom_array in self.rom_arrays.iter() {
            for state in rom_array.state.iter() {
                if state[0] == Self::UNINITIALIZED_MEMORY_RECORD {
                    *romcount += 2;
                }
            }
            *romcount += rom_array.records.len();
            *romcount += 1; // we add an addition gate after procesing a rom array
        }

        // each RAM gate adds +2 extra gates due to the ram reads being copied to a sorted list set,
        // as well as an extra gate to validate timestamps
        let mut ram_timestamps = Vec::with_capacity(self.ram_arrays.len());
        let mut ram_range_sizes = Vec::with_capacity(self.ram_arrays.len());
        let mut ram_range_exists = Vec::with_capacity(self.ram_arrays.len());
        for ram_array in self.ram_arrays.iter() {
            for state in ram_array.state.iter() {
                if *state == Self::UNINITIALIZED_MEMORY_RECORD {
                    *ramcount += Self::NUMBER_OF_GATES_PER_RAM_ACCESS;
                }
            }
            *ramcount += ram_array.records.len() * Self::NUMBER_OF_GATES_PER_RAM_ACCESS;
            *ramcount += Self::NUMBER_OF_ARITHMETIC_GATES_PER_RAM_ARRAY; // we add an addition gate after procesing a ram array

            // there will be 'max_timestamp' number of range checks, need to calculate.
            let max_timestamp = ram_array.access_count - 1;

            // if a range check of length `max_timestamp` already exists, we are double counting.
            // We record `ram_timestamps` to detect and correct for this error when we process range lists.
            ram_timestamps.push(max_timestamp);
            let mut padding = (NUM_WIRES - (max_timestamp % NUM_WIRES)) % NUM_WIRES;
            if max_timestamp == NUM_WIRES {
                padding += NUM_WIRES;
            }
            let ram_range_check_list_size = max_timestamp + padding;

            let mut ram_range_check_gate_count = ram_range_check_list_size / NUM_WIRES;
            ram_range_check_gate_count += 1; // we need to add 1 extra addition gates for every distinct range list

            ram_range_sizes.push(ram_range_check_gate_count);
            ram_range_exists.push(false);
        }
        for list in self.range_lists.iter() {
            let mut list_size = list.1.variable_indices.len();
            let mut padding = (NUM_WIRES - (list_size % NUM_WIRES)) % NUM_WIRES;
            if list_size == NUM_WIRES {
                padding += NUM_WIRES;
            }
            list_size += padding;

            for (time_stamp, ram_range_exist) in ram_timestamps
                .iter()
                .cloned()
                .zip(ram_range_exists.iter_mut())
            {
                if list.1.target_range as usize == time_stamp {
                    *ram_range_exist = true;
                }
            }

            *rangecount += list_size / NUM_WIRES;
            *rangecount += 1; // we need to add 1 extra addition gates for every distinct range list
        }
        // update rangecount to include the ram range checks the composer will eventually be creating
        for (ram_range_sizes, ram_range_exist) in ram_range_sizes
            .into_iter()
            .zip(ram_range_exists.into_iter())
        {
            if !ram_range_exist {
                *rangecount += ram_range_sizes;
            }
        }

        let mut nnf_copy = self.cached_partial_non_native_field_multiplications.clone();
        // update nnfcount
        nnf_copy.sort();

        nnf_copy.dedup();
        let num_nnf_ops = nnf_copy.len();
        *nnfcount = num_nnf_ops * Self::GATES_PER_NON_NATIVE_FIELD_MULTIPLICATION_ARITHMETIC;
    }

    fn add_recursive_proof(&mut self, proof_output_witness_indices: AggregationObjectIndices) {
        if self.contains_recursive_proof {
            panic!("added recursive proof when one already exists");
        }
        self.contains_recursive_proof = true;

        for (i, idx) in proof_output_witness_indices.into_iter().enumerate() {
            self.set_public_input(idx);
            self.recursive_proof_public_input_indices[i] = self.public_inputs.len() as u32 - 1;
        }
    }

    fn set_public_input(&mut self, witness_index: u32) {
        for public_input in self.public_inputs.iter().cloned() {
            if public_input == witness_index {
                panic!("Attempted to set a public input that is already public!");
            }
        }
        self.public_inputs.push(witness_index);
    }

    fn init_default_agg_obj_indices(&mut self) -> AggregationObjectIndices {
        const NUM_LIMBS: usize = 4;
        const NUM_LIMB_BITS: u32 = 68;

        let mask = (BigUint::one() << NUM_LIMB_BITS) - BigUint::one();

        // AZTEC TODO(https://github.com/AztecProtocol/barretenberg/issues/911): These are pairing points extracted from a valid
        // proof. This is a workaround because we can't represent the point at infinity in biggroup yet.
        let mut agg_obj_indices = AggregationObjectIndices::default();
        let x0 = Utils::field_from_hex_string::<P::BaseField>(
            "0x031e97a575e9d05a107acb64952ecab75c020998797da7842ab5d6d1986846cf",
        )
        .expect("x0 works");
        let y0 = Utils::field_from_hex_string::<P::BaseField>(
            "0x178cbf4206471d722669117f9758a4c410db10a01750aebb5666547acf8bd5a4",
        )
        .expect("y0 works");
        let x1 = Utils::field_from_hex_string::<P::BaseField>(
            "0x0f94656a2ca489889939f81e9c74027fd51009034b3357f0e91b8a11e7842c38",
        )
        .expect("x1 works");
        let y1 = Utils::field_from_hex_string::<P::BaseField>(
            "0x1b52c2020d7464a0c80c0da527a08193fe27776f50224bd6fb128b46c1ddb67f",
        )
        .expect("y1 works");

        let mut agg_obj_indices_idx = 0;
        let aggregation_object_fq_values = [x0, y0, x1, y1];

        for val in aggregation_object_fq_values {
            let mut x: BigUint = val.into();
            let x0 = &x & &mask;
            x >>= NUM_LIMB_BITS;
            let x1 = &x & &mask;
            x >>= NUM_LIMB_BITS;
            let x2 = &x & &mask;
            x >>= NUM_LIMB_BITS;
            let x3 = x;

            let val_limbs: [P::ScalarField; NUM_LIMBS] = [
                P::ScalarField::from(x0),
                P::ScalarField::from(x1),
                P::ScalarField::from(x2),
                P::ScalarField::from(x3),
            ];

            for val in val_limbs {
                let idx = self.add_variable(S::from_public(val));
                agg_obj_indices[agg_obj_indices_idx] = idx;
                agg_obj_indices_idx += 1;
            }
        }
        agg_obj_indices
    }

    fn poly_to_field_ct(&self, poly: &PolyTriple<P::ScalarField>) -> FieldCT<P::ScalarField> {
        assert!(poly.q_m.is_zero());
        assert!(poly.q_r.is_zero());
        assert!(poly.q_o.is_zero());
        if poly.q_l.is_zero() {
            return FieldCT::from(poly.q_c);
        }

        let mut x = FieldCT::from_witness_index(poly.a);
        x.additive_constant = poly.q_c;
        x.multiplicative_constant = poly.q_l;
        x
    }

    fn process_rom_operations(
        &mut self,
        constraint: &BlockConstraint<P::ScalarField>,
        has_valid_witness_assignments: bool,
        init: Vec<FieldCT<P::ScalarField>>,
    ) {
        let mut table = RomTable::new(init);

        // AZTEC TODO this is just implemented for the Plain backend
        for op in constraint.trace.iter() {
            assert_eq!(op.access_type, 0);
            let value = self.poly_to_field_ct(&op.value);
            let index = self.poly_to_field_ct(&op.index);
            // For a ROM table, constant read should be optimized out:
            // The rom_table won't work with a constant read because the table may not be initialized
            assert!(!op.index.q_l.is_zero());
            // We create a new witness w to avoid issues with non-valid witness assignements:
            // if witness are not assigned, then w will be zero and table[w] will work
            let w_value = if has_valid_witness_assignments {
                // If witness are assigned, we use the correct value for w
                index.get_value(self)
            } else {
                P::ScalarField::zero()
            };
            let w = FieldCT::from_witness(w_value, self);
            value.assert_equal(&table.index_field_ct(&w, self), self);
            w.assert_equal(&index, self);
        }
    }

    pub fn get_variable(&self, index: usize) -> S {
        assert!(self.variables.len() > index);
        self.variables[self.real_variable_index[index] as usize].to_owned()
    }

    pub(crate) fn assert_equal_constant(&mut self, a_idx: usize, b: P::ScalarField) {
        assert_eq!(self.variables[a_idx], S::from_public(b));
        let b_idx = self.put_constant_variable(b);
        self.assert_equal(a_idx, b_idx as usize);
    }

    pub(crate) fn assert_equal(&mut self, a_idx: usize, b_idx: usize) {
        self.is_valid_variable(a_idx);
        self.is_valid_variable(b_idx);
        assert_eq!(self.get_variable(a_idx), self.get_variable(b_idx));

        let a_real_idx = self.real_variable_index[a_idx] as usize;
        let b_real_idx = self.real_variable_index[b_idx] as usize;
        // If a==b is already enforced, exit method
        if a_real_idx == b_real_idx {
            return;
        }
        // Otherwise update the real_idx of b-chain members to that of a

        let b_start_idx = self.get_first_variable_in_class(b_idx);
        self.update_real_variable_indices(b_start_idx, a_real_idx as u32);
        // Now merge equivalence classes of a and b by tying last (= real) element of b-chain to first element of a-chain
        let a_start_idx = self.get_first_variable_in_class(a_idx);
        self.next_var_index[b_real_idx] = a_start_idx as u32;
        self.prev_var_index[a_start_idx] = b_real_idx as u32;
        assert!(
            self.real_variable_tags[a_real_idx] == Self::DUMMY_TAG
                || self.real_variable_tags[b_real_idx] == Self::DUMMY_TAG
                || self.real_variable_tags[a_real_idx] == self.real_variable_tags[b_real_idx]
        );

        if self.real_variable_tags[a_real_idx] == Self::DUMMY_TAG {
            self.real_variable_tags[a_real_idx] = self.real_variable_tags[b_real_idx];
        }
    }

    fn get_first_variable_in_class(&self, mut index: usize) -> usize {
        while self.prev_var_index[index] != Self::FIRST_VARIABLE_IN_CLASS {
            index = self.prev_var_index[index] as usize;
        }
        index
    }

    fn update_real_variable_indices(&mut self, index: usize, new_real_index: u32) {
        let mut cur_index = index;
        loop {
            self.real_variable_index[cur_index] = new_real_index;
            cur_index = self.next_var_index[cur_index] as usize;
            if cur_index == Self::REAL_VARIABLE as usize {
                break;
            }
        }
    }

    pub(crate) fn create_rom_array(&mut self, array_size: usize) -> usize {
        let mut new_transcript = RomTranscript::default();
        for _ in 0..array_size {
            new_transcript.state.push([
                Self::UNINITIALIZED_MEMORY_RECORD,
                Self::UNINITIALIZED_MEMORY_RECORD,
            ]);
        }
        self.rom_arrays.push(new_transcript);
        self.rom_arrays.len() - 1
    }

    pub(crate) fn set_rom_element(
        &mut self,
        rom_id: usize,
        index_value: usize,
        value_witness: u32,
    ) {
        assert!(self.rom_arrays.len() > rom_id);
        let index_witness = if index_value == 0 {
            self.zero_idx
        } else {
            self.put_constant_variable(P::ScalarField::from(index_value as u64))
        };

        assert!(self.rom_arrays[rom_id].state.len() > index_value);
        assert!(self.rom_arrays[rom_id].state[index_value][0] == Self::UNINITIALIZED_MEMORY_RECORD);
        // /**
        // * The structure MemoryRecord contains the following members in this order:
        // *   uint32_t index_witness;
        // *   uint32_t timestamp_witness;
        // *   uint32_t value_witness;
        // *   uint32_t index;
        // *   uint32_t timestamp;
        // *   uint32_t record_witness;
        // *   size_t gate_index;
        // * The second initialization value here is the witness, because in ROM it doesn't matter. We will decouple this
        // * logic later.
        // */
        let mut new_record = RomRecord {
            index_witness,
            value_column1_witness: value_witness,
            value_column2_witness: self.zero_idx,
            index: index_value as u32,
            record_witness: 0,
            gate_index: 0,
        };

        self.rom_arrays[rom_id].state[index_value][0] = value_witness;
        self.rom_arrays[rom_id].state[index_value][1] = self.zero_idx;
        self.create_rom_gate(&mut new_record);
        self.rom_arrays[rom_id].records.push(new_record);
    }

    fn create_rom_gate(&mut self, record: &mut RomRecord) {
        // Record wire value can't yet be computed
        record.record_witness = self.add_variable(S::from_public(P::ScalarField::zero()));
        self.apply_aux_selectors(AuxSelectors::RomRead);
        self.blocks.aux.populate_wires(
            record.index_witness,
            record.value_column1_witness,
            record.value_column2_witness,
            record.record_witness,
        );

        // Note: record the index into the block that contains the RAM/ROM gates
        record.gate_index = self.blocks.aux.len() - 1;
        self.num_gates += 1;
    }

    pub(crate) fn read_rom_array(&mut self, rom_id: usize, index_witness: u32) -> u32 {
        assert!(self.rom_arrays.len() > rom_id);
        let val: BigUint = self
            .get_variable(index_witness as usize)
            .public_into_field()
            .into();
        let index: usize = val.try_into().unwrap();

        assert!(self.rom_arrays[rom_id].state.len() > index);
        assert!(self.rom_arrays[rom_id].state[index][0] != Self::UNINITIALIZED_MEMORY_RECORD);
        let value = self.get_variable(self.rom_arrays[rom_id].state[index][0] as usize);
        let value_witness = self.add_variable(value);
        let mut new_record = RomRecord {
            index_witness,
            value_column1_witness: value_witness,
            value_column2_witness: self.zero_idx,
            index: index as u32,
            record_witness: 0,
            gate_index: 0,
        };
        self.create_rom_gate(&mut new_record);
        self.rom_arrays[rom_id].records.push(new_record);

        // create_read_gate
        value_witness
    }

    fn apply_aux_selectors(&mut self, type_: AuxSelectors) {
        let block = &mut self.blocks.aux;
        block.q_aux().push(if type_ == AuxSelectors::None {
            P::ScalarField::zero()
        } else {
            P::ScalarField::one()
        });
        // Set to zero the selectors that are not enabled for this gate
        block.q_delta_range().push(P::ScalarField::zero());
        block.q_lookup_type().push(P::ScalarField::zero());
        block.q_elliptic().push(P::ScalarField::zero());
        block.q_poseidon2_external().push(P::ScalarField::zero());
        block.q_poseidon2_internal().push(P::ScalarField::zero());

        match type_ {
            AuxSelectors::RomRead => {
                // Memory read gate for reading memory cells.
                // Validates record witness computation (r = read_write_flag + index * \eta + timestamp * \eta^2 + value *
                // \eta^3)
                block.q_1().push(P::ScalarField::one());
                block.q_2().push(P::ScalarField::zero());
                block.q_3().push(P::ScalarField::zero());
                block.q_4().push(P::ScalarField::zero());
                block.q_m().push(P::ScalarField::one()); // validate record witness is correctly computed
                block.q_c().push(P::ScalarField::zero()); // read/write flag stored in q_c
                block.q_arith().push(P::ScalarField::zero());

                self.check_selector_length_consistency();
            }
            _ => todo!("Aux selectors"),
        }
    }

    fn create_dummy_gate(
        // &mut self,
        block: &mut UltraTraceBlock<P::ScalarField>,
        idx_1: u32,
        idx_2: u32,
        idx_3: u32,
        idx_4: u32,
    ) {
        block.populate_wires(idx_1, idx_2, idx_3, idx_4);
        block.q_m().push(P::ScalarField::zero());
        block.q_1().push(P::ScalarField::zero());
        block.q_2().push(P::ScalarField::zero());
        block.q_3().push(P::ScalarField::zero());
        block.q_c().push(P::ScalarField::zero());
        block.q_arith().push(P::ScalarField::zero());
        block.q_4().push(P::ScalarField::zero());
        block.q_delta_range().push(P::ScalarField::zero());
        block.q_elliptic().push(P::ScalarField::zero());
        block.q_lookup_type().push(P::ScalarField::zero());
        block.q_aux().push(P::ScalarField::zero());
        block.q_poseidon2_external().push(P::ScalarField::zero());
        block.q_poseidon2_internal().push(P::ScalarField::zero());

        // TACEO TODO these are uncommented due to mutability issues
        // Taken care of by the caller
        // self.check_selector_length_consistency();
        // self.num_gates += 1;
    }

    pub fn add_gates_to_ensure_all_polys_are_non_zero(&mut self) {
        // q_m, q_1, q_2, q_3, q_4
        self.blocks.arithmetic.populate_wires(
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.blocks.arithmetic.q_m().push(P::ScalarField::one());
        self.blocks.arithmetic.q_1().push(P::ScalarField::one());
        self.blocks.arithmetic.q_2().push(P::ScalarField::one());
        self.blocks.arithmetic.q_3().push(P::ScalarField::one());
        self.blocks.arithmetic.q_4().push(P::ScalarField::one());
        self.blocks.arithmetic.q_c().push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_delta_range()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_arith()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_lookup_type()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_elliptic()
            .push(P::ScalarField::zero());
        self.blocks.arithmetic.q_aux().push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_external()
            .push(P::ScalarField::zero());
        self.blocks
            .arithmetic
            .q_poseidon2_internal()
            .push(P::ScalarField::zero());

        self.check_selector_length_consistency();
        self.num_gates += 1;

        // q_delta_range
        self.blocks.delta_range.populate_wires(
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.blocks.delta_range.q_m().push(P::ScalarField::zero());
        self.blocks.delta_range.q_1().push(P::ScalarField::zero());
        self.blocks.delta_range.q_2().push(P::ScalarField::zero());
        self.blocks.delta_range.q_3().push(P::ScalarField::zero());
        self.blocks.delta_range.q_4().push(P::ScalarField::zero());
        self.blocks.delta_range.q_c().push(P::ScalarField::zero());
        self.blocks
            .delta_range
            .q_delta_range()
            .push(P::ScalarField::one());
        self.blocks
            .delta_range
            .q_arith()
            .push(P::ScalarField::zero());
        self.blocks
            .delta_range
            .q_lookup_type()
            .push(P::ScalarField::zero());
        self.blocks
            .delta_range
            .q_elliptic()
            .push(P::ScalarField::zero());
        self.blocks.delta_range.q_aux().push(P::ScalarField::zero());
        self.blocks
            .delta_range
            .q_poseidon2_external()
            .push(P::ScalarField::zero());
        self.blocks
            .delta_range
            .q_poseidon2_internal()
            .push(P::ScalarField::zero());

        self.check_selector_length_consistency();
        self.num_gates += 1;
        Self::create_dummy_gate(
            &mut self.blocks.delta_range,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.check_selector_length_consistency();
        self.num_gates += 1; // necessary because create dummy gate cannot increment num_gates itself

        // q_elliptic
        self.blocks.elliptic.populate_wires(
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.blocks.elliptic.q_m().push(P::ScalarField::zero());
        self.blocks.elliptic.q_1().push(P::ScalarField::zero());
        self.blocks.elliptic.q_2().push(P::ScalarField::zero());
        self.blocks.elliptic.q_3().push(P::ScalarField::zero());
        self.blocks.elliptic.q_4().push(P::ScalarField::zero());
        self.blocks.elliptic.q_c().push(P::ScalarField::zero());
        self.blocks
            .elliptic
            .q_delta_range()
            .push(P::ScalarField::zero());
        self.blocks.elliptic.q_arith().push(P::ScalarField::zero());
        self.blocks
            .elliptic
            .q_lookup_type()
            .push(P::ScalarField::zero());
        self.blocks
            .elliptic
            .q_elliptic()
            .push(P::ScalarField::one());
        self.blocks.elliptic.q_aux().push(P::ScalarField::zero());
        self.blocks
            .elliptic
            .q_poseidon2_external()
            .push(P::ScalarField::zero());
        self.blocks
            .elliptic
            .q_poseidon2_internal()
            .push(P::ScalarField::zero());

        self.check_selector_length_consistency();
        self.num_gates += 1;
        Self::create_dummy_gate(
            &mut self.blocks.elliptic,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.check_selector_length_consistency();
        self.num_gates += 1; // necessary because create dummy gate cannot increment num_gates itself

        // q_aux
        self.blocks
            .aux
            .populate_wires(self.zero_idx, self.zero_idx, self.zero_idx, self.zero_idx);
        self.blocks.aux.q_m().push(P::ScalarField::zero());
        self.blocks.aux.q_1().push(P::ScalarField::zero());
        self.blocks.aux.q_2().push(P::ScalarField::zero());
        self.blocks.aux.q_3().push(P::ScalarField::zero());
        self.blocks.aux.q_4().push(P::ScalarField::zero());
        self.blocks.aux.q_c().push(P::ScalarField::zero());
        self.blocks.aux.q_delta_range().push(P::ScalarField::zero());
        self.blocks.aux.q_arith().push(P::ScalarField::zero());
        self.blocks.aux.q_lookup_type().push(P::ScalarField::zero());
        self.blocks.aux.q_elliptic().push(P::ScalarField::zero());
        self.blocks.aux.q_aux().push(P::ScalarField::one());
        self.blocks
            .aux
            .q_poseidon2_external()
            .push(P::ScalarField::zero());
        self.blocks
            .aux
            .q_poseidon2_internal()
            .push(P::ScalarField::zero());

        self.check_selector_length_consistency();
        self.num_gates += 1;
        Self::create_dummy_gate(
            &mut self.blocks.aux,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.check_selector_length_consistency();
        self.num_gates += 1; // necessary because create dummy gate cannot increment num_gates itself

        // Add nonzero values in w_4 and q_c (q_4*w_4 + q_c --> 1*1 - 1 = 0)
        self.one_idx = self.put_constant_variable(P::ScalarField::one());
        self.create_big_add_gate(
            &AddQuad {
                a: self.zero_idx,
                b: self.zero_idx,
                c: self.zero_idx,
                d: self.one_idx,
                a_scaling: P::ScalarField::zero(),
                b_scaling: P::ScalarField::zero(),
                c_scaling: P::ScalarField::zero(),
                d_scaling: P::ScalarField::one(),
                const_scaling: -P::ScalarField::one(),
            },
            false,
        );

        // Take care of all polys related to lookups (q_lookup, tables, sorted, etc)
        // by doing a dummy lookup with a special table.
        // Note: the 4th table poly is the table index: this is not the value of the table
        // type enum but rather the index of the table in the list of all tables utilized
        // in the circuit. Therefore we naively need two different basic tables (indices 0, 1)
        // to get a non-zero value in table_4.
        // The multitable operates on 2-bit values, so the maximum is 3
        let left_value = 3;
        let right_value = 3;

        let left_witness_value = P::ScalarField::from(left_value as u64);
        let right_witness_value = P::ScalarField::from(right_value as u64);

        let left_witness_index = self.add_variable(S::from_public(left_witness_value));
        let right_witness_index = self.add_variable(S::from_public(right_witness_value));
        let dummy_accumulators = self.plookup.get_lookup_accumulators(
            MultiTableId::HonkDummyMulti,
            left_witness_value,
            right_witness_value,
            true,
        );
        self.create_gates_from_plookup_accumulators(
            MultiTableId::HonkDummyMulti,
            dummy_accumulators,
            left_witness_index,
            Some(right_witness_index),
        );

        // mock a poseidon external gate, with all zeros as input
        self.blocks.poseidon_external.populate_wires(
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.blocks
            .poseidon_external
            .q_m()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_1()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_2()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_3()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_c()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_arith()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_4()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_delta_range()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_lookup_type()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_elliptic()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_aux()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_external
            .q_poseidon2_external()
            .push(P::ScalarField::one());
        self.blocks
            .poseidon_external
            .q_poseidon2_internal()
            .push(P::ScalarField::zero());

        self.check_selector_length_consistency();
        self.num_gates += 1;

        // dummy gate to be read into by previous poseidon external gate via shifts
        Self::create_dummy_gate(
            &mut self.blocks.poseidon_external,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.check_selector_length_consistency();
        self.num_gates += 1; // necessary because create dummy gate cannot increment num_gates itself

        // mock a poseidon internal gate, with all zeros as input
        self.blocks.poseidon_internal.populate_wires(
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.blocks
            .poseidon_internal
            .q_m()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_1()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_2()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_3()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_c()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_arith()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_4()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_delta_range()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_lookup_type()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_elliptic()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_aux()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_poseidon2_external()
            .push(P::ScalarField::zero());
        self.blocks
            .poseidon_internal
            .q_poseidon2_internal()
            .push(P::ScalarField::one());

        self.check_selector_length_consistency();
        self.num_gates += 1;

        // dummy gate to be read into by previous poseidon internal gate via shifts
        Self::create_dummy_gate(
            &mut self.blocks.poseidon_internal,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
            self.zero_idx,
        );
        self.check_selector_length_consistency();
        self.num_gates += 1; // necessary because create dummy gate cannot increment num_gates itself
    }

    pub fn get_num_gates_added_to_ensure_nonzero_polynomials() -> usize {
        let mut builder = Self::new(0);
        let num_gates_prior = builder.get_num_gates();
        builder.add_gates_to_ensure_all_polys_are_non_zero();
        let num_gates_post = builder.get_num_gates(); // accounts for finalization gates

        num_gates_post - num_gates_prior
    }

    pub fn get_circuit_subgroup_size(num_gates: usize) -> usize {
        let mut log2_n = Utils::get_msb64(num_gates as u64);
        if (1 << log2_n) != num_gates {
            log2_n += 1;
        }
        1 << log2_n
    }

    pub fn get_total_circuit_size(&self) -> usize {
        let minimum_circuit_size = self.get_tables_size() + self.get_lookups_size();
        let num_filled_gates = self.get_num_gates() + self.public_inputs.len();
        std::cmp::max(minimum_circuit_size, num_filled_gates) + Self::NUM_RESERVED_GATES
    }

    fn get_table(&mut self, id: BasicTableId) -> &mut PlookupBasicTable<P::ScalarField> {
        let mut index = self.lookup_tables.len();
        for (i, table) in self.lookup_tables.iter().enumerate() {
            if table.id == id {
                index = i;
                break;
            }
        }

        let len = self.lookup_tables.len();
        if index == len {
            // Table doesn't exist! So try to create it.
            self.lookup_tables
                .push(PlookupBasicTable::create_basic_table(id, len));
            self.lookup_tables.last_mut().unwrap()
        } else {
            &mut self.lookup_tables[index]
        }
    }

    fn create_gates_from_plookup_accumulators(
        &mut self,
        id: MultiTableId,
        read_values: ReadData<P::ScalarField>,
        key_a_index: u32,
        key_b_index: Option<u32>,
    ) -> ReadData<u32> {
        let id_usize = id as usize;

        let num_lookups = read_values[ColumnIdx::C1].len();
        let mut read_data = ReadData::default();

        for i in 0..num_lookups {
            // get basic lookup table; construct and add to builder.lookup_tables if not already present

            let basic_table_id = self.plookup.multi_tables[id_usize].basic_table_ids[i].clone();
            let table = self.get_table(basic_table_id);

            table
                .lookup_gates
                .push(read_values.lookup_entries[i].to_owned()); // used for constructing sorted polynomials
            let table_index = table.table_index;

            let first_idx = if i == 0 {
                key_a_index
            } else {
                self.add_variable(S::from_public(read_values[ColumnIdx::C1][i]))
            };

            #[allow(clippy::unnecessary_unwrap)]
            let second_idx = if i == 0 && (key_b_index.is_some()) {
                key_b_index.unwrap()
            } else {
                self.add_variable(S::from_public(read_values[ColumnIdx::C2][i]))
            };
            let third_idx = self.add_variable(S::from_public(read_values[ColumnIdx::C3][i]));

            read_data[ColumnIdx::C1].push(first_idx);
            read_data[ColumnIdx::C2].push(second_idx);
            read_data[ColumnIdx::C3].push(third_idx);
            self.assert_valid_variables(&[first_idx, second_idx, third_idx]);

            self.blocks
                .lookup
                .q_lookup_type()
                .push(P::ScalarField::one());
            self.blocks
                .lookup
                .q_3()
                .push(P::ScalarField::from(table_index as u64));
            self.blocks
                .lookup
                .populate_wires(first_idx, second_idx, third_idx, self.zero_idx);
            self.blocks.lookup.q_1().push(P::ScalarField::zero());
            self.blocks.lookup.q_2().push(if i == (num_lookups - 1) {
                P::ScalarField::zero()
            } else {
                -self.plookup.multi_tables[id_usize].column_1_step_sizes[i + 1]
            });
            self.blocks.lookup.q_m().push(if i == (num_lookups - 1) {
                P::ScalarField::zero()
            } else {
                -self.plookup.multi_tables[id_usize].column_2_step_sizes[i + 1]
            });
            self.blocks.lookup.q_c().push(if i == (num_lookups - 1) {
                P::ScalarField::zero()
            } else {
                -self.plookup.multi_tables[id_usize].column_3_step_sizes[i + 1]
            });
            self.blocks.lookup.q_arith().push(P::ScalarField::zero());
            self.blocks.lookup.q_4().push(P::ScalarField::zero());
            self.blocks
                .lookup
                .q_delta_range()
                .push(P::ScalarField::zero());
            self.blocks.lookup.q_elliptic().push(P::ScalarField::zero());
            self.blocks.lookup.q_aux().push(P::ScalarField::zero());
            self.blocks
                .lookup
                .q_poseidon2_external()
                .push(P::ScalarField::zero());
            self.blocks
                .lookup
                .q_poseidon2_internal()
                .push(P::ScalarField::zero());

            self.check_selector_length_consistency();
            self.num_gates += 1;
        }
        read_data
    }

    pub fn finalize_circuit(&mut self, ensure_nonzero: bool) {
        // /**
        //  * First of all, add the gates related to ROM arrays and range lists.
        //  * Note that the total number of rows in an UltraPlonk program can be divided as following:
        //  *  1. arithmetic gates:  n_computation (includes all computation gates)
        //  *  2. rom/memory gates:  n_rom
        //  *  3. range list gates:  n_range
        //  *  4. public inputs:     n_pub
        //  *
        //  * Now we have two variables referred to as `n` in the code:
        //  *  1. ComposerBase::n => refers to the size of the witness of a given program,
        //  *  2. proving_key::n => the next power of two ≥ total witness size.
        //  *
        //  * In this case, we have composer.num_gates = n_computation before we execute the following two functions.
        //  * After these functions are executed, the composer's `n` is incremented to include the ROM
        //  * and range list gates. Therefore we have:
        //  * composer.num_gates = n_computation + n_rom + n_range.
        //  *
        //  * Its necessary to include the (n_rom + n_range) gates at this point because if we already have a
        //  * proving key, and we just return it without including these ROM and range list gates, the overall
        //  * circuit size would not be correct (resulting in the code crashing while performing FFT
        //  * operations).
        //  *
        //  * Therefore, we introduce a boolean flag `circuit_finalized` here. Once we add the rom and range gates,
        //  * our circuit is finalized, and we must not to execute these functions again.
        //  */
        if self.circuit_finalized {
            // Gates added after first call to finalize will not be processed since finalization is only performed once
            tracing::info!("WARNING: Redundant call to finalize_circuit(). Is this intentional?");
        } else {
            if ensure_nonzero {
                self.add_gates_to_ensure_all_polys_are_non_zero();
            }

            self.process_non_native_field_multiplications();
            self.process_rom_arrays();
            self.process_ram_arrays();
            self.process_range_lists();
            self.circuit_finalized = true;
        }
    }

    fn process_rom_arrays(&mut self) {
        for _ in self.rom_arrays.iter() {
            todo!("process rom array");
        }
    }

    fn process_ram_arrays(&mut self) {
        for _ in self.ram_arrays.iter() {
            todo!("process ram array");
        }
    }

    fn process_range_lists(&mut self) {
        for _ in self.range_lists.iter() {
            todo!("process range lists");
        }
    }

    fn process_non_native_field_multiplications(&mut self) {
        for c in self
            .cached_partial_non_native_field_multiplications
            .iter_mut()
        {
            for i in 0..5 {
                c.a[i] = self.real_variable_index[c.a[i] as usize];
                c.b[i] = self.real_variable_index[c.b[i] as usize];
            }
        }
        let mut dedup = CachedPartialNonNativeFieldMultiplication::deduplicate(
            &self.cached_partial_non_native_field_multiplications,
        );

        // iterate over the cached items and create constraints
        for input in dedup.iter() {
            let input_lo_0: BigUint = input.lo_0.into();
            let input_lo_0: u32 = input_lo_0.try_into().expect("Invalid index");

            self.blocks
                .aux
                .populate_wires(input.a[1], input.b[1], self.zero_idx, input_lo_0);
            self.apply_aux_selectors(AuxSelectors::NonNativeField1);
            self.num_gates += 1;

            self.blocks
                .aux
                .populate_wires(input.a[0], input.b[0], input.a[3], input.b[3]);
            self.apply_aux_selectors(AuxSelectors::NonNativeField2);
            self.num_gates += 1;

            let input_hi_0: BigUint = input.hi_0.into();
            let input_hi_0: u32 = input_hi_0.try_into().expect("Invalid index");

            self.blocks
                .aux
                .populate_wires(input.a[2], input.b[2], self.zero_idx, input_hi_0);
            self.apply_aux_selectors(AuxSelectors::NonNativeField2);
            self.num_gates += 1;

            let input_hi_1: BigUint = input.hi_1.into();
            let input_hi_1: u32 = input_hi_1.try_into().expect("Invalid index");

            self.blocks
                .aux
                .populate_wires(input.a[1], input.b[1], self.zero_idx, input_hi_1);
            self.apply_aux_selectors(AuxSelectors::None);
            self.num_gates += 1;
        }
        std::mem::swap(
            &mut self.cached_partial_non_native_field_multiplications,
            &mut dedup,
        );
    }

    pub fn compute_dyadic_size(&self) -> usize {
        // for the lookup argument the circuit size must be at least as large as the sum of all tables used
        let min_size_due_to_lookups = self.get_tables_size();

        // minimum size of execution trace due to everything else
        let min_size_of_execution_trace = self.public_inputs.len() + self.num_gates;

        // The number of gates is the maximum required by the lookup argument or everything else, plus an optional zero row
        // to allow for shifts.
        let num_zero_rows = 1;
        let total_num_gates =
            num_zero_rows + std::cmp::max(min_size_due_to_lookups, min_size_of_execution_trace);

        // Next power of 2 (dyadic circuit size)
        Self::get_circuit_subgroup_size(total_num_gates)
    }

    pub fn populate_public_inputs_block(&mut self) {
        tracing::info!("Populating public inputs block");

        // Update the public inputs block
        for idx in self.public_inputs.iter() {
            for (wire_idx, wire) in self.blocks.pub_inputs.wires.iter_mut().enumerate() {
                if wire_idx < 2 {
                    // first two wires get a copy of the public inputs
                    wire.push(*idx);
                } else {
                    // the remaining wires get zeros
                    wire.push(self.zero_idx);
                }
            }
            for selector in self.blocks.pub_inputs.selectors.iter_mut() {
                selector.push(P::ScalarField::zero());
            }
        }
    }
}
