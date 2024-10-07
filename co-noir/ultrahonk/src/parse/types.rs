use super::builder::{GenericUltraCircuitBuilder, UltraCircuitBuilder, UltraCircuitVariable};
use super::plookup::{BasicTableId, MultiTableId};
use crate::decider::polynomial::Polynomial;
use crate::types::ProvingKey;
use crate::Utils;
use ark_ec::pairing::Pairing;
use ark_ff::{One, PrimeField, Zero};
use itertools::izip;
use num_bigint::BigUint;
use std::array;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};

#[derive(Default, PartialEq, Eq)]
pub(crate) struct PolyTriple<F: PrimeField> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) q_m: F,
    pub(crate) q_l: F,
    pub(crate) q_r: F,
    pub(crate) q_o: F,
    pub(crate) q_c: F,
}

#[derive(Default, PartialEq, Eq)]
pub(crate) struct AddTriple<F: PrimeField> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) a_scaling: F,
    pub(crate) b_scaling: F,
    pub(crate) c_scaling: F,
    pub(crate) const_scaling: F,
}

#[derive(Default, PartialEq, Eq)]
pub(crate) struct AddQuad<F: PrimeField> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) d: u32,
    pub(crate) a_scaling: F,
    pub(crate) b_scaling: F,
    pub(crate) c_scaling: F,
    pub(crate) d_scaling: F,
    pub(crate) const_scaling: F,
}

#[derive(Default, PartialEq, Eq)]
pub(crate) struct MulQuad<F: PrimeField> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) d: u32,
    pub(crate) mul_scaling: F,
    pub(crate) a_scaling: F,
    pub(crate) b_scaling: F,
    pub(crate) c_scaling: F,
    pub(crate) d_scaling: F,
    pub(crate) const_scaling: F,
}

pub(crate) struct MemOp<F: PrimeField> {
    pub(crate) access_type: u8,
    pub(crate) index: PolyTriple<F>,
    pub(crate) value: PolyTriple<F>,
}

#[derive(PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum BlockType {
    ROM = 0,
    RAM = 1,
    CallData = 2,
    ReturnData = 3,
}

impl Default for BlockType {
    fn default() -> Self {
        Self::ROM
    }
}

#[derive(Default)]
pub(crate) struct BlockConstraint<F: PrimeField> {
    pub(crate) init: Vec<PolyTriple<F>>,
    pub(crate) trace: Vec<MemOp<F>>,
    pub(crate) type_: BlockType,
    pub(crate) calldata: u32,
}

#[derive(Default)]
pub(crate) struct AcirFormatOriginalOpcodeIndices {
    // pub(crate)logic_constraints: Vec<usize>,
    // pub(crate)range_constraints: Vec<usize>,
    // pub(crate)aes128_constraints: Vec<usize>,
    // pub(crate)sha256_constraints: Vec<usize>,
    // pub(crate)sha256_compression: Vec<usize>,
    // pub(crate)schnorr_constraints: Vec<usize>,
    // pub(crate)ecdsa_k1_constraints: Vec<usize>,
    // pub(crate)ecdsa_r1_constraints: Vec<usize>,
    // pub(crate)blake2s_constraints: Vec<usize>,
    // pub(crate)blake3_constraints: Vec<usize>,
    // pub(crate)keccak_constraints: Vec<usize>,
    // pub(crate)keccak_permutations: Vec<usize>,
    // pub(crate)pedersen_constraints: Vec<usize>,
    // pub(crate)pedersen_hash_constraints: Vec<usize>,
    // pub(crate)poseidon2_constraints: Vec<usize>,
    // pub(crate)multi_scalar_mul_constraints: Vec<usize>,
    // pub(crate)ec_add_constraints: Vec<usize>,
    // pub(crate)recursion_constraints: Vec<usize>,
    // pub(crate)honk_recursion_constraints: Vec<usize>,
    // pub(crate)avm_recursion_constraints: Vec<usize>,
    // pub(crate)ivc_recursion_constraints: Vec<usize>,
    // pub(crate)bigint_from_le_bytes_constraints: Vec<usize>,
    // pub(crate)bigint_to_le_bytes_constraints: Vec<usize>,
    // pub(crate)bigint_operations: Vec<usize>,
    pub(crate) assert_equalities: Vec<usize>,
    pub(crate) poly_triple_constraints: Vec<usize>,
    pub(crate) quad_constraints: Vec<usize>,
    // Multiple opcode indices per block:
    pub(crate) block_constraints: Vec<Vec<usize>>,
}

pub struct UltraTraceBlocks<T: Default> {
    pub(crate) pub_inputs: T,
    pub(crate) arithmetic: T,
    pub(crate) delta_range: T,
    pub(crate) elliptic: T,
    pub(crate) aux: T,
    pub(crate) lookup: T,
    pub(crate) poseidon2_external: T,
    pub(crate) poseidon2_internal: T,
}

impl<T: Default> UltraTraceBlocks<T> {
    pub fn get(&self) -> [&T; 8] {
        [
            &self.pub_inputs,
            &self.arithmetic,
            &self.delta_range,
            &self.elliptic,
            &self.aux,
            &self.lookup,
            &self.poseidon2_external,
            &self.poseidon2_internal,
        ]
    }

    pub fn get_pub_inputs(&self) -> &T {
        &self.pub_inputs
    }
}

pub const NUM_WIRES: usize = 4;
pub const NUM_SELECTORS: usize = 13;
pub type UltraTraceBlock<F> = ExecutionTraceBlock<F, NUM_WIRES, NUM_SELECTORS>;

pub struct ExecutionTraceBlock<F: PrimeField, const NUM_WIRES: usize, const NUM_SELECTORS: usize> {
    pub wires: [Vec<u32>; NUM_WIRES], // vectors of indices into a witness variables array
    pub selectors: [Vec<F>; NUM_SELECTORS],
    pub has_ram_rom: bool,      // does the block contain RAM/ROM gates
    pub is_pub_inputs: bool,    // is this the public inputs block
    pub(crate) fixed_size: u32, // Fixed size for use in structured trace
}

impl<F: PrimeField, const NUM_WIRES: usize, const NUM_SELECTORS: usize> Default
    for ExecutionTraceBlock<F, NUM_WIRES, NUM_SELECTORS>
{
    fn default() -> Self {
        Self {
            wires: array::from_fn(|_| Vec::new()),
            selectors: array::from_fn(|_| Vec::new()),
            has_ram_rom: false,
            is_pub_inputs: false,
            fixed_size: 0,
        }
    }
}

impl<F: PrimeField> Default for UltraTraceBlocks<UltraTraceBlock<F>> {
    fn default() -> Self {
        let mut res = Self {
            pub_inputs: Default::default(),
            arithmetic: Default::default(),
            delta_range: Default::default(),
            elliptic: Default::default(),
            aux: Default::default(),
            lookup: Default::default(),
            poseidon2_external: Default::default(),
            poseidon2_internal: Default::default(),
        };

        res.pub_inputs.is_pub_inputs = true;
        res.aux.has_ram_rom = true;
        res
    }
}

impl<F: PrimeField> UltraTraceBlock<F> {
    /// wire column 0
    const W_L: usize = 0;
    /// wire column 1
    const W_R: usize = 1;
    /// wire column 2
    const W_O: usize = 2;
    /// wire column 3
    const W_4: usize = 3;

    /// selector column 0
    const Q_M: usize = 0;
    /// selector column 1
    const Q_C: usize = 1;
    /// selector column 2
    const Q_1: usize = 2;
    /// selector column 3
    const Q_2: usize = 3;
    /// selector column 4
    const Q_3: usize = 4;
    /// selector column 5
    const Q_4: usize = 5;
    /// selector column 6
    const Q_ARITH: usize = 6;
    /// selector column 7
    const Q_DELTA_RANGE: usize = 7;
    /// selector column 8
    const Q_ELLIPTIC: usize = 8;
    /// selector column 9
    const Q_AUX: usize = 9;
    /// selector column 10
    const Q_LOOKUP_TYPE: usize = 10;
    /// selector column 11
    const Q_POSEIDON2_EXTERNAL: usize = 11;
    /// selector column 12
    const Q_POSEIDON2_INTERNAL: usize = 12;

    pub(crate) fn w_l(&mut self) -> &mut Vec<u32> {
        &mut self.wires[Self::W_L]
    }

    pub(crate) fn w_r(&mut self) -> &mut Vec<u32> {
        &mut self.wires[Self::W_R]
    }

    pub(crate) fn w_o(&mut self) -> &mut Vec<u32> {
        &mut self.wires[Self::W_O]
    }

    pub(crate) fn w_4(&mut self) -> &mut Vec<u32> {
        &mut self.wires[Self::W_4]
    }

    pub(crate) fn q_m(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_M]
    }

    pub(crate) fn q_c(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_C]
    }

    pub(crate) fn q_1(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_1]
    }

    pub(crate) fn q_2(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_2]
    }

    pub(crate) fn q_3(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_3]
    }

    pub(crate) fn q_4(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_4]
    }

    pub(crate) fn q_arith(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_ARITH]
    }

    pub(crate) fn q_delta_range(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_DELTA_RANGE]
    }

    pub(crate) fn q_elliptic(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_ELLIPTIC]
    }

    pub(crate) fn q_aux(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_AUX]
    }

    pub(crate) fn q_lookup_type(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_LOOKUP_TYPE]
    }

    pub(crate) fn q_poseidon2_external(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_POSEIDON2_EXTERNAL]
    }

    pub(crate) fn q_poseidon2_internal(&mut self) -> &mut Vec<F> {
        &mut self.selectors[Self::Q_POSEIDON2_INTERNAL]
    }

    pub(crate) fn populate_wires(&mut self, idx1: u32, idx2: u32, idx3: u32, idx4: u32) {
        self.w_l().push(idx1);
        self.w_r().push(idx2);
        self.w_o().push(idx3);
        self.w_4().push(idx4);
    }

    pub fn get_fixed_size(&self, is_structured: bool) -> u32 {
        if is_structured {
            self.fixed_size
        } else {
            self.len() as u32
        }
    }

    pub fn len(&self) -> usize {
        self.wires[Self::W_L].len()
    }
}

pub(crate) struct GateCounter {
    collect_gates_per_opcode: bool,
    prev_gate_count: usize,
}

impl GateCounter {
    pub(crate) fn new(collect_gates_per_opcode: bool) -> Self {
        Self {
            collect_gates_per_opcode,
            prev_gate_count: 0,
        }
    }

    pub(crate) fn compute_diff<P: Pairing, S: UltraCircuitVariable<P::ScalarField>>(
        &mut self,
        builder: &GenericUltraCircuitBuilder<P, S>,
    ) -> usize {
        if !self.collect_gates_per_opcode {
            return 0;
        }
        let new_gate_count = builder.get_num_gates();
        let diff = new_gate_count - self.prev_gate_count;
        self.prev_gate_count = new_gate_count;
        diff
    }

    pub(crate) fn track_diff<P: Pairing, S: UltraCircuitVariable<P::ScalarField>>(
        &mut self,
        builder: &GenericUltraCircuitBuilder<P, S>,
        gates_per_opcode: &mut [usize],
        opcode_index: usize,
    ) {
        if self.collect_gates_per_opcode {
            gates_per_opcode[opcode_index] = self.compute_diff(builder);
        }
    }
}

pub(crate) struct RecursionConstraint {
    // An aggregation state is represented by two G1 affine elements. Each G1 point has
    // two field element coordinates (x, y). Thus, four field elements
    key: Vec<u32>,
    proof: Vec<u32>,
    public_inputs: Vec<u32>,
    key_hash: u32,
    proof_type: u32,
}

impl RecursionConstraint {
    const NUM_AGGREGATION_ELEMENTS: usize = 4;
}

pub(crate) const AGGREGATION_OBJECT_SIZE: usize = 16;
pub(crate) type AggregationObjectIndices = [u32; AGGREGATION_OBJECT_SIZE];
pub(crate) type AggregationObjectPubInputIndices = [u32; AGGREGATION_OBJECT_SIZE];

pub(crate) struct RomTable<F: PrimeField> {
    raw_entries: Vec<FieldCT<F>>,
    entries: Vec<FieldCT<F>>,
    length: usize,
    rom_id: usize, // Builder identifier for this ROM table
    initialized: bool,
}

impl<F: PrimeField> RomTable<F> {
    pub(crate) fn new(table_entries: Vec<FieldCT<F>>) -> Self {
        let raw_entries = table_entries;
        let length = raw_entries.len();

        // do not initialize the table yet. The input entries might all be constant,
        // if this is the case we might not have a valid pointer to a Builder
        // We get around this, by initializing the table when `operator[]` is called
        // with a non-const field element.

        Self {
            raw_entries,
            entries: Vec::new(),
            length,
            rom_id: 0,
            initialized: false,
        }
    }

    pub(crate) fn index_field_ct<P: Pairing, S: UltraCircuitVariable<P::ScalarField>>(
        &mut self,
        index: &FieldCT<F>,
        builder: &mut GenericUltraCircuitBuilder<P, S>,
    ) -> FieldCT<F>
    where
        F: From<P::ScalarField>,
        P::ScalarField: From<F>,
    {
        if index.is_constant() {
            let val: BigUint = index.get_value(builder).into();
            let val: usize = val.try_into().expect("Invalid index");
            return self[val].to_owned();
        }
        self.initialize_table(builder);

        let val: BigUint = index.get_value(builder).into();
        assert!(val < BigUint::from(self.length));

        let witness_index = index.normalize(builder).get_witness_index();
        let output_idx = builder.read_rom_array(self.rom_id, witness_index);
        FieldCT::from_witness_index(output_idx)
    }

    fn initialize_table<P: Pairing, S: UltraCircuitVariable<P::ScalarField>>(
        &mut self,
        builder: &mut GenericUltraCircuitBuilder<P, S>,
    ) where
        F: From<P::ScalarField>,
        P::ScalarField: From<F>,
    {
        if self.initialized {
            return;
        }
        // populate table. Table entries must be normalized and cannot be constants
        for entry in self.raw_entries.iter() {
            if entry.is_constant() {
                let val = entry.get_value(builder);
                self.entries.push(FieldCT::from_witness_index(
                    builder.put_constant_variable(P::ScalarField::from(val)),
                ));
            } else {
                self.entries.push(entry.normalize(builder));
            }
        }
        self.rom_id = builder.create_rom_array(self.length);

        for i in 0..self.length {
            builder.set_rom_element(self.rom_id, i, self.entries[i].get_witness_index());
        }

        self.initialized = true;
    }
}

impl<F: PrimeField> Index<usize> for RomTable<F> {
    type Output = FieldCT<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

pub(crate) struct RamTable<F: PrimeField> {
    raw_entries: Vec<FieldCT<F>>,
    index_initialized: Vec<bool>,
    length: usize,
    ram_id: usize, // Builder identifier for this RAM table
    ram_table_generated_in_builder: bool,
    all_entries_written_to_with_constant_index: bool,
}

impl<F: PrimeField> RamTable<F> {
    pub(crate) fn new(table_entries: Vec<FieldCT<F>>) -> Self {
        let raw_entries = table_entries;
        let length = raw_entries.len();
        let index_initialized = vec![false; length];

        // do not initialize the table yet. The input entries might all be constant,
        // if this is the case we might not have a valid pointer to a Builder
        // We get around this, by initializing the table when `read` or `write` operator is called
        // with a non-const field element.

        Self {
            raw_entries,
            index_initialized,
            length,
            ram_id: 0,
            ram_table_generated_in_builder: false,
            all_entries_written_to_with_constant_index: false,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FieldCT<F: PrimeField> {
    pub(crate) additive_constant: F,
    pub(crate) multiplicative_constant: F,
    pub(crate) witness_index: u32,
}

impl<F: PrimeField> FieldCT<F> {
    const IS_CONSTANT: u32 = u32::MAX;

    pub(crate) fn from_field(value: F) -> Self {
        Self {
            additive_constant: value,
            multiplicative_constant: F::one(),
            witness_index: Self::IS_CONSTANT,
        }
    }

    pub(crate) fn from_witness_index(witness_index: u32) -> Self {
        Self {
            additive_constant: F::zero(),
            multiplicative_constant: F::one(),
            witness_index,
        }
    }

    // TACEO TODO this is just implemented for the plain backend
    pub(crate) fn from_witness<P: Pairing, S: UltraCircuitVariable<P::ScalarField>>(
        input: P::ScalarField,
        builder: &mut GenericUltraCircuitBuilder<P, S>,
    ) -> Self
    where
        F: From<P::ScalarField>,
    {
        let witness = WitnessCT::from_field(input, builder);
        Self::from_witness_ct(witness)
    }

    pub(crate) fn from_witness_ct(value: WitnessCT<F>) -> Self {
        Self {
            additive_constant: F::zero(),
            multiplicative_constant: F::one(),
            witness_index: value.witness_index,
        }
    }

    pub(crate) fn get_value<P: Pairing, S: UltraCircuitVariable<P::ScalarField>>(
        &self,
        builder: &GenericUltraCircuitBuilder<P, S>,
    ) -> F
    where
        F: From<P::ScalarField>,
    {
        if self.witness_index != Self::IS_CONSTANT {
            let variable = builder
                .get_variable(self.witness_index as usize)
                .public_into_field(); // TACEO TODO this is just implemented for the Plain backend
            self.multiplicative_constant * F::from(variable) + self.additive_constant
        } else {
            self.additive_constant.to_owned()
        }
    }

    pub(crate) fn get_witness_index(&self) -> u32 {
        self.witness_index
    }

    /**
     * @brief Constrain that this field is equal to the given field.
     *
     * @warning: After calling this method, both field values *will* be equal, regardless of whether the constraint
     * succeeds or fails. This can lead to confusion when debugging. If you want to log the inputs, do so before
     * calling this method.
     */
    pub(crate) fn assert_equal<P: Pairing, S: UltraCircuitVariable<P::ScalarField>>(
        &self,
        other: &Self,
        builder: &mut GenericUltraCircuitBuilder<P, S>,
    ) where
        F: From<P::ScalarField>,
        P::ScalarField: From<F>,
    {
        if self.is_constant() && other.is_constant() {
            assert_eq!(self.get_value(builder), other.get_value(builder));
        } else if self.is_constant() {
            let right = other.normalize(builder);
            let left = P::ScalarField::from(self.get_value(builder));
            builder.assert_equal_constant(right.witness_index as usize, left);
        } else if other.is_constant() {
            let left = self.normalize(builder);
            let right = P::ScalarField::from(other.get_value(builder));
            builder.assert_equal_constant(left.witness_index as usize, right);
        } else {
            let left = self.normalize(builder);
            let right = other.normalize(builder);
            builder.assert_equal(left.witness_index as usize, right.witness_index as usize);
        }
    }

    fn is_constant(&self) -> bool {
        self.witness_index == Self::IS_CONSTANT
    }

    fn normalize<P: Pairing, S: UltraCircuitVariable<P::ScalarField>>(
        &self,
        builder: &mut GenericUltraCircuitBuilder<P, S>,
    ) -> Self
    where
        F: From<P::ScalarField>,
        P::ScalarField: From<F>,
    {
        if self.is_constant()
            || ((self.multiplicative_constant == F::one()) && (self.additive_constant == F::zero()))
        {
            return self.to_owned();
        }

        // Value of this = this.v * this.mul + this.add; // where this.v = context->variables[this.witness_index]
        // Normalised result = result.v * 1 + 0;         // where result.v = this.v * this.mul + this.add
        // We need a new gate to enforce that the `result` was correctly calculated from `this`.

        let mut result = FieldCT::default();
        let value = F::from(
            builder
                .get_variable(self.witness_index as usize)
                .public_into_field(),
        ); // TACEO TODO this is just implemented for the Plain backend
        let out = self.multiplicative_constant * value + self.additive_constant;

        result.witness_index = builder.add_variable(S::from_public(P::ScalarField::from(out)));
        result.additive_constant = F::zero();
        result.multiplicative_constant = F::one();

        // Aim of new gate: this.v * this.mul + this.add == result.v
        // <=>                           this.v * [this.mul] +                  result.v * [ -1] + [this.add] == 0
        // <=> this.v * this.v * [ 0 ] + this.v * [this.mul] + this.v * [ 0 ] + result.v * [ -1] + [this.add] == 0
        // <=> this.v * this.v * [q_m] + this.v * [   q_l  ] + this.v * [q_r] + result.v * [q_o] + [   q_c  ] == 0

        builder.create_add_gate(&AddTriple {
            a: self.witness_index,
            b: self.witness_index,
            c: result.witness_index,
            a_scaling: P::ScalarField::from(self.multiplicative_constant),
            b_scaling: P::ScalarField::zero(),
            c_scaling: -P::ScalarField::one(),
            const_scaling: P::ScalarField::from(self.additive_constant),
        });
        result
    }
}

impl<F: PrimeField> From<F> for FieldCT<F> {
    fn from(value: F) -> Self {
        Self::from_field(value)
    }
}

impl<F: PrimeField> Default for FieldCT<F> {
    fn default() -> Self {
        Self {
            additive_constant: F::zero(),
            multiplicative_constant: F::one(),
            witness_index: Self::IS_CONSTANT,
        }
    }
}

// TACEO TODO this is just implemented for the Plain backend
pub(crate) struct WitnessCT<F: PrimeField> {
    pub(crate) witness: F,
    pub(crate) witness_index: u32,
}

impl<F: PrimeField> WitnessCT<F> {
    const IS_CONSTANT: u32 = FieldCT::<F>::IS_CONSTANT;

    pub(crate) fn from_field<P: Pairing, S: UltraCircuitVariable<P::ScalarField>>(
        value: P::ScalarField,
        builder: &mut GenericUltraCircuitBuilder<P, S>,
    ) -> Self
    where
        F: From<P::ScalarField>,
    {
        builder.add_variable(S::from_public(value));
        Self {
            witness: F::from(value),
            witness_index: Self::IS_CONSTANT,
        }
    }
}

#[derive(Default)]
pub(crate) struct RomRecord {
    pub(crate) index_witness: u32,
    pub(crate) value_column1_witness: u32,
    pub(crate) value_column2_witness: u32,
    pub(crate) index: u32,
    pub(crate) record_witness: u32,
    pub(crate) gate_index: usize,
}

#[derive(Default)]
pub(crate) struct RomTranscript {
    // Contains the value of each index of the array
    pub(crate) state: Vec<[u32; 2]>,

    // A vector of records, each of which contains:
    // + The constant witness with the index
    // + The value in the memory slot
    // + The actual index value
    pub(crate) records: Vec<RomRecord>,
}

pub(crate) enum AccessType {
    Read,
    Write,
}

pub(crate) struct RamRecord {
    pub(crate) index_witness: u32,
    pub(crate) timestamp_witness: u32,
    pub(crate) value_witness: u32,
    pub(crate) index: u32,
    pub(crate) access_type: AccessType,
    pub(crate) record_witness: u32,
    pub(crate) gate_index: usize,
}

impl Default for RamRecord {
    fn default() -> Self {
        Self {
            index_witness: 0,
            timestamp_witness: 0,
            value_witness: 0,
            index: 0,
            access_type: AccessType::Read,
            record_witness: 0,
            gate_index: 0,
        }
    }
}

#[derive(Default)]
pub(crate) struct RamTranscript {
    // Contains the value of each index of the array
    pub(crate) state: Vec<u32>,

    // A vector of records, each of which contains:
    // + The constant witness with the index
    // + The value in the memory slot
    // + The actual index value
    pub(crate) records: Vec<RamRecord>,

    // used for RAM records, to compute the timestamp when performing a read/write
    pub(crate) access_count: usize,
}

#[derive(PartialEq, Eq)]
pub(crate) enum AuxSelectors {
    None,
    LimbAccumulate1,
    LimbAccumulate2,
    NonNativeField1,
    NonNativeField2,
    NonNativeField3,
    RamConsistencyCheck,
    RomConsistencyCheck,
    RamTimestampCheck,
    RomRead,
    RamRead,
    RamWrite,
}

#[derive(Clone)]
pub(crate) struct LookupEntry<F: Clone> {
    pub(crate) key: [BigUint; 2],
    pub(crate) value: [F; 2],
}

impl<F: PrimeField> LookupEntry<F> {
    pub(crate) fn to_table_components(&self, use_two_key: bool) -> [F; 3] {
        [
            F::from(self.key[0].to_owned()),
            if use_two_key {
                F::from(self.key[1].to_owned())
            } else {
                self.value[0]
            },
            if use_two_key {
                self.value[0]
            } else {
                self.value[1]
            },
        ]
    }
}

pub(crate) struct PlookupBasicTable<F: PrimeField> {
    pub(crate) id: BasicTableId,
    pub(crate) table_index: usize,
    pub(crate) use_twin_keys: bool,
    pub(crate) column_1_step_size: F,
    pub(crate) column_2_step_size: F,
    pub(crate) column_3_step_size: F,
    pub(crate) column_1: Vec<F>,
    pub(crate) column_2: Vec<F>,
    pub(crate) column_3: Vec<F>,
    pub(crate) lookup_gates: Vec<LookupEntry<F>>,
    pub(crate) index_map: LookupHashMap<F>,
    pub(crate) get_values_from_key: fn([u64; 2]) -> [F; 2],
}

impl<F: PrimeField> Default for PlookupBasicTable<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField> PlookupBasicTable<F> {
    fn new() -> Self {
        Self {
            id: BasicTableId::HonkDummyBasic1,
            table_index: 0,
            use_twin_keys: false,
            column_1_step_size: F::zero(),
            column_2_step_size: F::zero(),
            column_3_step_size: F::zero(),
            column_1: Vec::new(),
            column_2: Vec::new(),
            column_3: Vec::new(),
            lookup_gates: Vec::new(),
            index_map: LookupHashMap::default(),
            get_values_from_key: BasicTableId::get_value_from_key::<
                F,
                { BasicTableId::HonkDummyBasic1 as u64 },
            >,
        }
    }
}

impl<F: PrimeField> PlookupBasicTable<F> {
    pub(crate) fn len(&self) -> usize {
        assert_eq!(self.column_1.len(), self.column_2.len());
        assert_eq!(self.column_1.len(), self.column_3.len());
        self.column_1.len()
    }

    fn generate_honk_dummy_table<const ID: u64>(
        id: BasicTableId,
        table_index: usize,
    ) -> PlookupBasicTable<F> {
        // We do the assertion, since this function is templated, but the general API for these functions contains the id,
        // too. This helps us ensure that the correct instantion is used for a particular BasicTableId
        assert_eq!(ID, usize::from(id.to_owned()) as u64);
        let base = 1 << 1; // Probably has to be a power of 2
        let mut table = PlookupBasicTable::new();
        table.id = id;
        table.table_index = table_index;
        table.use_twin_keys = true;
        for i in 0..base {
            for j in 0..base {
                table.column_1.push(F::from(i));
                table.column_2.push(F::from(j));
                table.column_3.push(F::from(i * 3 + j * 4 + ID * 0x1337));
            }
        }

        table.get_values_from_key = BasicTableId::get_value_from_key::<F, ID>;
        let base = F::from(base);
        table.column_1_step_size = base;
        table.column_2_step_size = base;
        table.column_3_step_size = base;

        table
    }

    pub(crate) fn create_basic_table(id: BasicTableId, index: usize) -> Self {
        // TACEO TODO this is a dummy implementation
        assert!(id == BasicTableId::HonkDummyBasic1 || id == BasicTableId::HonkDummyBasic2);

        match id {
            BasicTableId::HonkDummyBasic1 => Self::generate_honk_dummy_table::<
                { BasicTableId::HonkDummyBasic1 as u64 },
            >(id, index),
            BasicTableId::HonkDummyBasic2 => Self::generate_honk_dummy_table::<
                { BasicTableId::HonkDummyBasic2 as u64 },
            >(id, index),
            _ => {
                todo!()
            }
        }
    }

    pub(crate) fn initialize_index_map(&mut self) {
        for (i, (c1, c2, c3)) in izip!(
            self.column_1.iter().cloned(),
            self.column_2.iter().cloned(),
            self.column_3.iter().cloned()
        )
        .enumerate()
        {
            self.index_map.index_map.insert([c1, c2, c3], i);
        }
    }
}

#[derive(Default)]
pub(crate) struct LookupHashMap<F: PrimeField> {
    pub(crate) index_map: HashMap<[F; 3], usize>,
}

impl<F: PrimeField> Index<[F; 3]> for LookupHashMap<F> {
    type Output = usize;

    fn index(&self, index: [F; 3]) -> &Self::Output {
        self.index_map.index(&index)
    }
}

pub(crate) struct PlookupMultiTable<F: PrimeField> {
    pub(crate) column_1_coefficients: Vec<F>,
    pub(crate) column_2_coefficients: Vec<F>,
    pub(crate) column_3_coefficients: Vec<F>,
    pub(crate) id: MultiTableId,
    pub(crate) basic_table_ids: Vec<BasicTableId>,
    pub(crate) slice_sizes: Vec<u64>,
    pub(crate) column_1_step_sizes: Vec<F>,
    pub(crate) column_2_step_sizes: Vec<F>,
    pub(crate) column_3_step_sizes: Vec<F>,
    pub(crate) get_table_values: Vec<fn([u64; 2]) -> [F; 2]>,
}

impl<F: PrimeField> Default for PlookupMultiTable<F> {
    fn default() -> Self {
        Self {
            column_1_coefficients: Vec::new(),
            column_2_coefficients: Vec::new(),
            column_3_coefficients: Vec::new(),
            id: MultiTableId::HonkDummyMulti,
            basic_table_ids: Vec::new(),
            slice_sizes: Vec::new(),
            column_1_step_sizes: Vec::new(),
            column_2_step_sizes: Vec::new(),
            column_3_step_sizes: Vec::new(),
            get_table_values: Vec::new(),
        }
    }
}

impl<F: PrimeField> PlookupMultiTable<F> {
    pub(crate) fn new(
        col1_repeated_coeff: F,
        col2_repeated_coeff: F,
        col3_repeated_coeff: F,
        num_lookups: usize,
    ) -> Self {
        let mut column_1_coefficients = Vec::with_capacity(num_lookups + 1);
        let mut column_2_coefficients = Vec::with_capacity(num_lookups + 1);
        let mut column_3_coefficients = Vec::with_capacity(num_lookups + 1);

        column_1_coefficients.push(F::one());
        column_2_coefficients.push(F::one());
        column_3_coefficients.push(F::one());

        for _ in 0..num_lookups {
            column_1_coefficients.push(col1_repeated_coeff * column_1_coefficients.last().unwrap());
            column_2_coefficients.push(col2_repeated_coeff * column_2_coefficients.last().unwrap());
            column_3_coefficients.push(col3_repeated_coeff * column_3_coefficients.last().unwrap());
        }

        let mut res = Self {
            column_1_coefficients,
            column_2_coefficients,
            column_3_coefficients,
            ..Default::default()
        };
        res.init_step_sizes();
        res
    }

    fn init_step_sizes(&mut self) {
        let num_lookups = self.column_1_coefficients.len();
        self.column_1_step_sizes.push(F::one());
        self.column_2_step_sizes.push(F::one());
        self.column_3_step_sizes.push(F::one());

        let mut coefficient_inverses = self.column_1_coefficients.clone();
        coefficient_inverses.extend(&self.column_2_coefficients);
        coefficient_inverses.extend(&self.column_3_coefficients);

        Utils::batch_invert(&mut coefficient_inverses);

        for i in 1..num_lookups {
            self.column_1_step_sizes
                .push(self.column_1_coefficients[i] * coefficient_inverses[i - 1]);
            self.column_2_step_sizes
                .push(self.column_2_coefficients[i] * coefficient_inverses[num_lookups + i - 1]);
            self.column_3_step_sizes.push(
                self.column_3_coefficients[i] * coefficient_inverses[2 * num_lookups + i - 1],
            );
        }
    }
}

#[derive(Default)]
pub(crate) struct ReadData<F: Clone> {
    pub(crate) lookup_entries: Vec<LookupEntry<F>>,
    pub(crate) columns: [Vec<F>; 3],
}

impl<F: Clone> Index<ColumnIdx> for ReadData<F> {
    type Output = Vec<F>;

    fn index(&self, index: ColumnIdx) -> &Self::Output {
        self.columns.index(index as usize)
    }
}

impl<F: Clone> IndexMut<ColumnIdx> for ReadData<F> {
    fn index_mut(&mut self, index: ColumnIdx) -> &mut Self::Output {
        self.columns.index_mut(index as usize)
    }
}

pub(crate) enum ColumnIdx {
    C1,
    C2,
    C3,
}

pub(crate) struct RangeList {
    pub(crate) target_range: u64,
    pub(crate) range_tag: u32,
    pub(crate) tau_tag: u32,
    pub(crate) variable_indices: Vec<u32>,
}

#[derive(Clone)]
pub(crate) struct CachedPartialNonNativeFieldMultiplication<F: PrimeField> {
    pub(crate) a: [u32; 5],
    pub(crate) b: [u32; 5],
    pub(crate) lo_0: F,
    pub(crate) hi_0: F,
    pub(crate) hi_1: F,
}

impl<F: PrimeField> CachedPartialNonNativeFieldMultiplication<F> {
    fn equal(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b
    }

    fn less_than(&self, other: &Self) -> bool {
        if self.a < other.a {
            return true;
        }
        if other.a < self.a {
            return false;
        }
        if self.b < other.b {
            return true;
        }
        other.b < self.b
    }

    pub(crate) fn deduplicate(inp: &[Self]) -> Vec<Self> {
        let mut hash_set = HashSet::new();
        let mut unique_vec = Vec::new();

        for element in inp.iter() {
            if hash_set.insert(element.clone()) {
                unique_vec.push(element.clone());
            }
        }
        unique_vec
    }
}

impl<F: PrimeField> PartialOrd for CachedPartialNonNativeFieldMultiplication<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: PrimeField> Ord for CachedPartialNonNativeFieldMultiplication<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.eq(other) {
            Ordering::Equal
        } else if self.less_than(other) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

impl<F: PrimeField> PartialEq for CachedPartialNonNativeFieldMultiplication<F> {
    fn eq(&self, other: &Self) -> bool {
        self.equal(other)
    }
}

impl<F: PrimeField> Eq for CachedPartialNonNativeFieldMultiplication<F> {}

impl<F: PrimeField> Hash for CachedPartialNonNativeFieldMultiplication<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.a.hash(state);
        self.b.hash(state);
    }
}

#[derive(Clone)]
pub struct CycleNode {
    pub wire_index: u32,
    pub gate_index: u32,
}
pub type CyclicPermutation = Vec<CycleNode>;

pub(crate) struct TraceData<'a, P: Pairing> {
    pub(crate) wires: &'a mut [Polynomial<P::ScalarField>; NUM_WIRES],
    pub(crate) selectors: &'a mut [Polynomial<P::ScalarField>; NUM_SELECTORS],
    pub(crate) copy_cycles: Vec<CyclicPermutation>,
    pub(crate) ram_rom_offset: u32,
    pub(crate) pub_inputs_offset: u32,
}

impl<'a, P: Pairing> TraceData<'a, P> {
    pub(crate) fn new(
        builder: &UltraCircuitBuilder<P>,
        proving_key: &'a mut ProvingKey<P>,
    ) -> Self {
        let copy_cycles = vec![vec![]; builder.variables.len()];

        Self {
            wires: proving_key
                .polynomials
                .witness
                .get_wires_mut()
                .try_into()
                .unwrap(),
            selectors: proving_key
                .polynomials
                .precomputed
                .get_selectors_mut()
                .try_into()
                .unwrap(),
            copy_cycles,
            ram_rom_offset: 0,
            pub_inputs_offset: 0,
        }
    }

    pub(crate) fn construct_trace_data(
        &mut self,
        builder: &mut UltraCircuitBuilder<P>,
        is_structured: bool,
    ) {
        tracing::info!("Construct trace data");
        // Complete the public inputs execution trace block from builder.public_inputs
        builder.populate_public_inputs_block();

        let mut offset = 1; // Offset at which to place each block in the trace polynomials
                            // For each block in the trace, populate wire polys, copy cycles and selector polys

        for block in builder.blocks.get() {
            let block_size = block.len();

            // Update wire polynomials and copy cycles
            // NB: The order of row/column loops is arbitrary but needs to be row/column to match old copy_cycle code

            for block_row_idx in 0..block_size {
                for wire_idx in 0..NUM_WIRES {
                    let var_idx = block.wires[wire_idx][block_row_idx] as usize; // an index into the variables array
                    let real_var_idx = builder.real_variable_index[var_idx] as usize;
                    let trace_row_idx = block_row_idx + offset;
                    // Insert the real witness values from this block into the wire polys at the correct offset
                    self.wires[wire_idx][trace_row_idx] = builder.get_variable(var_idx);
                    // Add the address of the witness value to its corresponding copy cycle
                    self.copy_cycles[real_var_idx].push(CycleNode {
                        wire_index: wire_idx as u32,
                        gate_index: trace_row_idx as u32,
                    });
                }
            }

            // Insert the selector values for this block into the selector polynomials at the correct offset
            // AZTEC TODO(https://github.com/AztecProtocol/barretenberg/issues/398): implicit arithmetization/flavor consistency
            for (selector_poly, selector) in self.selectors.iter_mut().zip(block.selectors.iter()) {
                debug_assert_eq!(selector.len(), block_size);

                for (src, des) in selector.iter().zip(selector_poly.iter_mut().skip(offset)) {
                    *des = *src;
                }
            }

            // Store the offset of the block containing RAM/ROM read/write gates for use in updating memory records
            if block.has_ram_rom {
                self.ram_rom_offset = offset as u32;
            }
            // Store offset of public inputs block for use in the pub(crate)input mechanism of the permutation argument
            if block.is_pub_inputs {
                self.pub_inputs_offset = offset as u32;
            }

            // If the trace is structured, we populate the data from the next block at a fixed block size offset
            // otherwise, the next block starts immediately following the previous one
            offset += block.get_fixed_size(is_structured) as usize;
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PermutationSubgroupElement {
    pub(crate) row_index: u32,
    pub(crate) column_index: u32,
    pub(crate) is_public_input: bool,
    pub(crate) is_tag: bool,
}

impl PermutationSubgroupElement {
    fn new(row_index: u32, column_index: u32) -> Self {
        Self {
            row_index,
            column_index,
            is_public_input: false,
            is_tag: false,
        }
    }
}

pub(crate) type Mapping = [Vec<PermutationSubgroupElement>; NUM_WIRES];
pub(crate) struct PermutationMapping {
    pub(crate) sigmas: Mapping,
    pub(crate) ids: Mapping,
}

impl PermutationMapping {
    pub(crate) fn new(circuit_size: usize) -> Self {
        let mut sigmas = array::from_fn(|_| Vec::with_capacity(circuit_size));
        let mut ids = array::from_fn(|_| Vec::with_capacity(circuit_size));

        for col_idx in 0..NUM_WIRES {
            for row_idx in 0..circuit_size {
                let perm_el = PermutationSubgroupElement::new(row_idx as u32, col_idx as u32);
                sigmas[col_idx].push(perm_el.to_owned());
                ids[col_idx].push(perm_el);
            }
        }

        Self { sigmas, ids }
    }
}
