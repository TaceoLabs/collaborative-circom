use crate::{decider::polynomial::Polynomial, prover::HonkProofResult, Serialize};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;

pub struct ProvingKey<P: Pairing> {
    pub(crate) crs: ProverCrs<P>,
    pub(crate) circuit_size: u32,
    pub(crate) public_inputs: Vec<P::ScalarField>,
    pub(crate) num_public_inputs: u32,
    pub(crate) pub_inputs_offset: u32,
    pub(crate) polynomials: Polynomials<P::ScalarField>,
    pub(crate) memory_read_records: Vec<u32>,
    pub(crate) memory_write_records: Vec<u32>,
}

pub struct VerifyingKey<P: Pairing> {
    pub crs: P::G2Affine,
    pub circuit_size: u32,
    pub num_public_inputs: u32,
    pub pub_inputs_offset: u32,
    pub commitments: PrecomputedEntities<P::G1Affine>,
}

// This is what we get from the proving key, we shift at a later point
#[derive(Default)]
pub(crate) struct Polynomials<F: PrimeField> {
    pub(crate) witness: ProverWitnessEntities<Polynomial<F>>,
    pub(crate) precomputed: PrecomputedEntities<Polynomial<F>>,
}

impl<F: PrimeField> Polynomials<F> {
    pub(crate) fn new(circuit_size: usize) -> Self {
        let mut polynomials = Self::default();
        // Shifting is done at a later point
        polynomials
            .iter_mut()
            .for_each(|el| el.resize(circuit_size, Default::default()));

        polynomials
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut Polynomial<F>> {
        self.witness.iter_mut().chain(self.precomputed.iter_mut())
    }
}

pub struct Crs<P: Pairing> {
    pub monomials: Vec<P::G1Affine>,
    pub g2_x: P::G2Affine,
}

pub struct ProverCrs<P: Pairing> {
    pub monomials: Vec<P::G1Affine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HonkProof<F: PrimeField> {
    proof: Vec<F>,
}

impl<F: PrimeField> HonkProof<F> {
    pub(crate) fn new(proof: Vec<F>) -> Self {
        Self { proof }
    }

    pub(crate) fn inner(self) -> Vec<F> {
        self.proof
    }

    pub fn to_buffer(&self) -> Vec<u8> {
        Serialize::to_buffer(&self.proof, true)
    }

    pub fn from_buffer(buf: &[u8]) -> HonkProofResult<Self> {
        let res = Serialize::from_buffer(buf, true)?;
        Ok(Self::new(res))
    }
}

pub(crate) const NUM_ALL_ENTITIES: usize = WITNESS_ENTITIES_SIZE
    + PRECOMPUTED_ENTITIES_SIZE
    + SHIFTED_TABLE_ENTITIES_SIZE
    + SHIFTED_WITNESS_ENTITIES_SIZE;
#[derive(Default)]
pub(crate) struct AllEntities<T: Default> {
    pub(crate) witness: WitnessEntities<T>,
    pub(crate) precomputed: PrecomputedEntities<T>,
    pub(crate) shifted_witness: ShiftedWitnessEntities<T>,
    pub(crate) shifted_tables: ShiftedTableEntities<T>,
}

impl<T: Default> AllEntities<T> {
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.precomputed
            .iter()
            .chain(self.witness.iter())
            .chain(self.shifted_tables.iter())
            .chain(self.shifted_witness.iter())
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.precomputed
            .iter_mut()
            .chain(self.witness.iter_mut())
            .chain(self.shifted_tables.iter_mut())
            .chain(self.shifted_witness.iter_mut())
    }
}

impl<T: Default + Clone> AllEntities<Vec<T>> {
    pub(crate) fn new(circuit_size: usize) -> Self {
        let mut polynomials = Self::default();
        // Shifting is done at a later point
        polynomials
            .iter_mut()
            .for_each(|el| el.resize(circuit_size, Default::default()));

        polynomials
    }
}

const WITNESS_ENTITIES_SIZE: usize = 8;
#[derive(Default)]
pub(crate) struct WitnessEntities<T: Default> {
    pub(crate) elements: [T; WITNESS_ENTITIES_SIZE],
}

const PROVER_WITNESS_ENTITIES_SIZE: usize = 6;
#[derive(Default)]
pub(crate) struct ProverWitnessEntities<T: Default> {
    pub(crate) elements: [T; PROVER_WITNESS_ENTITIES_SIZE],
}

const SHIFTED_WITNESS_ENTITIES_SIZE: usize = 5;
#[derive(Default)]
pub struct ShiftedWitnessEntities<T: Default> {
    pub(crate) elements: [T; SHIFTED_WITNESS_ENTITIES_SIZE],
}

const SHIFTED_TABLE_ENTITIES_SIZE: usize = 4;
#[derive(Default)]
pub struct ShiftedTableEntities<T: Default> {
    pub(crate) elements: [T; SHIFTED_TABLE_ENTITIES_SIZE],
}

pub(crate) const PRECOMPUTED_ENTITIES_SIZE: usize = 27;
#[derive(Default)]
pub struct PrecomputedEntities<T: Default> {
    pub(crate) elements: [T; PRECOMPUTED_ENTITIES_SIZE],
}

impl<T: Default> IntoIterator for PrecomputedEntities<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, PRECOMPUTED_ENTITIES_SIZE>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<T: Default> IntoIterator for WitnessEntities<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, WITNESS_ENTITIES_SIZE>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<T: Default> IntoIterator for ProverWitnessEntities<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, PROVER_WITNESS_ENTITIES_SIZE>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<T: Default> IntoIterator for ShiftedTableEntities<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, SHIFTED_TABLE_ENTITIES_SIZE>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<T: Default> IntoIterator for ShiftedWitnessEntities<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, SHIFTED_WITNESS_ENTITIES_SIZE>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<T: Default> IntoIterator for AllEntities<T> {
    type Item = T;
    type IntoIter = std::iter::Chain<
        std::iter::Chain<
            std::iter::Chain<
                std::array::IntoIter<T, PRECOMPUTED_ENTITIES_SIZE>,
                std::array::IntoIter<T, WITNESS_ENTITIES_SIZE>,
            >,
            std::array::IntoIter<T, SHIFTED_TABLE_ENTITIES_SIZE>,
        >,
        std::array::IntoIter<T, SHIFTED_WITNESS_ENTITIES_SIZE>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.precomputed
            .into_iter()
            .chain(self.witness)
            .chain(self.shifted_tables)
            .chain(self.shifted_witness)
    }
}

impl<T: Default> ProverWitnessEntities<T> {
    /// column 0
    const W_L: usize = 0;
    /// column 1
    const W_R: usize = 1;
    /// column 2
    const W_O: usize = 2;
    /// column 3 (modified by prover)
    const W_4: usize = 3;
    /// column 6
    const LOOKUP_READ_COUNTS: usize = 4;
    /// column 7
    const LOOKUP_READ_TAGS: usize = 5;

    // const Z_PERM: usize = 4; // column 4 (computed by prover)
    // const LOOKUP_INVERSES: usize = 5; // column 5 (computed by prover);

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.elements.iter_mut()
    }

    pub(crate) fn into_wires(self) -> impl Iterator<Item = T> {
        self.elements
            .into_iter()
            // .skip(Self::W_L)
            .take(Self::W_4 + 1 - Self::W_L)
    }

    pub(crate) fn get_wires_mut(&mut self) -> &mut [T] {
        &mut self.elements[Self::W_L..=Self::W_4]
    }

    pub(crate) fn w_l(&self) -> &T {
        &self.elements[Self::W_L]
    }

    pub(crate) fn w_r(&self) -> &T {
        &self.elements[Self::W_R]
    }

    pub(crate) fn w_o(&self) -> &T {
        &self.elements[Self::W_O]
    }

    pub(crate) fn w_4(&self) -> &T {
        &self.elements[Self::W_4]
    }

    pub(crate) fn lookup_read_counts(&self) -> &T {
        &self.elements[Self::LOOKUP_READ_COUNTS]
    }

    pub(crate) fn lookup_read_tags(&self) -> &T {
        &self.elements[Self::LOOKUP_READ_TAGS]
    }

    pub(crate) fn lookup_read_counts_and_tags_mut(&mut self) -> &mut [T] {
        &mut self.elements[Self::LOOKUP_READ_COUNTS..Self::LOOKUP_READ_TAGS + 1]
    }
}

impl<T: Default> WitnessEntities<T> {
    /// column 0
    const W_L: usize = 0;
    /// column 1
    const W_R: usize = 1;
    /// column 2
    const W_O: usize = 2;
    /// column 3 (computed by prover)
    const W_4: usize = 3;
    /// column 4 (computed by prover)
    const Z_PERM: usize = 4;
    /// column 5 (computed by prover);
    pub(crate) const LOOKUP_INVERSES: usize = 5;
    /// column 6
    pub(crate) const LOOKUP_READ_COUNTS: usize = 6;
    /// column 7
    pub(crate) const LOOKUP_READ_TAGS: usize = 7;

    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.elements.iter_mut()
    }

    pub(crate) fn to_be_shifted(&self) -> &[T] {
        &self.elements[Self::W_L..=Self::Z_PERM]
    }

    pub(crate) fn to_be_shifted_mut(&mut self) -> &mut [T] {
        &mut self.elements[Self::W_L..=Self::Z_PERM]
    }

    pub(crate) fn w_l(&self) -> &T {
        &self.elements[Self::W_L]
    }

    pub(crate) fn w_r(&self) -> &T {
        &self.elements[Self::W_R]
    }

    pub(crate) fn w_o(&self) -> &T {
        &self.elements[Self::W_O]
    }

    pub(crate) fn w_4(&self) -> &T {
        &self.elements[Self::W_4]
    }

    pub(crate) fn z_perm(&self) -> &T {
        &self.elements[Self::Z_PERM]
    }

    pub(crate) fn lookup_inverses(&self) -> &T {
        &self.elements[Self::LOOKUP_INVERSES]
    }

    pub(crate) fn lookup_read_counts(&self) -> &T {
        &self.elements[Self::LOOKUP_READ_COUNTS]
    }

    pub(crate) fn lookup_read_tags(&self) -> &T {
        &self.elements[Self::LOOKUP_READ_TAGS]
    }

    pub(crate) fn w_l_mut(&mut self) -> &mut T {
        &mut self.elements[Self::W_L]
    }

    pub(crate) fn w_r_mut(&mut self) -> &mut T {
        &mut self.elements[Self::W_R]
    }

    pub(crate) fn w_o_mut(&mut self) -> &mut T {
        &mut self.elements[Self::W_O]
    }

    pub(crate) fn w_4_mut(&mut self) -> &mut T {
        &mut self.elements[Self::W_4]
    }

    pub(crate) fn z_perm_mut(&mut self) -> &mut T {
        &mut self.elements[Self::Z_PERM]
    }

    pub(crate) fn lookup_inverses_mut(&mut self) -> &mut T {
        &mut self.elements[Self::LOOKUP_INVERSES]
    }

    pub(crate) fn lookup_read_counts_mut(&mut self) -> &mut T {
        &mut self.elements[Self::LOOKUP_READ_COUNTS]
    }

    pub(crate) fn lookup_read_tags_mut(&mut self) -> &mut T {
        &mut self.elements[Self::LOOKUP_READ_TAGS]
    }
}

impl<T: Default> ShiftedWitnessEntities<T> {
    /// column 0
    const W_L: usize = 0;
    /// column 1
    const W_R: usize = 1;
    /// column 2
    const W_O: usize = 2;
    /// column 3
    const W_4: usize = 3;
    /// column 4
    const Z_PERM: usize = 4;

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.elements.iter_mut()
    }

    pub fn w_l(&self) -> &T {
        &self.elements[Self::W_L]
    }

    pub fn w_r(&self) -> &T {
        &self.elements[Self::W_R]
    }

    pub fn w_o(&self) -> &T {
        &self.elements[Self::W_O]
    }

    pub fn w_4(&self) -> &T {
        &self.elements[Self::W_4]
    }

    pub fn z_perm(&self) -> &T {
        &self.elements[Self::Z_PERM]
    }
}

#[allow(unused)]
impl<T: Default> ShiftedTableEntities<T> {
    /// column 0
    const TABLE_1: usize = 0;
    /// column 1
    const TABLE_2: usize = 1;
    /// column 2
    const TABLE_3: usize = 2;
    /// column 3
    const TABLE_4: usize = 3;

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.elements.iter_mut()
    }

    pub(crate) fn table_1(&self) -> &T {
        &self.elements[Self::TABLE_1]
    }

    pub(crate) fn table_2(&self) -> &T {
        &self.elements[Self::TABLE_2]
    }

    pub(crate) fn table_3(&self) -> &T {
        &self.elements[Self::TABLE_3]
    }

    pub(crate) fn table_4(&self) -> &T {
        &self.elements[Self::TABLE_4]
    }
}

impl<T: Default> PrecomputedEntities<T> {
    /// column 0
    const Q_M: usize = 0;
    /// column 1
    const Q_C: usize = 1;
    /// column 2
    const Q_L: usize = 2;
    /// column 3
    const Q_R: usize = 3;
    /// column 4
    const Q_O: usize = 4;
    /// column 5
    const Q_4: usize = 5;
    /// column 6
    const Q_ARITH: usize = 6;
    /// column 7
    const Q_DELTA_RANGE: usize = 7;
    /// column 8
    const Q_ELLIPTIC: usize = 8;
    /// column 9
    const Q_AUX: usize = 9;
    /// column 10
    const Q_LOOKUP: usize = 10;
    /// column 11
    const Q_POSEIDON2_EXTERNAL: usize = 11;
    /// column 12
    const Q_POSEIDON2_INTERNAL: usize = 12;
    /// column 13
    const SIGMA_1: usize = 13;
    /// column 14
    const SIGMA_2: usize = 14;
    /// column 15
    const SIGMA_3: usize = 15;
    /// column 16
    const SIGMA_4: usize = 16;
    /// column 17
    const ID_1: usize = 17;
    /// column 18
    const ID_2: usize = 18;
    /// column 19
    const ID_3: usize = 19;
    /// column 20
    const ID_4: usize = 20;
    /// column 21
    const TABLE_1: usize = 21;
    /// column 22
    const TABLE_2: usize = 22;
    /// column 23
    const TABLE_3: usize = 23;
    /// column 24
    const TABLE_4: usize = 24;
    /// column 25
    const LAGRANGE_FIRST: usize = 25;
    /// column 26
    const LAGRANGE_LAST: usize = 26;

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.elements.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.elements.iter_mut()
    }

    pub fn get_table_polynomials(&self) -> &[T] {
        &self.elements[Self::TABLE_1..=Self::TABLE_4]
    }

    pub fn get_selectors_mut(&mut self) -> &mut [T] {
        &mut self.elements[Self::Q_M..=Self::Q_POSEIDON2_INTERNAL]
    }

    pub fn get_sigmas_mut(&mut self) -> &mut [T] {
        &mut self.elements[Self::SIGMA_1..=Self::SIGMA_4]
    }

    pub fn get_ids_mut(&mut self) -> &mut [T] {
        &mut self.elements[Self::ID_1..=Self::ID_4]
    }

    pub fn get_table_polynomials_mut(&mut self) -> &mut [T] {
        &mut self.elements[Self::TABLE_1..=Self::TABLE_4]
    }

    pub fn q_m(&self) -> &T {
        &self.elements[Self::Q_M]
    }

    pub fn q_c(&self) -> &T {
        &self.elements[Self::Q_C]
    }

    pub fn q_l(&self) -> &T {
        &self.elements[Self::Q_L]
    }

    pub fn q_r(&self) -> &T {
        &self.elements[Self::Q_R]
    }

    pub fn q_o(&self) -> &T {
        &self.elements[Self::Q_O]
    }

    pub fn q_4(&self) -> &T {
        &self.elements[Self::Q_4]
    }

    pub fn q_arith(&self) -> &T {
        &self.elements[Self::Q_ARITH]
    }

    pub fn q_delta_range(&self) -> &T {
        &self.elements[Self::Q_DELTA_RANGE]
    }

    pub fn q_elliptic(&self) -> &T {
        &self.elements[Self::Q_ELLIPTIC]
    }

    pub fn q_aux(&self) -> &T {
        &self.elements[Self::Q_AUX]
    }

    pub fn q_lookup(&self) -> &T {
        &self.elements[Self::Q_LOOKUP]
    }

    pub fn q_poseidon2_external(&self) -> &T {
        &self.elements[Self::Q_POSEIDON2_EXTERNAL]
    }

    pub fn q_poseidon2_internal(&self) -> &T {
        &self.elements[Self::Q_POSEIDON2_INTERNAL]
    }

    pub fn sigma_1(&self) -> &T {
        &self.elements[Self::SIGMA_1]
    }

    pub fn sigma_2(&self) -> &T {
        &self.elements[Self::SIGMA_2]
    }

    pub fn sigma_3(&self) -> &T {
        &self.elements[Self::SIGMA_3]
    }

    pub fn sigma_4(&self) -> &T {
        &self.elements[Self::SIGMA_4]
    }

    pub fn id_1(&self) -> &T {
        &self.elements[Self::ID_1]
    }

    pub fn id_2(&self) -> &T {
        &self.elements[Self::ID_2]
    }

    pub fn id_3(&self) -> &T {
        &self.elements[Self::ID_3]
    }

    pub fn id_4(&self) -> &T {
        &self.elements[Self::ID_4]
    }

    pub fn table_1(&self) -> &T {
        &self.elements[Self::TABLE_1]
    }

    pub fn table_2(&self) -> &T {
        &self.elements[Self::TABLE_2]
    }

    pub fn table_3(&self) -> &T {
        &self.elements[Self::TABLE_3]
    }

    pub fn table_4(&self) -> &T {
        &self.elements[Self::TABLE_4]
    }

    pub fn lagrange_first(&self) -> &T {
        &self.elements[Self::LAGRANGE_FIRST]
    }

    pub fn lagrange_last(&self) -> &T {
        &self.elements[Self::LAGRANGE_LAST]
    }

    pub fn lagrange_first_mut(&mut self) -> &mut T {
        &mut self.elements[Self::LAGRANGE_FIRST]
    }

    pub fn lagrange_last_mut(&mut self) -> &mut T {
        &mut self.elements[Self::LAGRANGE_LAST]
    }
}
