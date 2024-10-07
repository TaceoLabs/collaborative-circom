use crate::{
    honk_curve::HonkCurve,
    poseidon2::poseidon2_permutation::Poseidon2,
    prover::{HonkProofError, HonkProofResult},
    sponge_hasher::{FieldHash, FieldSponge},
    types::HonkProof,
};
use ark_ec::AffineRepr;
use ark_ff::{PrimeField, Zero};
use std::{collections::BTreeMap, ops::Index};

pub type TranscriptFieldType = ark_bn254::Fr;
pub type Poseidon2Sponge =
    FieldSponge<TranscriptFieldType, 4, 3, Poseidon2<TranscriptFieldType, 4, 5>>;

pub trait TranscriptHasher<F: PrimeField> {
    fn hash(buffer: Vec<F>) -> F;
}

impl<F: PrimeField, const T: usize, const R: usize, H: FieldHash<F, T> + Default>
    TranscriptHasher<F> for FieldSponge<F, T, R, H>
{
    fn hash(buffer: Vec<F>) -> F {
        Self::hash_fixed_lenth::<1>(&buffer)[0]
    }
}

pub struct Transcript<F, H>
where
    F: PrimeField,
    H: TranscriptHasher<F>,
{
    proof_data: Vec<F>,
    manifest: TranscriptManifest,
    num_frs_written: usize, // the number of bb::frs written to proof_data by the prover or the verifier
    num_frs_read: usize,    // the number of bb::frs read from proof_data by the verifier
    round_number: usize,
    is_first_challenge: bool,
    current_round_data: Vec<F>,
    previous_challenge: F,
    phantom_data: std::marker::PhantomData<H>,
}

impl<F, H> Default for Transcript<F, H>
where
    F: PrimeField,
    H: TranscriptHasher<F>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F, H> Transcript<F, H>
where
    F: PrimeField,
    H: TranscriptHasher<F>,
{
    pub fn new() -> Self {
        Self {
            proof_data: Default::default(),
            manifest: Default::default(),
            num_frs_written: 0,
            num_frs_read: 0,
            round_number: 0,
            is_first_challenge: true,
            current_round_data: Default::default(),
            previous_challenge: Default::default(),
            phantom_data: Default::default(),
        }
    }

    pub fn new_verifier(proof: HonkProof<F>) -> Self {
        Self {
            proof_data: proof.inner(),
            manifest: Default::default(),
            num_frs_written: 0,
            num_frs_read: 0,
            round_number: 0,
            is_first_challenge: true,
            current_round_data: Default::default(),
            previous_challenge: Default::default(),
            phantom_data: Default::default(),
        }
    }

    pub fn get_proof(self) -> HonkProof<F> {
        HonkProof::new(self.proof_data)
    }

    #[allow(unused)]
    pub(crate) fn print(&self) {
        self.manifest.print();
    }

    #[allow(unused)]
    pub(crate) fn get_manifest(&self) -> &TranscriptManifest {
        &self.manifest
    }

    fn consume_prover_elements(&mut self, label: String, elements: &[F]) {
        // Add an entry to the current round of the manifest
        let len = elements.len();
        self.manifest.add_entry(self.round_number, label, len);
        self.current_round_data.extend(elements);
        self.num_frs_written += len;
    }

    fn convert_point<P: HonkCurve<F>>(element: P::G1Affine) -> Vec<F> {
        let (x, y) = if element.is_zero() {
            // we are at infinity
            (P::BaseField::zero(), P::BaseField::zero())
        } else {
            P::g1_affine_to_xy(&element)
        };

        let mut res = P::convert_basefield_into(&x);
        res.extend(P::convert_basefield_into(&y));

        res
    }

    fn send_to_verifier(&mut self, label: String, elements: &[F]) {
        self.proof_data.extend(elements);
        self.consume_prover_elements(label, elements);
    }

    pub fn send_fr_to_verifier<P: HonkCurve<F>>(&mut self, label: String, element: P::ScalarField) {
        let elements = P::convert_scalarfield_into(&element);
        self.send_to_verifier(label, &elements);
    }

    pub fn send_u64_to_verifier(&mut self, label: String, element: u64) {
        let el = F::from(element);
        self.send_to_verifier(label, &[el]);
    }

    pub fn send_point_to_verifier<P: HonkCurve<F>>(&mut self, label: String, element: P::G1Affine) {
        let elements = Self::convert_point::<P>(element);
        self.send_to_verifier(label, &elements);
    }

    pub fn send_fr_iter_to_verifier<
        'a,
        P: HonkCurve<F>,
        I: IntoIterator<Item = &'a P::ScalarField>,
    >(
        &mut self,
        label: String,
        element: I,
    ) {
        let elements = element
            .into_iter()
            .flat_map(P::convert_scalarfield_into)
            .collect::<Vec<_>>();
        self.send_to_verifier(label, &elements);
    }

    fn receive_n_from_prover(&mut self, label: String, n: usize) -> HonkProofResult<Vec<F>> {
        if self.num_frs_read + n > self.proof_data.len() {
            return Err(HonkProofError::ProofTooSmall);
        }
        let elements = self.proof_data[self.num_frs_read..self.num_frs_read + n].to_owned();
        self.num_frs_read += n;

        self.consume_prover_elements(label, &elements);
        Ok(elements)
    }

    pub(super) fn receive_u64_from_prover(&mut self, label: String) -> HonkProofResult<u64> {
        let element = self.receive_n_from_prover(label, 1)?[0];
        let r = element.into_bigint();
        Ok(r.as_ref().first().unwrap().to_owned())
    }

    pub(super) fn receive_fr_from_prover<P: HonkCurve<F>>(
        &mut self,
        label: String,
    ) -> HonkProofResult<P::ScalarField> {
        let elements = self.receive_n_from_prover(label, P::NUM_SCALARFIELD_ELEMENTS)?;

        Ok(P::convert_scalarfield_back(&elements))
    }

    pub(super) fn receive_point_from_prover<P: HonkCurve<F>>(
        &mut self,
        label: String,
    ) -> HonkProofResult<P::G1Affine> {
        let elements = self.receive_n_from_prover(label, P::NUM_BASEFIELD_ELEMENTS * 2)?;

        let coords = elements
            .chunks_exact(P::NUM_BASEFIELD_ELEMENTS)
            .map(P::convert_basefield_back)
            .collect::<Vec<_>>();

        let x = coords[0];
        let y = coords[1];

        let res = if x.is_zero() && y.is_zero() {
            P::G1Affine::zero()
        } else {
            P::g1_affine_from_xy(x, y)
        };

        Ok(res)
    }

    pub(super) fn receive_fr_vec_from_verifier<P: HonkCurve<F>>(
        &mut self,
        label: String,
        n: usize,
    ) -> HonkProofResult<Vec<P::ScalarField>> {
        let elements = self.receive_n_from_prover(label, P::NUM_SCALARFIELD_ELEMENTS * n)?;

        let elements = elements
            .chunks_exact(P::NUM_SCALARFIELD_ELEMENTS)
            .map(P::convert_scalarfield_back)
            .collect();

        Ok(elements)
    }

    pub(super) fn receive_fr_array_from_verifier<P: HonkCurve<F>, const SIZE: usize>(
        &mut self,
        label: String,
    ) -> HonkProofResult<[P::ScalarField; SIZE]> {
        let mut res: [P::ScalarField; SIZE] = [P::ScalarField::zero(); SIZE];
        let elements = self.receive_n_from_prover(label, P::NUM_SCALARFIELD_ELEMENTS * SIZE)?;

        for (src, des) in elements
            .chunks_exact(P::NUM_SCALARFIELD_ELEMENTS)
            .zip(res.iter_mut())
        {
            let el = P::convert_scalarfield_back(src);
            *des = el;
        }
        Ok(res)
    }

    pub(super) fn get_next_challenge_buffer(&mut self) -> F {
        // Prevent challenge generation if this is the first challenge we're generating,
        // AND nothing was sent by the prover.
        if self.is_first_challenge {
            assert!(!self.current_round_data.is_empty());
        }
        // concatenate the previous challenge (if this is not the first challenge) with the current round data.
        // AZTEC TODO(Adrian): Do we want to use a domain separator as the initial challenge buffer?
        // We could be cheeky and use the hash of the manifest as domain separator, which would prevent us from having
        // to domain separate all the data. (See https://safe-hash.dev)

        let mut full_buffer = Vec::new();
        std::mem::swap(&mut full_buffer, &mut self.current_round_data);

        if self.is_first_challenge {
            // Update is_first_challenge for the future
            self.is_first_challenge = false;
        } else {
            // if not the first challenge, we can use the previous_challenge
            full_buffer.insert(0, self.previous_challenge);
        }

        // Hash the full buffer with poseidon2, which is believed to be a collision resistant hash function and a random
        // oracle, removing the need to pre-hash to compress and then hash with a random oracle, as we previously did
        // with Pedersen and Blake3s.
        let new_challenge = H::hash(full_buffer);

        // update previous challenge buffer for next time we call this function
        self.previous_challenge = new_challenge;
        new_challenge
    }

    pub fn get_challenge<P: HonkCurve<F>>(&mut self, label: String) -> P::ScalarField {
        self.manifest.add_challenge(self.round_number, &[label]);
        let challenge = self.get_next_challenge_buffer();
        let res = P::convert_destinationfield_to_scalarfield(&challenge);
        self.round_number += 1;
        res
    }

    pub fn get_challenges<P: HonkCurve<F>>(&mut self, labels: &[String]) -> Vec<P::ScalarField> {
        self.manifest.add_challenge(self.round_number, labels);
        let mut res = Vec::with_capacity(labels.len());
        for _ in 0..labels.len() {
            let challenge = self.get_next_challenge_buffer();
            let res_ = P::convert_destinationfield_to_scalarfield(&challenge);
            res.push(res_);
        }
        self.round_number += 1;
        res
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub(crate) struct RoundData {
    challenge_label: Vec<String>,
    entries: Vec<(String, usize)>,
}

impl RoundData {
    pub(crate) fn print(&self) {
        for label in self.challenge_label.iter() {
            println!("\tchallenge: {}", label);
        }
        for entry in self.entries.iter() {
            println!("\telement ({}): {}", entry.1, entry.0);
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TranscriptManifest {
    manifest: BTreeMap<usize, RoundData>,
}

impl TranscriptManifest {
    pub(crate) fn print(&self) {
        for round in self.manifest.iter() {
            println!("Round: {}", round.0);
            round.1.print();
        }
    }

    pub(crate) fn add_challenge(&mut self, round: usize, labels: &[String]) {
        self.manifest
            .entry(round)
            .or_default()
            .challenge_label
            .extend_from_slice(labels);
    }

    pub(crate) fn add_entry(&mut self, round: usize, element_label: String, element_size: usize) {
        self.manifest
            .entry(round)
            .or_default()
            .entries
            .push((element_label, element_size));
    }

    #[allow(unused)]
    pub(crate) fn size(&self) -> usize {
        self.manifest.len()
    }
}

impl Index<usize> for TranscriptManifest {
    type Output = RoundData;

    fn index(&self, index: usize) -> &Self::Output {
        &self.manifest[&index]
    }
}
