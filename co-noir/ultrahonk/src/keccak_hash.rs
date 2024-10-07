use crate::prelude::TranscriptHasher;
use ark_ff::PrimeField;
use sha3::Keccak256;

impl<F: PrimeField> TranscriptHasher<F> for Keccak256 {
    fn hash(buffer: Vec<F>) -> F {
        todo!()
    }
}
