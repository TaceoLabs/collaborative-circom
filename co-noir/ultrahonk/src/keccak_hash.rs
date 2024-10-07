use crate::{prelude::TranscriptHasher, Serialize};
use ark_ff::PrimeField;
use sha3::{Digest, Keccak256};

impl<F: PrimeField> TranscriptHasher<F> for Keccak256 {
    fn hash(buffer: Vec<F>) -> F {
        // Losing 2 bits of this is not an issue -> we can just reduce mod p

        let vec = Serialize::to_buffer(&buffer, false);
        let mut hasher = Keccak256::default();
        hasher.update(vec);
        let hash_result = hasher.finalize();

        let mut offset = 0;
        Serialize::read_field_element(&hash_result, &mut offset)
    }
}
