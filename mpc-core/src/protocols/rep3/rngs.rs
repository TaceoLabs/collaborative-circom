//! Rep3 RNGs
//!
//! This module contains implementations of rep3 rngs

use crate::RngType;
use ark_ec::CurveGroup;
use ark_ff::{One, PrimeField};
use num_bigint::BigUint;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;

#[derive(Debug)]
/// A correlated rng for rep3
pub struct Rep3CorrelatedRng {
    pub(crate) rand: Rep3Rand,
    pub(crate) bitcomp1: Rep3RandBitComp,
    pub(crate) bitcomp2: Rep3RandBitComp,
}

impl Rep3CorrelatedRng {
    /// Construct a new [`Rep3CorrelatedRng`]
    pub fn new(rand: Rep3Rand, bitcomp1: Rep3RandBitComp, bitcomp2: Rep3RandBitComp) -> Self {
        Self {
            rand,
            bitcomp1,
            bitcomp2,
        }
    }

    /// Create a fork of the current rng
    pub fn fork(&mut self) -> Self {
        let rand = self.rand.fork();
        let bitcomp1 = self.bitcomp1.fork();
        let bitcomp2 = self.bitcomp2.fork();
        Self {
            rand,
            bitcomp1,
            bitcomp2,
        }
    }
}

#[derive(Debug)]
/// Rep3 rng with this party's rng and the prev party's rng
pub struct Rep3Rand {
    rng1: RngType,
    rng2: RngType,
}

impl Rep3Rand {
    /// Construct a new [`Rep3Rand`]
    pub fn new(seed1: [u8; crate::SEED_SIZE], seed2: [u8; crate::SEED_SIZE]) -> Self {
        let rng1 = RngType::from_seed(seed1);
        let rng2 = RngType::from_seed(seed2);
        Self { rng1, rng2 }
    }

    /// Create a fork of this rng
    pub fn fork(&mut self) -> Self {
        let (seed1, seed2) = self.random_seeds();
        Self::new(seed1, seed2)
    }

    /// Generate a masking field element
    pub fn masking_field_element<F: PrimeField>(&mut self) -> F {
        let (a, b) = self.random_fes::<F>();
        a - b
    }

    /// Generate two random field elements
    pub fn random_fes<F: PrimeField>(&mut self) -> (F, F) {
        let a = F::rand(&mut self.rng1);
        let b = F::rand(&mut self.rng2);
        (a, b)
    }

    /// Generate a vector of masking field elements
    pub fn masking_field_elements_vec<F: PrimeField>(&mut self, len: usize) -> Vec<F> {
        let field_size = usize::try_from(F::MODULUS_BIT_SIZE)
            .expect("u32 fits into usize")
            .div_ceil(8);
        let mut a = vec![0_u8; field_size * len];
        let mut b = vec![0_u8; field_size * len];
        rayon::join(
            || {
                self.rng1.fill_bytes(&mut a);
            },
            || {
                self.rng2.fill_bytes(&mut b);
            },
        );
        a.par_chunks(field_size)
            .zip_eq(b.par_chunks(field_size))
            .with_min_len(512)
            .map(|(a, b)| F::from_be_bytes_mod_order(a) - F::from_be_bytes_mod_order(b))
            .collect()
    }

    /// Create a masking elliptic cureve element
    pub fn masking_ec_element<C: CurveGroup>(&mut self) -> C {
        let (a, b) = self.random_ecs::<C>();
        a - b
    }

    /// Generate two random elliptic cureve elements
    pub fn random_ecs<C: CurveGroup>(&mut self) -> (C, C) {
        let a = C::rand(&mut self.rng1);
        let b = C::rand(&mut self.rng2);
        (a, b)
    }

    /// Generate two random [`BigUint`]s with given `bitlen`
    pub fn random_biguint(&mut self, bitlen: usize) -> (BigUint, BigUint) {
        let limbsize = bitlen.div_ceil(8);
        let a = BigUint::new((0..limbsize).map(|_| self.rng1.gen()).collect());
        let b = BigUint::new((0..limbsize).map(|_| self.rng2.gen()).collect());
        let mask = (BigUint::from(1u32) << bitlen) - BigUint::one();
        (a & &mask, b & mask)
    }

    /// Generate a seed from each rng
    pub fn random_seeds(&mut self) -> ([u8; crate::SEED_SIZE], [u8; crate::SEED_SIZE]) {
        let seed1 = self.rng1.gen();
        let seed2 = self.rng2.gen();
        (seed1, seed2)
    }
}

/// This struct is responsible for creating random shares for the Binary to Arithmetic conversion. The approach is the following: for a final sharing x = x1 + x2 + x3, we want to have random values x2, x3 and subtract these from the original value x using a binary circuit to get the share x1. Hence, we need to sample random x2 and x3 and share them amongst the parties. One RandBitComp struct is responsible for either sampling x2 or x3. For sampling x2, parties 1 and 2 will get x2 in plain (since this is the final share of x), so they need to have a PRF key from all parties. party 3, however, will not get x2 in plain and must thus only be able to sample its shares of x2, requiring two PRF keys.
#[derive(Debug)]
pub struct Rep3RandBitComp {
    rng1: RngType,
    rng2: RngType,
    rng3: Option<RngType>,
}

impl Rep3RandBitComp {
    /// Contruct a new [`Rep3RandBitComp`] w rngs
    pub fn new_2keys(rng1: [u8; crate::SEED_SIZE], rng2: [u8; crate::SEED_SIZE]) -> Self {
        Self {
            rng1: RngType::from_seed(rng1),
            rng2: RngType::from_seed(rng2),
            rng3: None,
        }
    }

    /// Contruct a new [`Rep3RandBitComp`] with 3 rngs
    pub fn new_3keys(
        rng1: [u8; crate::SEED_SIZE],
        rng2: [u8; crate::SEED_SIZE],
        rng3: [u8; crate::SEED_SIZE],
    ) -> Self {
        Self {
            rng1: RngType::from_seed(rng1),
            rng2: RngType::from_seed(rng2),
            rng3: Some(RngType::from_seed(rng3)),
        }
    }

    /// Generate three random field elements
    pub fn random_fes_3keys<F: PrimeField>(&mut self) -> (F, F, F) {
        let a = F::rand(&mut self.rng1);
        let b = F::rand(&mut self.rng2);
        let c = if let Some(rng3) = &mut self.rng3 {
            F::rand(rng3)
        } else {
            unreachable!()
        };
        (a, b, c)
    }

    /// Create a fork of this rng
    pub fn fork(&mut self) -> Self {
        let rng1 = RngType::from_seed(self.rng1.gen());
        let rng2 = RngType::from_seed(self.rng2.gen());
        let rng3 = self.rng3.as_mut().map(|rng| RngType::from_seed(rng.gen()));
        Self { rng1, rng2, rng3 }
    }
}
