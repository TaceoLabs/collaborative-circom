//! # REP3
//!
//! This module implements the rep3 share and combine operations

pub mod arithmetic;
pub mod binary;
pub mod conversion;
mod detail;
pub mod id;
pub mod lut;
pub mod network;
pub mod pointshare;
pub mod poly;
pub mod rngs;

use ark_ec::CurveGroup;
use num_bigint::BigUint;

use ark_ff::{One, PrimeField};
use rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, SeedableRng};

pub use arithmetic::types::Rep3PrimeFieldShare;
pub use binary::types::Rep3BigUintShare;
pub use pointshare::Rep3PointShare;

pub(crate) type IoResult<T> = std::io::Result<T>;

#[derive(Clone)]
pub enum SeededType<T: Clone, U: Clone> {
    Shares(T),
    Seed(U),
}

#[derive(Clone)]
pub struct ReplicatedSeedType<T: Clone, U: Clone> {
    pub(crate) a: SeededType<T, U>,
    pub(crate) b: SeededType<T, U>,
}

/// Secret shares a field element using replicated secret sharing and the provided random number generator. The field element is split into three additive shares, where each party holds two. The outputs are of type [Rep3PrimeFieldShare].
pub fn share_field_element<F: PrimeField, R: Rng + CryptoRng>(
    val: F,
    rng: &mut R,
) -> [Rep3PrimeFieldShare<F>; 3] {
    let a = F::rand(rng);
    let b = F::rand(rng);
    let c = val - a - b;
    let share1 = Rep3PrimeFieldShare::new(a, c);
    let share2 = Rep3PrimeFieldShare::new(b, a);
    let share3 = Rep3PrimeFieldShare::new(c, b);
    [share1, share2, share3]
}

/// Secret shares a field element using additive secret sharing and the provided random number generator. The field element is split into three additive shares. The outputs are of type [F].
pub fn share_field_element_additive<F: PrimeField, R: Rng + CryptoRng>(
    val: F,
    rng: &mut R,
) -> [F; 3] {
    let a = F::rand(rng);
    let b = F::rand(rng);
    let c = val - a - b;
    [a, b, c]
}

/// Secret shares a field element using replicated secret sharing, whereas only one additive share is stored while the others are compressed as seeds derived form the provided random number generator. The outputs are of type [Rep3ShareType].
pub fn share_field_element_seeded<
    F: PrimeField,
    R: Rng + CryptoRng,
    U: Rng + SeedableRng + CryptoRng,
>(
    val: F,
    rng: &mut R,
) -> [ReplicatedSeedType<F, U::Seed>; 3]
where
    U::Seed: Clone,
    Standard: Distribution<U::Seed>,
{
    let seed_b = rng.gen::<U::Seed>();
    let seed_c = rng.gen::<U::Seed>();

    let mut rng_b = U::from_seed(seed_b.to_owned());
    let mut rng_c = U::from_seed(seed_c.to_owned());

    let b = F::rand(&mut rng_b);
    let c = F::rand(&mut rng_c);
    let a = val - b - c;

    let a = SeededType::Shares(a);
    let b = SeededType::Seed(seed_b);
    let c = SeededType::Seed(seed_c);

    let share1 = ReplicatedSeedType {
        a: a.to_owned(),
        b: c.to_owned(),
    };
    let share2 = ReplicatedSeedType {
        a: b.to_owned(),
        b: a,
    };
    let share3 = ReplicatedSeedType { a: c, b };
    [share1, share2, share3]
}

/// Secret shares a field element using additive secret sharing, whereas only one additive share is stored while the others are compressed as seeds derived form the provided random number generator. The outputs are of type [SeededType].
pub fn share_field_element_additive_seeded<
    F: PrimeField,
    R: Rng + CryptoRng,
    U: Rng + SeedableRng + CryptoRng,
>(
    val: F,
    rng: &mut R,
) -> [SeededType<F, U::Seed>; 3]
where
    U::Seed: Clone,
    Standard: Distribution<U::Seed>,
{
    let seed_b = rng.gen::<U::Seed>();
    let seed_c = rng.gen::<U::Seed>();

    let mut rng_b = U::from_seed(seed_b.to_owned());
    let mut rng_c = U::from_seed(seed_c.to_owned());

    let b = F::rand(&mut rng_b);
    let c = F::rand(&mut rng_c);
    let a = val - b - c;

    let a = SeededType::Shares(a);
    let b = SeededType::Seed(seed_b);
    let c = SeededType::Seed(seed_c);

    [a, b, c]
}

/// Secret shares a vector of field element using replicated secret sharing and the provided random number generator. The field elements are split into three additive shares each, where each party holds two. The outputs are of type [Rep3PrimeFieldShareVec].
pub fn share_field_elements<F: PrimeField, R: Rng + CryptoRng>(
    vals: &[F],
    rng: &mut R,
) -> [Vec<Rep3PrimeFieldShare<F>>; 3] {
    let mut shares1 = Vec::with_capacity(vals.len());
    let mut shares2 = Vec::with_capacity(vals.len());
    let mut shares3 = Vec::with_capacity(vals.len());
    for val in vals {
        let [share1, share2, share3] = share_field_element(*val, rng);
        shares1.push(share1);
        shares2.push(share2);
        shares3.push(share3);
    }
    [shares1, shares2, shares3]
}

/// Secret shares a vector of field element using additive secret sharing and the provided random number generator. The field elements are split into three additive shares each. The outputs are of type [Rep3PrimeFieldShareVec].
pub fn share_field_elements_additive<F: PrimeField, R: Rng + CryptoRng>(
    vals: &[F],
    rng: &mut R,
) -> [Vec<F>; 3] {
    let mut shares1 = Vec::with_capacity(vals.len());
    let mut shares2 = Vec::with_capacity(vals.len());
    let mut shares3 = Vec::with_capacity(vals.len());
    for val in vals {
        let [share1, share2, share3] = share_field_element_additive(*val, rng);
        shares1.push(share1);
        shares2.push(share2);
        shares3.push(share3);
    }
    [shares1, shares2, shares3]
}

/// Secret shares a vector of field element using replicated secret sharing, whereas only one additive share is stored while the others are compressed as seeds derived form the provided random number generator. The outputs are of type [ReplicatedSeedType].
pub fn share_field_elements_seeded<
    F: PrimeField,
    R: Rng + CryptoRng,
    U: Rng + SeedableRng + CryptoRng,
>(
    vals: &[F],
    rng: &mut R,
) -> [ReplicatedSeedType<Vec<F>, U::Seed>; 3]
where
    U::Seed: Clone,
    Standard: Distribution<U::Seed>,
{
    let seed_b = rng.gen::<U::Seed>();
    let seed_c = rng.gen::<U::Seed>();

    let mut rng_b = U::from_seed(seed_b.to_owned());
    let mut rng_c = U::from_seed(seed_c.to_owned());

    let b = SeededType::Seed(seed_b);
    let c = SeededType::Seed(seed_c);

    let mut a = Vec::with_capacity(vals.len());
    for val in vals {
        let b_ = F::rand(&mut rng_b);
        let c_ = F::rand(&mut rng_c);
        let a_ = *val - b_ - c_;
        a.push(a_);
    }

    let a = SeededType::Shares(a);

    let share1 = ReplicatedSeedType {
        a: a.to_owned(),
        b: c.to_owned(),
    };
    let share2 = ReplicatedSeedType {
        a: b.to_owned(),
        b: a,
    };
    let share3 = ReplicatedSeedType { a: c, b };
    [share1, share2, share3]
}

/// Secret shares a vector of field element using additive secret sharing, whereas only one additive share is stored while the others are compressed as seeds derived form the provided random number generator. The outputs are of type [SeededType].
pub fn share_field_elements_additive_seeded<
    F: PrimeField,
    R: Rng + CryptoRng,
    U: Rng + SeedableRng + CryptoRng,
>(
    vals: &[F],
    rng: &mut R,
) -> [SeededType<Vec<F>, U::Seed>; 3]
where
    U::Seed: Clone,
    Standard: Distribution<U::Seed>,
{
    let seed_b = rng.gen::<U::Seed>();
    let seed_c = rng.gen::<U::Seed>();

    let mut rng_b = U::from_seed(seed_b.to_owned());
    let mut rng_c = U::from_seed(seed_c.to_owned());

    let b = SeededType::Seed(seed_b);
    let c = SeededType::Seed(seed_c);

    let mut a = Vec::with_capacity(vals.len());
    for val in vals {
        let b_ = F::rand(&mut rng_b);
        let c_ = F::rand(&mut rng_c);
        let a_ = *val - b_ - c_;
        a.push(a_);
    }

    let a = SeededType::Shares(a);

    [a, b, c]
}

/// Secret shares a field element using replicated secret sharing and the provided random number generator. The field element is split into three binary shares, where each party holds two. The outputs are of type [Rep3BigUintShare].
pub fn share_biguint<F: PrimeField, R: Rng + CryptoRng>(
    val: F,
    rng: &mut R,
) -> [Rep3BigUintShare<F>; 3] {
    let val: BigUint = val.into();
    let limbsize = F::MODULUS_BIT_SIZE.div_ceil(8);
    let mask = (BigUint::from(1u32) << F::MODULUS_BIT_SIZE) - BigUint::one();
    let a = BigUint::new((0..limbsize).map(|_| rng.gen()).collect()) & &mask;
    let b = BigUint::new((0..limbsize).map(|_| rng.gen()).collect()) & mask;

    let c = val ^ &a ^ &b;
    let share1 = Rep3BigUintShare::new(a.to_owned(), c.to_owned());
    let share2 = Rep3BigUintShare::new(b.to_owned(), a);
    let share3 = Rep3BigUintShare::new(c, b);
    [share1, share2, share3]
}

/// Secret shares a curve point using replicated secret sharing and the provided random number generator. The point is split into three additive shares, where each party holds two. The outputs are of type [Rep3PointShare].
pub fn share_curve_point<C: CurveGroup, R: Rng + CryptoRng>(
    val: C,
    rng: &mut R,
) -> [Rep3PointShare<C>; 3] {
    let a = C::rand(rng);
    let b = C::rand(rng);
    let c = val - a - b;
    let share1 = Rep3PointShare::new(a, c);
    let share2 = Rep3PointShare::new(b, a);
    let share3 = Rep3PointShare::new(c, b);
    [share1, share2, share3]
}

//TODO RENAME ME TO COMBINE_ARITHMETIC_SHARE
/// Reconstructs a field element from its arithmetic replicated shares.
pub fn combine_field_element<F: PrimeField>(
    share1: Rep3PrimeFieldShare<F>,
    share2: Rep3PrimeFieldShare<F>,
    share3: Rep3PrimeFieldShare<F>,
) -> F {
    share1.a + share2.a + share3.a
}

/// Reconstructs a vector of field elements from its arithmetic replicated shares.
/// # Panics
/// Panics if the provided `Vec` sizes do not match.
pub fn combine_field_elements<F: PrimeField>(
    share1: Vec<Rep3PrimeFieldShare<F>>,
    share2: Vec<Rep3PrimeFieldShare<F>>,
    share3: Vec<Rep3PrimeFieldShare<F>>,
) -> Vec<F> {
    assert_eq!(share1.len(), share2.len());
    assert_eq!(share2.len(), share3.len());

    itertools::multizip((share1.into_iter(), share2.into_iter(), share3.into_iter()))
        .map(|(x1, x2, x3)| x1.a + x2.a + x3.a)
        .collect::<Vec<_>>()
}

/// Reconstructs a value (represented as [BigUint]) from its binary replicated shares. Since binary operations can lead to results >= p, the result is not guaranteed to be a valid field element.
pub fn combine_binary_element<F: PrimeField>(
    share1: Rep3BigUintShare<F>,
    share2: Rep3BigUintShare<F>,
    share3: Rep3BigUintShare<F>,
) -> BigUint {
    share1.a ^ share2.a ^ share3.a
}

/// Reconstructs a curve point from its arithmetic replicated shares.
pub fn combine_curve_point<C: CurveGroup>(
    share1: Rep3PointShare<C>,
    share2: Rep3PointShare<C>,
    share3: Rep3PointShare<C>,
) -> C {
    share1.a + share2.a + share3.a
}
