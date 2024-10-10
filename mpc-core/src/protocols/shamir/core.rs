//! # Shamir Core
//!
//! This module implements core functionality of shamir share and combine operations

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use rand::Rng;

pub(crate) fn share<F: PrimeField, R: Rng>(
    secret: F,
    num_shares: usize,
    degree: usize,
    rng: &mut R,
) -> Vec<F> {
    let mut shares = Vec::with_capacity(num_shares);
    let mut coeffs = Vec::with_capacity(degree);
    for _ in 0..degree {
        coeffs.push(F::rand(rng));
    }
    for i in 1..=num_shares {
        let mut share = secret;
        let i = F::from(i as u64);
        let mut x_pow = i;
        for coeff in coeffs.iter() {
            share += x_pow * coeff;
            x_pow *= i;
        }
        shares.push(share);
    }
    shares
}

pub(crate) fn share_point<C: CurveGroup, R: Rng>(
    secret: C,
    num_shares: usize,
    degree: usize,
    rng: &mut R,
) -> Vec<C> {
    let mut shares = Vec::with_capacity(num_shares);
    let mut coeffs = Vec::with_capacity(degree);
    for _ in 0..degree {
        coeffs.push(C::rand(rng));
    }
    for i in 1..=num_shares {
        let mut share = secret;
        let i = C::ScalarField::from(i as u64);
        let mut x_pow = i;
        for coeff in coeffs.iter() {
            share += coeff.mul(x_pow);
            x_pow *= i;
        }
        shares.push(share);
    }
    shares
}

pub(crate) fn lagrange_from_coeff<F: PrimeField>(coeffs: &[usize]) -> Vec<F> {
    let num = coeffs.len();
    let mut res = Vec::with_capacity(num);
    for i in coeffs.iter() {
        let mut num = F::one();
        let mut den = F::one();
        let i_ = F::from(*i as u64);
        for j in coeffs.iter() {
            if i != j {
                let j_ = F::from(*j as u64);
                num *= j_;
                den *= j_ - i_;
            }
        }
        let res_ = num * den.inverse().unwrap();
        res.push(res_);
    }
    res
}

#[cfg(test)]
pub(crate) fn lagrange<F: PrimeField>(amount: usize) -> Vec<F> {
    let mut res = Vec::with_capacity(amount);
    for i in 1..=amount {
        let mut num = F::one();
        let mut den = F::one();
        let i_ = F::from(i as u64);
        for j in 1..=amount {
            if i != j {
                let j_ = F::from(j as u64);
                num *= j_;
                den *= j_ - i_;
            }
        }
        let res_ = num * den.inverse().unwrap();
        res.push(res_);
    }
    res
}

pub(crate) fn reconstruct<F: PrimeField>(shares: &[F], lagrange: &[F]) -> F {
    debug_assert_eq!(shares.len(), lagrange.len());
    let mut res = F::zero();
    for (s, l) in shares.iter().zip(lagrange.iter()) {
        res += *s * l
    }

    res
}

/// Reconstructs a curve point from its Shamir shares and lagrange coefficients.
pub fn reconstruct_point<C: CurveGroup>(shares: &[C], lagrange: &[C::ScalarField]) -> C {
    debug_assert_eq!(shares.len(), lagrange.len());
    let mut res = C::zero();
    for (s, l) in shares.iter().zip(lagrange.iter()) {
        res += *s * l
    }

    res
}

#[cfg(test)]
mod shamir_test {
    use super::*;
    use ark_ff::UniformRand;
    use rand::{seq::IteratorRandom, SeedableRng};
    use rand_chacha::ChaCha12Rng;

    const TESTRUNS: usize = 5;

    fn test_shamir<F: PrimeField, const NUM_PARTIES: usize, const DEGREE: usize>() {
        let mut rng = ChaCha12Rng::from_entropy();

        for _ in 0..TESTRUNS {
            let secret = F::rand(&mut rng);
            let shares = super::share(secret, NUM_PARTIES, DEGREE, &mut rng);

            // Test first D+1 shares
            let lagrange = super::lagrange(DEGREE + 1);
            let reconstructed = super::reconstruct(&shares[..=DEGREE], &lagrange);
            assert_eq!(secret, reconstructed);

            // Test random D+1 shares
            let parties = (1..=NUM_PARTIES).choose_multiple(&mut rng, DEGREE + 1);
            let shares = parties.iter().map(|&i| shares[i - 1]).collect::<Vec<_>>();
            let lagrange = super::lagrange_from_coeff(&parties);
            let reconstructed = super::reconstruct(&shares, &lagrange);
            assert_eq!(secret, reconstructed);
        }
    }

    fn test_shamir_point<C: CurveGroup, const NUM_PARTIES: usize, const DEGREE: usize>() {
        let mut rng = ChaCha12Rng::from_entropy();

        for _ in 0..TESTRUNS {
            let secret = C::rand(&mut rng);
            let shares = super::share_point(secret, NUM_PARTIES, DEGREE, &mut rng);

            // Test first D+1 shares
            let lagrange = super::lagrange(DEGREE + 1);
            let reconstructed = super::reconstruct_point(&shares[..=DEGREE], &lagrange);
            assert_eq!(secret, reconstructed);

            // Test random D+1 shares
            let parties = (1..=NUM_PARTIES).choose_multiple(&mut rng, DEGREE + 1);
            let shares = parties.iter().map(|&i| shares[i - 1]).collect::<Vec<_>>();
            let lagrange = super::lagrange_from_coeff(&parties);
            let reconstructed = super::reconstruct_point(&shares, &lagrange);
            assert_eq!(secret, reconstructed);
        }
    }

    fn test_shamir_field_to_point<C: CurveGroup, const NUM_PARTIES: usize, const DEGREE: usize>() {
        let mut rng = ChaCha12Rng::from_entropy();

        for _ in 0..TESTRUNS {
            let secret = C::ScalarField::rand(&mut rng);
            let shares = super::share(secret, NUM_PARTIES, DEGREE, &mut rng);

            // To point
            let secret = C::generator().mul(secret);
            let shares = shares
                .into_iter()
                .map(|s| C::generator().mul(s))
                .collect::<Vec<_>>();

            // Test first D+1 shares
            let lagrange = super::lagrange(DEGREE + 1);
            let reconstructed = super::reconstruct_point(&shares[..=DEGREE], &lagrange);
            assert_eq!(secret, reconstructed);

            // Test random D+1 shares
            let parties = (1..=NUM_PARTIES).choose_multiple(&mut rng, DEGREE + 1);
            let shares = parties.iter().map(|&i| shares[i - 1]).collect::<Vec<_>>();
            let lagrange = super::lagrange_from_coeff(&parties);
            let reconstructed = super::reconstruct_point(&shares, &lagrange);
            assert_eq!(secret, reconstructed);
        }
    }

    #[test]
    fn test_shamir_3_1() {
        const NUM_PARTIES: usize = 3;
        const DEGREE: usize = 1;
        test_shamir::<ark_bn254::Fr, NUM_PARTIES, DEGREE>();
        test_shamir_point::<ark_bn254::G1Projective, NUM_PARTIES, DEGREE>();
        test_shamir_field_to_point::<ark_bn254::G1Projective, NUM_PARTIES, DEGREE>();
    }

    #[test]
    fn test_shamir_10_6() {
        const NUM_PARTIES: usize = 10;
        const DEGREE: usize = 6;
        test_shamir::<ark_bn254::Fr, NUM_PARTIES, DEGREE>();
        test_shamir_point::<ark_bn254::G1Projective, NUM_PARTIES, DEGREE>();
        test_shamir_field_to_point::<ark_bn254::G1Projective, NUM_PARTIES, DEGREE>();
    }
}
