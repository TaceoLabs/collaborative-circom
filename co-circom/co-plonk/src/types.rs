use ark_ec::AffineRepr;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use circom_types::plonk::ZKey;
use co_circom_snarks::SharedWitness;
use std::marker::PhantomData;

use crate::{mpc::CircomPlonkProver, PlonkProofError, PlonkProofResult};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use num_traits::Zero;
use sha3::{Digest, Keccak256};

pub(super) type Keccak256Transcript<P> = Transcript<Keccak256, P>;

pub(super) struct Transcript<D, P>
where
    D: Digest,
    P: Pairing,
{
    digest: D,
    phantom_data: PhantomData<P>,
}

pub(super) struct PolyEval<P: Pairing, T: CircomPlonkProver<P>> {
    pub(super) poly: Vec<T::ArithmeticShare>,
    pub(super) eval: Vec<T::ArithmeticShare>,
}

pub(super) struct Domains<F: PrimeField> {
    pub(super) domain: Radix2EvaluationDomain<F>,
    pub(super) extended_domain: Radix2EvaluationDomain<F>,
    pub(super) root_of_unity_pow: F,
    pub(super) root_of_unity_2: F,
    pub(super) root_of_unity_pow_2: F,
}

pub(super) struct PlonkWitness<P: Pairing, T: CircomPlonkProver<P>> {
    pub(super) public_inputs: Vec<P::ScalarField>,
    pub(super) witness: Vec<T::ArithmeticShare>,
    pub(super) addition_witness: Vec<T::ArithmeticShare>,
}

pub(super) struct PlonkData<'a, P: Pairing, T: CircomPlonkProver<P>> {
    pub(super) witness: PlonkWitness<P, T>,
    pub(super) zkey: &'a ZKey<P>,
}

impl<F: PrimeField> Domains<F> {
    pub(super) fn new(domain_size: usize) -> PlonkProofResult<Self> {
        tracing::debug!("building domains/roots of unity for domain size: {domain_size}");
        if domain_size & (domain_size - 1) != 0 || domain_size == 0 {
            Err(PlonkProofError::InvalidDomainSize(domain_size))
        } else {
            let mut domain = Radix2EvaluationDomain::<F>::new(domain_size)
                .ok_or(PlonkProofError::PolynomialDegreeTooLarge)?;
            let mut extended_domain = Radix2EvaluationDomain::<F>::new(domain_size * 4)
                .ok_or(PlonkProofError::PolynomialDegreeTooLarge)?;
            let (_, roots_of_unity) = co_circom_snarks::utils::roots_of_unity();
            let pow = usize::try_from(domain_size.ilog2()).expect("u32 fits into usize");

            tracing::trace!(
                "setting arkworks root of unity (domain size) by hand: {}",
                roots_of_unity[pow]
            );
            tracing::trace!(
                "setting arkworks root of unity (extended) by hand: {}",
                roots_of_unity[pow + 2]
            );
            // snarkjs and arkworks use different roots of unity to compute (i)fft.
            // therefore we compute the roots of unity by hand like snarkjs and
            // set the root of unity accordingly by hand
            domain.group_gen = roots_of_unity[pow];
            domain.group_gen_inv = domain.group_gen.inverse().expect("can compute inverse");
            extended_domain.group_gen = roots_of_unity[pow + 2];
            extended_domain.group_gen_inv = extended_domain
                .group_gen
                .inverse()
                .expect("can compute inverse");

            Ok(Self {
                domain,
                extended_domain,
                root_of_unity_2: roots_of_unity[2],
                root_of_unity_pow: roots_of_unity[pow],
                root_of_unity_pow_2: roots_of_unity[pow + 2],
            })
        }
    }
}
impl<P: Pairing, T: CircomPlonkProver<P>> PlonkWitness<P, T> {
    pub(super) fn new(
        mut shared_witness: SharedWitness<P::ScalarField, T::ArithmeticShare>,
        n_additions: usize,
    ) -> Self {
        // we have a Groth16 witness, therefore there is a leading one in the witness.
        // we just write zero here instead of one to mirror snarkjs.
        shared_witness.public_inputs[0] = P::ScalarField::zero();
        Self {
            public_inputs: shared_witness.public_inputs,
            witness: shared_witness.witness,
            addition_witness: Vec::with_capacity(n_additions),
        }
    }
}
impl<P: Pairing> Default for Keccak256Transcript<P> {
    fn default() -> Self {
        Self {
            digest: Default::default(),
            phantom_data: Default::default(),
        }
    }
}

impl<D, P> Transcript<D, P>
where
    D: Digest,
    P: Pairing,
{
    pub(super) fn add_scalar(&mut self, scalar: P::ScalarField) {
        let mut buf = vec![];
        scalar
            .serialize_uncompressed(&mut buf)
            .expect("Can Fr write into Vec<u8>");
        buf.reverse();
        self.digest.update(&buf);
    }

    pub(super) fn add_point(&mut self, point: P::G1Affine) {
        let byte_len: usize = P::BaseField::MODULUS_BIT_SIZE
            .div_ceil(8)
            .try_into()
            .expect("u32 fits into usize");
        let mut buf = Vec::with_capacity(byte_len);
        if let Some((x, y)) = point.xy() {
            x.serialize_uncompressed(&mut buf)
                .expect("Can write Fq into Vec<u8>");
            buf.reverse();
            self.digest.update(&buf);
            buf.clear();
            y.serialize_uncompressed(&mut buf)
                .expect("Can write Fq into Vec<u8>");
            buf.reverse();
            self.digest.update(&buf);
        } else {
            // we are at infinity - in this case, snarkjs writes (MODULUS_BIT_SIZE / 8) Zero-bytes
            // to the input buffer. If we serialize with arkworks, we would
            // get (MODULUS_BIT_SIZE / 8 - 1) Zero-bytes with a trailing byte indicating the length of
            // the serialized group element, resulting in an incompatible hash. Therefore we simple resize
            // the buffer with Zeros and write it to the hash instance.
            buf.resize(byte_len * 2, 0);
            self.digest.update(&buf);
        }
    }

    pub(super) fn get_challenge(self) -> P::ScalarField {
        let bytes = self.digest.finalize();
        P::ScalarField::from_be_bytes_mod_order(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::Keccak256Transcript;
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use std::str::FromStr;

    //this is copied from circom-type/groth16/mod/test_utils. Maybe we can
    //create a test-utils crate where we gather such definitions
    macro_rules! to_g1_bn254 {
        ($x: expr, $y: expr) => {
            <ark_bn254::Bn254 as Pairing>::G1Affine::new(
                ark_bn254::Fq::from_str($x).unwrap(),
                ark_bn254::Fq::from_str($y).unwrap(),
            )
        };
    }

    #[test]
    fn test_keccak_transcript() {
        let mut transcript = Keccak256Transcript::<Bn254>::default();
        transcript.add_point(to_g1_bn254!(
            "20825949499069110345561489838956415747250622568151984013116057026259498945798",
            "4633888776580597789536778273539625207986785465104156818397550354894072332743"
        ));
        transcript.add_point(to_g1_bn254!(
            "13502414797941204782598195942532580786194839256223737894432362681935424485706",
            "18673738305240077401477088441313771484023070622513584695135539045403188608753"
        ));
        transcript.add_point(ark_bn254::G1Affine::identity());
        transcript.add_scalar(
            ark_bn254::Fr::from_str(
                "18493166935391704183319420574241503914733913248159936156014286513312199455",
            )
            .unwrap(),
        );
        transcript.add_point(to_g1_bn254!(
            "20825949499069110345561489838956415747250622568151984013116057026259498945798",
            "17254354095258677432709627471717649880709525692193666844291487539751153875840"
        ));
        transcript.add_scalar(
            ark_bn254::Fr::from_str(
                "18493166935391704183319420574241503914733913248159936156014286513312199455",
            )
            .unwrap(),
        );
        let is_challenge = transcript.get_challenge();
        assert_eq!(
            ark_bn254::Fr::from_str(
                "16679357168864952869972350724842033299710155825088243463992129238972103889312",
            )
            .unwrap(),
            is_challenge
        );
    }
}
