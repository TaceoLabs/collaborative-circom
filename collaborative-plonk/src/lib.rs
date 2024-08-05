//! A Plonk proof protocol that uses a collaborative MPC protocol to generate the proof.

use ark_ec::pairing::Pairing;
use ark_ec::AffineRepr;
use ark_ff::Field;
use ark_ff::PrimeField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::SynthesisError;
use ark_serialize::CanonicalSerialize;
use circom_types::groth16::zkey;
use circom_types::plonk::ZKey;
use circom_types::traits::CircomArkworksPairingBridge;
use circom_types::traits::CircomArkworksPrimeFieldBridge;
use collaborative_groth16::groth16::CollaborativeGroth16;
use collaborative_groth16::groth16::SharedWitness;
use eyre::Result;
use mpc_core::traits::{
    EcMpcProtocol, FFTProvider, MSMProvider, PairingEcMpcProtocol, PrimeFieldMpcProtocol,
};
use num_traits::One;
use num_traits::Zero;
use round1::Round1Challenges;
use round1::Round1Proof;
use sha3::Digest;
use sha3::Keccak256;
use std::marker::PhantomData;
use std::ops::MulAssign;

mod round1;
pub(crate) mod types;

type FieldShare<T, P> = <T as PrimeFieldMpcProtocol<<P as Pairing>::ScalarField>>::FieldShare;
type FieldShareVec<T, P> = <T as PrimeFieldMpcProtocol<<P as Pairing>::ScalarField>>::FieldShareVec;
type PointShare<T, C> = <T as EcMpcProtocol<C>>::PointShare;

pub(crate) struct Domains<P: Pairing> {
    constraint_domain: GeneralEvaluationDomain<P::ScalarField>,
    constraint_domain4: GeneralEvaluationDomain<P::ScalarField>,
    phantom_data: PhantomData<P>,
}

impl<P: Pairing> Domains<P> {
    fn new(zkey: &ZKey<P>) -> Result<Self> {
        let domain1 = GeneralEvaluationDomain::<P::ScalarField>::new(zkey.n_constraints)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain2 = GeneralEvaluationDomain::<P::ScalarField>::new(zkey.n_constraints * 4)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        Ok(Self {
            constraint_domain: domain1,
            constraint_domain4: domain2,
            phantom_data: PhantomData,
        })
    }
}

enum Round<T, P: Pairing>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>
        + PairingEcMpcProtocol<P>
        + FFTProvider<P::ScalarField>
        + MSMProvider<P::G1>
        + MSMProvider<P::G2>,
    P::ScalarField: mpc_core::traits::FFTPostProcessing,
{
    Init,
    Round1 {
        domains: Domains<P>,
        challenges: Round1Challenges<T, P>,
    },
    Round2 {
        domains: Domains<P>,
        challenges: Round1Challenges<T, P>,
        proof: Round1Proof<P>,
    },
    Round3,
    Round4,
    Round5,
    Round6,
    Finished,
}

/// A Plonk proof protocol that uses a collaborative MPC protocol to generate the proof.
pub struct CollaborativePlonk<T, P: Pairing>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>
        + PairingEcMpcProtocol<P>
        + FFTProvider<P::ScalarField>
        + MSMProvider<P::G1>
        + MSMProvider<P::G2>,
    P::ScalarField: mpc_core::traits::FFTPostProcessing,
{
    pub(crate) driver: T,
    pub(crate) state: Round<T, P>,
    phantom_data: PhantomData<P>,
}

impl<T, P> CollaborativePlonk<T, P>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>
        + PairingEcMpcProtocol<P>
        + FFTProvider<P::ScalarField>
        + MSMProvider<P::G1>
        + MSMProvider<P::G2>,
    P::ScalarField: mpc_core::traits::FFTPostProcessing + CircomArkworksPrimeFieldBridge,
    P: Pairing + CircomArkworksPairingBridge,
    P::BaseField: CircomArkworksPrimeFieldBridge,
{
    /// Creates a new [CollaborativePlonk] protocol with a given MPC driver.
    pub fn new(driver: T) -> Self {
        Self {
            driver,
            state: Round::Init,
            phantom_data: PhantomData,
        }
    }

    pub fn proof(self) {}

    pub(crate) fn next_round(&mut self) {}
}

impl<T, P: Pairing> Round<T, P>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>
        + PairingEcMpcProtocol<P>
        + FFTProvider<P::ScalarField>
        + MSMProvider<P::G1>
        + MSMProvider<P::G2>,
    P::ScalarField: mpc_core::traits::FFTPostProcessing + CircomArkworksPrimeFieldBridge,
    P: Pairing + CircomArkworksPairingBridge,
    P::BaseField: CircomArkworksPrimeFieldBridge,
{
    fn next_round(
        self,
        driver: &mut T,
        zkey: &ZKey<P>,
        private_witness: &mut SharedWitness<T, P>,
    ) -> Result<Self> {
        match self {
            Round::Init => Self::init_round(driver, zkey, private_witness),
            Round::Round1 {
                domains,
                challenges,
            } => Self::round1(driver, domains, challenges, zkey, private_witness),
            Round::Round2 {
                domains,
                challenges,
                proof,
            } => todo!(),
            Round::Round3 => todo!(),
            Round::Round4 => todo!(),
            Round::Round5 => todo!(),
            Round::Round6 => todo!(),
            Round::Finished => todo!(),
        }
    }

    fn init_round(
        driver: &mut T,
        zkey: &ZKey<P>,
        private_witness: &mut SharedWitness<T, P>,
    ) -> Result<Self> {
        //TODO calculate additions
        //set first element to zero as it is not used
        private_witness.public_inputs[0] = P::ScalarField::zero();
        let domain1 = GeneralEvaluationDomain::<P::ScalarField>::new(zkey.n_constraints)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain2 = GeneralEvaluationDomain::<P::ScalarField>::new(zkey.n_constraints * 4)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        Ok(Round::Round1 {
            domains: Domains {
                constraint_domain: domain1,
                constraint_domain4: domain2,
                phantom_data: PhantomData,
            },

            challenges: Round1Challenges::random(driver)?,
        })
    }

    // TODO check if this is correct
    fn get_witness(
        driver: &mut T,
        private_witness: &SharedWitness<T, P>,
        zkey: &ZKey<P>,
        index: usize,
    ) -> FieldShare<T, P> {
        if index <= zkey.n_public {
            driver.promote_to_trivial_share(private_witness.public_inputs[index])
        } else {
            //subtract public values and the leading 1 in witness
            T::index_sharevec(&private_witness.witness, index - zkey.n_public - 1)
        }
    }

    fn blind_coefficients(
        driver: &mut T,
        poly: &FieldShareVec<T, P>,
        coeff: &[FieldShare<T, P>],
    ) -> Vec<FieldShare<T, P>> {
        let mut res = poly.clone().into_iter().collect::<Vec<_>>();
        for (p, c) in res.iter_mut().zip(coeff.iter()) {
            *p = driver.sub(p, c);
        }
        res.extend_from_slice(coeff);
        res
    }
}
