use crate::{
    round1::{Round1Challenges, Round1Polys, Round1Proof},
    types::{Keccak256Transcript, PolyEval},
    Domains, FieldShareVec, PlonkData, PlonkProofError, PlonkProofResult, Round,
};
use ark_ec::pairing::Pairing;
use ark_poly::GeneralEvaluationDomain;
use circom_types::{groth16::public_input, plonk::ZKey};
use collaborative_groth16::groth16::CollaborativeGroth16;
use mpc_core::traits::EcMpcProtocol;
use mpc_core::traits::{
    FFTPostProcessing, FFTProvider, MSMProvider, MontgomeryField, MpcToMontgomery,
    PairingEcMpcProtocol, PrimeFieldMpcProtocol,
};
use num_traits::One;

pub(super) struct Round2Challenges<T, P: Pairing>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>,
{
    pub(crate) b: [T::FieldShare; 11],
    pub(crate) beta: P::ScalarField,
    pub(crate) gamma: P::ScalarField,
}

pub(super) struct Round2Proof<P: Pairing> {
    pub(crate) commit_a: P::G1,
    pub(crate) commit_b: P::G1,
    pub(crate) commit_c: P::G1,
    pub(crate) commit_z: P::G1,
}

pub(super) struct Round2Polys<T, P: Pairing>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>,
{
    pub(crate) buffer_a: FieldShareVec<T, P>,
    pub(crate) buffer_b: FieldShareVec<T, P>,
    pub(crate) buffer_c: FieldShareVec<T, P>,
    pub(crate) poly_eval_a: PolyEval<T, P>,
    pub(crate) poly_eval_b: PolyEval<T, P>,
    pub(crate) poly_eval_c: PolyEval<T, P>,
    pub(crate) z: PolyEval<T, P>,
}

impl<T, P: Pairing> Round2Challenges<T, P>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>,
{
    fn new(
        round1_challenges: Round1Challenges<T, P>,
        beta: P::ScalarField,
        gamma: P::ScalarField,
    ) -> Self {
        Self {
            b: round1_challenges.b,
            beta,
            gamma,
        }
    }
}

impl<P: Pairing> Round2Proof<P> {
    fn new(round1_proof: Round1Proof<P>, commit_z: P::G1) -> Self {
        Self {
            commit_a: round1_proof.commit_a,
            commit_b: round1_proof.commit_b,
            commit_c: round1_proof.commit_c,
            commit_z,
        }
    }
}

impl<T, P: Pairing> Round2Polys<T, P>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>,
{
    fn new(polys: Round1Polys<T, P>, z: PolyEval<T, P>) -> Self {
        Self {
            buffer_a: polys.buffer_a,
            buffer_b: polys.buffer_b,
            buffer_c: polys.buffer_c,
            poly_eval_a: polys.poly_eval_a,
            poly_eval_b: polys.poly_eval_b,
            poly_eval_c: polys.poly_eval_c,
            z,
        }
    }
}

impl<T, P: Pairing> Round<T, P>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>
        + PairingEcMpcProtocol<P>
        + FFTProvider<P::ScalarField>
        + MSMProvider<P::G1>
        + MSMProvider<P::G2>,
    P::ScalarField: FFTPostProcessing,
{
    fn compute_z(
        driver: &mut T,
        zkey: &ZKey<P>,
        domains: &Domains<P>,
        challenges: &Round2Challenges<T, P>,
        wire_polys: &Round1Polys<T, P>,
    ) -> PlonkProofResult<PolyEval<T, P>> {
        let mut num_arr = Vec::with_capacity(zkey.domain_size);
        let mut den_arr = Vec::with_capacity(zkey.domain_size);

        num_arr.push(driver.promote_to_trivial_share(P::ScalarField::one()));
        den_arr.push(driver.promote_to_trivial_share(P::ScalarField::one()));

        let mut w = P::ScalarField::one();
        let pow_root_of_unity = domains.roots_of_unity[zkey.power];
        for i in 0..zkey.domain_size {
            let a = T::index_sharevec(&wire_polys.buffer_a, i);
            let b = T::index_sharevec(&wire_polys.buffer_b, i);
            let c = T::index_sharevec(&wire_polys.buffer_c, i);

            // Z(X) := numArr / denArr
            // numArr := (a + beta·ω + gamma)(b + beta·ω·k1 + gamma)(c + beta·ω·k2 + gamma)
            let betaw = challenges.beta * w;

            let n1 = driver.add_with_public(&betaw, &a);
            let n1 = driver.add_with_public(&challenges.gamma, &n1);

            let n2 = driver.add_with_public(&(zkey.verifying_key.k1 * betaw), &b);
            let n2 = driver.add_with_public(&challenges.gamma, &n2);

            let n3 = driver.add_with_public(&(zkey.verifying_key.k2 * betaw), &c);
            let n3 = driver.add_with_public(&challenges.gamma, &n3);

            let num = driver.mul(&n1, &n2)?;
            let mut num = driver.mul(&num, &n3)?;

            // denArr := (a + beta·sigma1 + gamma)(b + beta·sigma2 + gamma)(c + beta·sigma3 + gamma)
            let d1 =
                driver.add_with_public(&(challenges.beta * zkey.s1_poly.evaluations[i * 4]), &a);
            let d1 = driver.add_with_public(&challenges.gamma, &d1);

            let d2 =
                driver.add_with_public(&(challenges.beta * zkey.s2_poly.evaluations[i * 4]), &b);
            let d2 = driver.add_with_public(&challenges.gamma, &d2);

            let d3 =
                driver.add_with_public(&(challenges.beta * zkey.s3_poly.evaluations[i * 4]), &c);
            let d3 = driver.add_with_public(&challenges.gamma, &d3);

            let den = driver.mul(&d1, &d2)?;
            let mut den = driver.mul(&den, &d3)?;

            // Multiply current num value with the previous one saved in num_arr/den_arr
            if i != 0 {
                // TODO parallelize
                num = driver.mul(&num, &num_arr[i])?;
                den = driver.mul(&den, &den_arr[i])?;
            }

            if i == zkey.domain_size - 1 {
                num_arr[0] = num;
                den_arr[0] = den;
            } else {
                num_arr.push(num);
                den_arr.push(den);
            }

            w *= &pow_root_of_unity;
        }

        // Compute the inverse of denArr to compute in the next command the
        // division numArr/denArr by multiplying num · 1/denArr
        for den_arr in den_arr.iter_mut() {
            // TODO parallerlize
            *den_arr = driver.inv(den_arr)?;
        }
        let buffer_z = driver.mul_vec(&num_arr.into(), &den_arr.into())?;

        // Compute polynomial coefficients z(X) from buffer_z
        let poly_z = driver.ifft(&buffer_z, &domains.constraint_domain4);

        // Compute extended evaluations of z(X) polynomial
        let eval_z = driver.fft(poly_z.to_owned(), &domains.constraint_domain16);

        let poly_z = Self::blind_coefficients(driver, &poly_z, &challenges.b[6..9]);

        if poly_z.len() > zkey.domain_size + 3 {
            Err(PlonkProofError::PolynomialDegreeTooLarge)
        } else {
            Ok(PolyEval {
                poly: poly_z.into(),
                eval: eval_z,
            })
        }
    }

    pub(super) fn round2(
        driver: &mut T,
        domains: Domains<P>,
        challenges: Round1Challenges<T, P>,
        proof: Round1Proof<P>,
        polys: Round1Polys<T, P>,
        data: PlonkData<T, P>,
    ) -> PlonkProofResult<Self> {
        let zkey = &data.zkey;
        let public_input = &data.witness.shared_witness.public_inputs;
        let mut transcript = Keccak256Transcript::<P>::default();
        transcript.add_point(zkey.verifying_key.qm);
        transcript.add_point(zkey.verifying_key.ql);
        transcript.add_point(zkey.verifying_key.qr);
        transcript.add_point(zkey.verifying_key.qo);
        transcript.add_point(zkey.verifying_key.qc);
        transcript.add_point(zkey.verifying_key.s1);
        transcript.add_point(zkey.verifying_key.s2);
        transcript.add_point(zkey.verifying_key.s3);
        for val in public_input.iter().skip(1).cloned() {
            transcript.add_scalar(val);
        }
        transcript.add_point(proof.commit_a.into());
        transcript.add_point(proof.commit_b.into());
        transcript.add_point(proof.commit_c.into());

        let beta = transcript.get_challenge();

        let mut transcript = Keccak256Transcript::<P>::default();
        transcript.add_scalar(beta);
        let gamma = transcript.get_challenge();
        let challenges = Round2Challenges::new(challenges, beta, gamma);
        let z = Self::compute_z(driver, zkey, &domains, &challenges, &polys)?;
        // STEP 2.3 - Compute permutation [z]_1
        let commit_z = MSMProvider::<P::G1>::msm_public_points(driver, &zkey.p_tau, &z.poly);
        let proof = Round2Proof::new(proof, driver.open_point(&commit_z)?);

        Ok(Round::Round3 {
            domains,
            challenges,
            proof,
            polys: Round2Polys::new(polys, z),
            data,
        })
    }
}

#[cfg(test)]
pub mod tests {

    use std::{fs::File, io::BufReader};

    use ark_bn254::Bn254;
    use circom_types::{groth16::witness::Witness, plonk::ZKey};
    use collaborative_groth16::groth16::SharedWitness;
    use mpc_core::protocols::plain::PlainDriver;

    use crate::{Domains, PlonkData, Round};
    macro_rules! g1_from_xy {
        ($x: expr,$y: expr) => {
            <ark_bn254::Bn254 as Pairing>::G1Affine::new(
                ark_bn254::Fq::from_str($x).unwrap(),
                ark_bn254::Fq::from_str($y).unwrap(),
            )
        };
    }

    use super::Round1Challenges;
    use ark_ec::pairing::Pairing;
    use num_traits::Zero;
    use std::str::FromStr;
    #[test]
    fn test_round2_multiplier2() {
        let mut driver = PlainDriver::<ark_bn254::Fr>::default();
        let mut reader =
            BufReader::new(File::open("../test_vectors/Plonk/bn254/multiplier2.zkey").unwrap());
        let zkey = ZKey::<Bn254>::from_reader(&mut reader).unwrap();
        let witness_file = File::open("../test_vectors/Plonk/bn254/multiplier2_wtns.wtns").unwrap();
        let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file).unwrap();
        let witness = SharedWitness::<PlainDriver<ark_bn254::Fr>, Bn254> {
            public_inputs: vec![ark_bn254::Fr::zero(), witness.values[1]],
            witness: vec![witness.values[2], witness.values[3]],
        };

        let round1 = Round::<PlainDriver<ark_bn254::Fr>, Bn254>::Round1 {
            domains: Domains::new(&zkey).unwrap(),
            challenges: Round1Challenges::deterministic(&mut driver),
            data: PlonkData {
                witness: witness.into(),
                zkey,
            },
        };
        let round2 = round1.next_round(&mut driver).unwrap();
        if let Round::Round3 {
            domains,
            challenges,
            proof,
            polys,
            data,
        } = round2.next_round(&mut driver).unwrap()
        {
            assert_eq!(
                proof.commit_z,
                g1_from_xy!(
                    "3193853225338636777745825115445130632572036577821757614861727562557294229347",
                    "2273189363544172111301372917925262144551721107173868058666388265726379163785"
                )
            );
        } else {
            panic!("must be round2 after round1");
        }
    }
}
