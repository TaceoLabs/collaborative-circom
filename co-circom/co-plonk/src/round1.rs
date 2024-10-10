use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use circom_types::plonk::ZKey;
use co_circom_snarks::SharedWitness;
use tracing::instrument;

use crate::{
    mpc::CircomPlonkProver,
    plonk_utils::{self, rayon_join},
    round2::Round2,
    types::{Domains, PlonkData, PlonkWitness, PolyEval},
    PlonkProofError, PlonkProofResult,
};

// Round 1 of https://eprint.iacr.org/2019/953.pdf (page 28)
pub(super) struct Round1<'a, P: Pairing, T: CircomPlonkProver<P>> {
    pub(super) driver: T,
    pub(super) domains: Domains<P::ScalarField>,
    pub(super) challenges: Round1Challenges<P, T>,
    pub(super) data: PlonkDataRound1<'a, P, T>,
}

pub(super) struct PlonkDataRound1<'a, P: Pairing, T: CircomPlonkProver<P>> {
    witness: PlonkWitness<P, T>,
    zkey: &'a ZKey<P>,
}

impl<'a, P: Pairing, T: CircomPlonkProver<P>> From<PlonkDataRound1<'a, P, T>>
    for PlonkData<'a, P, T>
{
    fn from(mut data: PlonkDataRound1<'a, P, T>) -> Self {
        //when we are done, we remove the leading zero of the public inputs
        data.witness.public_inputs = data.witness.public_inputs[1..].to_vec();
        Self {
            witness: data.witness,
            zkey: data.zkey,
        }
    }
}

pub(super) struct Round1Challenges<P: Pairing, T: CircomPlonkProver<P>> {
    pub(super) b: [T::ArithmeticShare; 11],
}

pub(super) struct Round1Proof<P: Pairing> {
    pub(super) commit_a: P::G1,
    pub(super) commit_b: P::G1,
    pub(super) commit_c: P::G1,
}
pub(super) struct Round1Polys<P: Pairing, T: CircomPlonkProver<P>> {
    pub(super) buffer_a: Vec<T::ArithmeticShare>,
    pub(super) buffer_b: Vec<T::ArithmeticShare>,
    pub(super) buffer_c: Vec<T::ArithmeticShare>,
    pub(super) a: PolyEval<P, T>,
    pub(super) b: PolyEval<P, T>,
    pub(super) c: PolyEval<P, T>,
}

impl<P: Pairing> std::fmt::Display for Round1Proof<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(
            f,
            "Round1Proof(a: {}, b: {}, c: {})",
            self.commit_a.into_affine(),
            self.commit_b.into_affine(),
            self.commit_c.into_affine()
        )
    }
}

impl<P: Pairing, T: CircomPlonkProver<P>> Round1Challenges<P, T> {
    pub(super) async fn random(driver: &mut T) -> PlonkProofResult<Self> {
        let mut b = core::array::from_fn(|_| T::ArithmeticShare::default());
        #[allow(unused_mut)]
        for mut x in b.iter_mut() {
            *x = driver.rand()?;
        }
        Ok(Self { b })
    }

    #[cfg(test)]
    pub(super) fn deterministic(driver: &mut T) -> Self {
        let party_id = driver.get_party_id();
        Self {
            b: core::array::from_fn(|i| {
                T::promote_to_trivial_share(party_id, P::ScalarField::from(i as u64))
            }),
        }
    }
}

// Round 1 of https://eprint.iacr.org/2019/953.pdf (page 28)
#[allow(clippy::type_complexity)]
impl<'a, P: Pairing, T: CircomPlonkProver<P>> Round1<'a, P, T> {
    fn compute_single_wire_poly(
        party_id: T::PartyID,
        witness: &PlonkWitness<P, T>,
        domains: &Domains<P::ScalarField>,
        blind_factors: &[T::ArithmeticShare],
        zkey: &ZKey<P>,
        map: &[usize],
    ) -> PlonkProofResult<(Vec<T::ArithmeticShare>, PolyEval<P, T>)> {
        let mut buffer = Vec::with_capacity(zkey.n_constraints);
        debug_assert_eq!(zkey.n_constraints, map.len());
        for index in map {
            match plonk_utils::get_witness(party_id, witness, zkey, *index) {
                Ok(witness) => buffer.push(witness),
                Err(err) => return Err(err),
            }
        }
        buffer.resize(zkey.domain_size, T::ArithmeticShare::default());
        // Compute the coefficients of the wire polynomials a(X), b(X) and c(X) from A,B & C buffers
        let mut poly = T::ifft(&buffer, &domains.domain);

        tracing::debug!("ffts for evals..");
        // Compute extended evaluations of a(X), b(X) and c(X) polynomials
        let eval = T::fft(&poly, &domains.extended_domain);

        tracing::debug!("blinding coefficients");
        plonk_utils::blind_coefficients::<P, T>(&mut poly, blind_factors);
        Ok((buffer, PolyEval { poly, eval }))
    }

    // Essentially the fft of the trace columns
    #[instrument(level = "debug", name = "compute wire polys", skip_all)]
    fn compute_wire_polynomials(
        driver: &mut T,
        domains: &Domains<P::ScalarField>,
        challenges: &Round1Challenges<P, T>,
        zkey: &ZKey<P>,
        witness: &PlonkWitness<P, T>,
    ) -> PlonkProofResult<Round1Polys<P, T>> {
        let party_id = driver.get_party_id();
        let (wire_a, wire_b, wire_c) = rayon_join!(
            Self::compute_single_wire_poly(
                party_id,
                witness,
                domains,
                &challenges.b[..2],
                zkey,
                &zkey.map_a,
            ),
            Self::compute_single_wire_poly(
                party_id,
                witness,
                domains,
                &challenges.b[2..4],
                zkey,
                &zkey.map_b,
            ),
            Self::compute_single_wire_poly(
                party_id,
                witness,
                domains,
                &challenges.b[4..6],
                zkey,
                &zkey.map_c,
            )
        );
        let (buffer_a, poly_a) = wire_a?;
        let (buffer_b, poly_b) = wire_b?;
        let (buffer_c, poly_c) = wire_c?;

        if poly_a.poly.len() > zkey.domain_size + 2
            || poly_b.poly.len() > zkey.domain_size + 2
            || poly_c.poly.len() > zkey.domain_size + 2
        {
            return Err(PlonkProofError::PolynomialDegreeTooLarge);
        }
        tracing::debug!("computing wire polys done!");
        Ok(Round1Polys {
            buffer_a,
            buffer_b,
            buffer_c,
            a: poly_a,
            b: poly_b,
            c: poly_c,
        })
    }

    // Calculate the witnesses for the additions, since they are not part of the SharedWitness
    #[instrument(level = "debug", name = "calculate additions", skip_all)]
    fn calculate_additions(
        driver: &mut T,
        witness: SharedWitness<P::ScalarField, T::ArithmeticShare>,
        zkey: &ZKey<P>,
    ) -> PlonkProofResult<PlonkWitness<P, T>> {
        let party_id = driver.get_party_id();
        let mut witness = PlonkWitness::new(witness, zkey.n_additions);
        // This is hard to multithread as we have to add the results
        // to the vec as they are needed for the later steps.
        // We leave it like that as it does not take to much time (<1ms for poseidon).
        // Keep an eye on the span duration, maybe we have to come back to that later.
        for addition in zkey.additions.iter() {
            let witness1 = plonk_utils::get_witness(
                party_id,
                &witness,
                zkey,
                addition.signal_id1.try_into().expect("u32 fits into usize"),
            )?;
            let witness2 = plonk_utils::get_witness(
                party_id,
                &witness,
                zkey,
                addition.signal_id2.try_into().expect("u32 fits into usize"),
            )?;

            let f1 = T::mul_with_public(witness1, addition.factor1);
            let f2 = T::mul_with_public(witness2, addition.factor2);
            let result = T::add(f1, f2);
            witness.addition_witness.push(result);
        }
        Ok(witness)
    }

    #[instrument(level = "debug", name = "Plonk - Round Init", skip_all)]
    pub(super) async fn init_round(
        mut driver: T,
        zkey: &'a ZKey<P>,
        private_witness: SharedWitness<P::ScalarField, T::ArithmeticShare>,
    ) -> PlonkProofResult<Self> {
        let plonk_witness = Self::calculate_additions(&mut driver, private_witness, zkey)?;
        // TODO: we do not want that to be async
        let challenges = Round1Challenges::random(&mut driver).await?;
        let domains = Domains::new(zkey.domain_size)?;
        Ok(Self {
            challenges,
            driver,
            domains,
            data: PlonkDataRound1 {
                witness: plonk_witness,
                zkey,
            },
        })
    }

    #[instrument(level = "debug", name = "Plonk - Round 1", skip_all)]
    // Round 1 of https://eprint.iacr.org/2019/953.pdf (page 28)
    pub(super) async fn round1(self) -> PlonkProofResult<Round2<'a, P, T>> {
        let Self {
            mut driver,
            domains,
            challenges,
            data,
        } = self;
        let witness = &data.witness;
        let zkey = &data.zkey;
        let p_tau = &zkey.p_tau;

        // STEP 1.2 - Compute wire polynomials a(X), b(X) and c(X)
        let polys =
            Self::compute_wire_polynomials(&mut driver, &domains, &challenges, zkey, witness)?;

        let commit_span = tracing::debug_span!("committing to polys (MSMs)").entered();
        // STEP 1.3 - Compute [a]_1, [b]_1, [c]_1
        let (commit_a, commit_b, commit_c) = rayon_join!(
            T::msm_public_points_g1(&p_tau[..polys.a.poly.len()], &polys.a.poly),
            T::msm_public_points_g1(&p_tau[..polys.b.poly.len()], &polys.b.poly),
            T::msm_public_points_g1(&p_tau[..polys.c.poly.len()], &polys.c.poly)
        );

        // network round
        commit_span.exit();
        let opening_span = tracing::debug_span!("opening commits").entered();
        let opened = driver
            .open_point_vec_g1(&[commit_a, commit_b, commit_c])
            .await?;
        opening_span.exit();
        let proof = Round1Proof::<P> {
            commit_a: opened[0],
            commit_b: opened[1],
            commit_c: opened[2],
        };
        tracing::debug!("round1 result: {proof}");
        Ok(Round2 {
            driver,
            domains,
            challenges,
            proof,
            polys,
            data: data.into(),
        })
    }
}

#[cfg(test)]
pub mod tests {
    use std::{fs::File, io::BufReader};

    use ark_bls12_381::Bls12_381;
    use ark_bn254::Bn254;
    use circom_types::plonk::ZKey;
    use co_circom_snarks::SharedWitness;

    use crate::mpc::plain::PlainPlonkDriver;

    use super::{Round1, Round1Challenges};
    use ark_ec::{pairing::Pairing, CurveGroup};
    use circom_types::Witness;
    use std::str::FromStr;

    macro_rules! g1_bn254_from_xy {
        ($x: expr,$y: expr) => {
            <ark_bn254::Bn254 as Pairing>::G1Affine::new(
                ark_bn254::Fq::from_str($x).unwrap(),
                ark_bn254::Fq::from_str($y).unwrap(),
            )
        };
    }

    macro_rules! g1_bls12_381_from_xy {
        ($x: expr,$y: expr) => {
            <ark_bls12_381::Bls12_381 as Pairing>::G1Affine::new(
                ark_bls12_381::Fq::from_str($x).unwrap(),
                ark_bls12_381::Fq::from_str($y).unwrap(),
            )
        };
    }

    #[tokio::test]
    async fn test_round1_multiplier2() {
        let mut driver = PlainPlonkDriver;
        let mut reader = BufReader::new(
            File::open("../../test_vectors/Plonk/bn254/multiplier2/circuit.zkey").unwrap(),
        );
        let zkey = ZKey::<Bn254>::from_reader(&mut reader).unwrap();
        let witness_file =
            File::open("../../test_vectors/Plonk/bn254/multiplier2/witness.wtns").unwrap();
        let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file).unwrap();
        let witness = SharedWitness {
            public_inputs: witness.values[..=zkey.n_public].to_vec(),
            witness: witness.values[zkey.n_public + 1..].to_vec(),
        };
        let challenges = Round1Challenges::deterministic(&mut driver);
        let mut round1 = Round1::init_round(driver, &zkey, witness).await.unwrap();
        round1.challenges = challenges;
        let round2 = round1.round1().await.unwrap();
        assert_eq!(
            round2.proof.commit_a,
            g1_bn254_from_xy!(
                "17605081043163307645214588229802469503664729145403357283635330564965670333858",
                "6586266374304386912414685272642968153787280144323447197846781700256409557611"
            )
        );
        assert_eq!(
            round2.proof.commit_b,
            g1_bn254_from_xy!(
                "5630355441221157622116381279042400483431873694148526624610332736752309357481",
                "459435968793897134848228876468434334542717512356212242962101833939899171644"
            )
        );
        assert_eq!(
            round2.proof.commit_c,
            g1_bn254_from_xy!(
                "15206827023183180947877311390140741127921188782225553575654415094642569639438",
                "14970166502897037710457760872123795383312785044242798403684409588772714154874"
            )
        );
    }

    #[tokio::test]
    async fn test_round1_poseidon_bls12_381() {
        let mut driver = PlainPlonkDriver;
        let mut reader = BufReader::new(
            File::open("../../test_vectors/Plonk/bls12_381/poseidon/circuit.zkey").unwrap(),
        );
        let zkey = ZKey::<Bls12_381>::from_reader(&mut reader).unwrap();
        let witness_file =
            File::open("../../test_vectors/Plonk/bls12_381/poseidon/witness.wtns").unwrap();
        let witness = Witness::<ark_bls12_381::Fr>::from_reader(witness_file).unwrap();

        let public_input = witness.values[..=zkey.n_public].to_vec();
        let witness = SharedWitness {
            public_inputs: public_input.clone(),
            witness: witness.values[zkey.n_public + 1..].to_vec(),
        };

        let challenges = Round1Challenges::deterministic(&mut driver);
        let mut round1 = Round1::init_round(driver, &zkey, witness).await.unwrap();
        round1.challenges = challenges;
        let round2 = round1.round1().await.unwrap();
        assert_eq!(
            round2.proof.commit_a.into_affine(),
            g1_bls12_381_from_xy!(
                "1998528185362278337803945478659945086542519630073413629642105010067028189206141975508238821825915421715338325238864",
                "436066057394619309469331627881449668678557518497178283348448576242129245895320288313540996356612092203769711134939"
            )
        );
        assert_eq!(
            round2.proof.commit_b,
            g1_bls12_381_from_xy!(
                "905523078516729029387874920505888326057985585766807058529621596028494573503715980387105934346404133401227192848784",
                "817813208457279034981972137280354075285704598923875006670861630006742541882069169563142367502699866422101983374962"
            )
        );
        assert_eq!(
            round2.proof.commit_c,
            g1_bls12_381_from_xy!(
                "2045702311111033155343546707999313330868835292331631548140598745513449880984849831136790392158415943067742290277175",
                "2263708941732971465915801396733005622347769540424301431567098497278413189155761949973582649025461644335372679621757"
            )
        );
    }
}
