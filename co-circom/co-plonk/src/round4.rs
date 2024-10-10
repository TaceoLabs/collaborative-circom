use crate::{
    mpc::CircomPlonkProver,
    round3::{FinalPolys, Round3Challenges, Round3Proof},
    round5::Round5,
    types::{Domains, Keccak256Transcript, PlonkData},
    PlonkProofResult,
};
use ark_ec::pairing::Pairing;

// Round 4 of https://eprint.iacr.org/2019/953.pdf (page 29)
pub(super) struct Round4<'a, P: Pairing, T: CircomPlonkProver<P>> {
    pub(super) driver: T,
    pub(super) domains: Domains<P::ScalarField>,
    pub(super) challenges: Round3Challenges<P, T>,
    pub(super) proof: Round3Proof<P>,
    pub(super) polys: FinalPolys<P, T>,
    pub(super) data: PlonkData<'a, P, T>,
}
pub(super) struct Round4Challenges<P: Pairing> {
    pub(super) beta: P::ScalarField,
    pub(super) gamma: P::ScalarField,
    pub(super) alpha: P::ScalarField,
    pub(super) xi: P::ScalarField,
}
impl<P: Pairing> Round4Challenges<P> {
    fn new<T>(round3_challenges: Round3Challenges<P, T>, xi: P::ScalarField) -> Self
    where
        T: CircomPlonkProver<P>,
    {
        Self {
            beta: round3_challenges.beta,
            gamma: round3_challenges.gamma,
            alpha: round3_challenges.alpha,
            xi,
        }
    }
}

pub(super) struct Round4Proof<P: Pairing> {
    pub(super) commit_a: P::G1,
    pub(super) commit_b: P::G1,
    pub(super) commit_c: P::G1,
    pub(super) commit_z: P::G1,
    pub(super) commit_t1: P::G1,
    pub(super) commit_t2: P::G1,
    pub(super) commit_t3: P::G1,
    pub(super) eval_a: P::ScalarField,
    pub(super) eval_b: P::ScalarField,
    pub(super) eval_c: P::ScalarField,
    pub(super) eval_zw: P::ScalarField,
    pub(super) eval_s1: P::ScalarField,
    pub(super) eval_s2: P::ScalarField,
}

impl<P: Pairing> std::fmt::Display for Round4Proof<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(
            f,
            "Round4Proof(eval_a: {}, eval_b: {}, eval_c: {}, eval_s1: {}, eval_s2: {}, eval_zw: {})",
            self.eval_a,
            self.eval_b,
            self.eval_c,
            self.eval_s1,
            self.eval_s2,
            self.eval_zw,
        )
    }
}

impl<P: Pairing> Round4Proof<P> {
    fn new(
        round3_proof: Round3Proof<P>,
        eval_a: P::ScalarField,
        eval_b: P::ScalarField,
        eval_c: P::ScalarField,
        eval_zw: P::ScalarField,
        eval_s1: P::ScalarField,
        eval_s2: P::ScalarField,
    ) -> Self {
        Self {
            commit_a: round3_proof.commit_a,
            commit_b: round3_proof.commit_b,
            commit_c: round3_proof.commit_c,
            commit_z: round3_proof.commit_z,
            commit_t1: round3_proof.commit_t1,
            commit_t2: round3_proof.commit_t2,
            commit_t3: round3_proof.commit_t3,
            eval_a,
            eval_b,
            eval_c,
            eval_zw,
            eval_s1,
            eval_s2,
        }
    }
}

// Round 4 of https://eprint.iacr.org/2019/953.pdf (page 29)
impl<'a, P: Pairing, T: CircomPlonkProver<P>> Round4<'a, P, T> {
    // Round 4 of https://eprint.iacr.org/2019/953.pdf (page 29)
    pub(super) async fn round4(self) -> PlonkProofResult<Round5<'a, P, T>> {
        let Self {
            mut driver,
            domains,
            challenges,
            proof,
            mut polys,
            data,
        } = self;
        tracing::debug!("building challenges for round4 with Keccak256..");
        // STEP 4.1 - Compute evaluation challenge xi \in F_p
        let mut transcript = Keccak256Transcript::<P>::default();
        transcript.add_scalar(challenges.alpha);
        transcript.add_point(proof.commit_t1.into());
        transcript.add_point(proof.commit_t2.into());
        transcript.add_point(proof.commit_t3.into());
        let xi = transcript.get_challenge();
        let xiw = xi * domains.root_of_unity_pow;
        let challenges = Round4Challenges::new(challenges, xi);
        tracing::debug!("xi: {xi}");
        tracing::debug!("evaluating poly a");
        let poly_a = std::mem::take(&mut polys.a.poly);
        let poly_b = std::mem::take(&mut polys.b.poly);
        let poly_c = std::mem::take(&mut polys.c.poly);
        let poly_z = std::mem::take(&mut polys.z.poly);
        let (eval_a, poly_a) = T::evaluate_poly_public(poly_a, challenges.xi);
        tracing::debug!("evaluating poly b");
        let (eval_b, poly_b) = T::evaluate_poly_public(poly_b, challenges.xi);
        tracing::debug!("evaluating poly c");
        let (eval_c, poly_c) = T::evaluate_poly_public(poly_c, challenges.xi);
        tracing::debug!("evaluating poly z");
        let (eval_z, poly_z) = T::evaluate_poly_public(poly_z, xiw);
        polys.a.poly = poly_a;
        polys.b.poly = poly_b;
        polys.c.poly = poly_c;
        polys.z.poly = poly_z;

        let opened = driver.open_vec(&[eval_a, eval_b, eval_c, eval_z]).await?;

        let eval_a = opened[0];
        let eval_b = opened[1];
        let eval_c = opened[2];
        let eval_zw = opened[3];

        let eval_s1 = data.zkey.s1_poly.evaluate(&challenges.xi);
        let eval_s2 = data.zkey.s2_poly.evaluate(&challenges.xi);
        let proof = Round4Proof::new(proof, eval_a, eval_b, eval_c, eval_zw, eval_s1, eval_s2);
        tracing::debug!("round4 result: {proof}");

        Ok(Round5 {
            driver,
            domains,
            challenges,
            proof,
            polys,
            data,
        })
    }
}

#[cfg(test)]
pub mod tests {

    use std::{fs::File, io::BufReader};

    use ark_bn254::Bn254;
    use circom_types::plonk::ZKey;
    use circom_types::Witness;
    use co_circom_snarks::SharedWitness;

    use crate::{
        mpc::plain::PlainPlonkDriver,
        round1::{Round1, Round1Challenges},
    };

    use std::str::FromStr;
    #[tokio::test]
    async fn test_round4_multiplier2() {
        let mut driver = PlainPlonkDriver;
        let mut reader = BufReader::new(
            File::open("../../test_vectors/Plonk/bn254/multiplier2/circuit.zkey").unwrap(),
        );
        let zkey = ZKey::<Bn254>::from_reader(&mut reader).unwrap();
        let witness_file =
            File::open("../../test_vectors/Plonk/bn254/multiplier2/witness.wtns").unwrap();
        let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file).unwrap();
        let public_input = witness.values[..=zkey.n_public].to_vec();
        let witness = SharedWitness {
            public_inputs: public_input.clone(),
            witness: witness.values[zkey.n_public + 1..].to_vec(),
        };

        let challenges = Round1Challenges::deterministic(&mut driver);
        let mut round1 = Round1::init_round(driver, &zkey, witness).await.unwrap();
        round1.challenges = challenges;
        let round2 = round1.round1().await.unwrap();
        let round3 = round2.round2().await.unwrap();
        let round4 = round3.round3().await.unwrap();
        let round5 = round4.round4().await.unwrap();
        assert_eq!(
            round5.proof.eval_a,
            ark_bn254::Fr::from_str(
                "9577617118727487156038114503197927927393325100881782676071854181913228129519"
            )
            .unwrap()
        );
        assert_eq!(
            round5.proof.eval_b,
            ark_bn254::Fr::from_str(
                "20597878711220885145139457487405665380092038394343281979206937623212519986448"
            )
            .unwrap()
        );
        assert_eq!(
            round5.proof.eval_c,
            ark_bn254::Fr::from_str(
                "15265494263612694384441473331344570152140354050926476508657731330784430744915"
            )
            .unwrap()
        );
        assert_eq!(
            round5.proof.eval_zw,
            ark_bn254::Fr::from_str(
                "13208748067365350181326696119359571057028048827339239951085850234164749233153"
            )
            .unwrap()
        );
        assert_eq!(
            round5.proof.eval_s1,
            ark_bn254::Fr::from_str(
                "14333100636430622287126878289812189552775054994479690945797668457655414216377"
            )
            .unwrap()
        );
        assert_eq!(
            round5.proof.eval_s2,
            ark_bn254::Fr::from_str(
                "5227675743165392606371559215386333900775466821923985579976650047914227054429"
            )
            .unwrap()
        );
    }
}
