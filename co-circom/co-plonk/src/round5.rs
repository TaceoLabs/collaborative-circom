use crate::{
    mpc::CircomPlonkProver,
    plonk_utils,
    round3::FinalPolys,
    round4::{Round4Challenges, Round4Proof},
    types::{Domains, Keccak256Transcript, PlonkData},
    PlonkProofResult,
};
use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use ark_ff::Field;
use circom_types::{
    plonk::PlonkProof,
    traits::{CircomArkworksPairingBridge, CircomArkworksPrimeFieldBridge},
};
use num_traits::One;
use num_traits::Zero;

// Round 5 of https://eprint.iacr.org/2019/953.pdf (page 30)
pub(super) struct Round5<'a, P: Pairing, T: CircomPlonkProver<P>> {
    pub(super) driver: T,
    pub(super) domains: Domains<P::ScalarField>,
    pub(super) challenges: Round4Challenges<P>,
    pub(super) proof: Round4Proof<P>,
    pub(super) polys: FinalPolys<P, T>,
    pub(super) data: PlonkData<'a, P, T>,
}
pub(super) struct Round5Challenges<P: Pairing> {
    beta: P::ScalarField,
    gamma: P::ScalarField,
    alpha: P::ScalarField,
    xi: P::ScalarField,
    v: [P::ScalarField; 5],
}

impl<P: Pairing> Round5Challenges<P> {
    fn new(round4_challenges: Round4Challenges<P>, v: [P::ScalarField; 5]) -> Self {
        Self {
            beta: round4_challenges.beta,
            gamma: round4_challenges.gamma,
            alpha: round4_challenges.alpha,
            xi: round4_challenges.xi,
            v,
        }
    }
}

impl<P: Pairing + CircomArkworksPairingBridge> Round4Proof<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    fn into_final_proof(self, commit_wxi: P::G1, commit_wxiw: P::G1) -> PlonkProof<P> {
        PlonkProof {
            a: P::G1Affine::from(self.commit_a),
            b: P::G1Affine::from(self.commit_b),
            c: P::G1Affine::from(self.commit_c),
            z: P::G1Affine::from(self.commit_z),
            t1: P::G1Affine::from(self.commit_t1),
            t2: P::G1Affine::from(self.commit_t2),
            t3: P::G1Affine::from(self.commit_t3),
            eval_a: self.eval_a,
            eval_b: self.eval_b,
            eval_c: self.eval_c,
            eval_s1: self.eval_s1,
            eval_s2: self.eval_s2,
            eval_zw: self.eval_zw,
            wxi: commit_wxi.into(),
            wxiw: P::G1Affine::from(commit_wxiw),
            protocol: "plonk".to_string(),
            curve: P::get_circom_name(),
        }
    }
}

// Round 5 of https://eprint.iacr.org/2019/953.pdf (page 30)
impl<'a, P: Pairing, T: CircomPlonkProver<P>> Round5<'a, P, T>
where
    P: CircomArkworksPairingBridge,
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    fn div_by_zerofier(inout: &mut Vec<T::ArithmeticShare>, n: usize, beta: P::ScalarField) {
        let inv_beta = beta.inverse().expect("Highly unlikely to be zero");
        let inv_beta_neg = -inv_beta;

        #[allow(unused_mut)]
        for mut el in inout.iter_mut().take(n) {
            *el = T::mul_with_public(*el, inv_beta_neg);
        }
        for i in n..inout.len() {
            let element = T::sub(inout[i - n], inout[i]);
            inout[i] = T::mul_with_public(element, inv_beta);
        }
        // We cannot check whether the polyonmial is divisible by the zerofier, but we resize accordingly
        inout.resize(inout.len() - n, T::ArithmeticShare::default());
    }

    fn add_poly(inout: &mut Vec<P::ScalarField>, add_poly: &[P::ScalarField]) {
        if add_poly.len() > inout.len() {
            inout.resize(add_poly.len(), P::ScalarField::zero());
        }

        #[allow(unused_mut)]
        for (mut inout, add) in inout.iter_mut().zip(add_poly.iter()) {
            *inout += *add;
        }
    }

    fn add_factor_poly(
        inout: &mut Vec<P::ScalarField>,
        add_poly: &[P::ScalarField],
        factor: P::ScalarField,
    ) {
        if add_poly.len() > inout.len() {
            inout.resize(add_poly.len(), P::ScalarField::zero());
        }

        #[allow(unused_mut)]
        for (mut inout, add) in inout.iter_mut().zip(add_poly.iter()) {
            *inout += *add * factor;
        }
    }

    // The linearisation polynomial R(X) (see https://eprint.iacr.org/2019/953.pdf)
    fn compute_r(
        party_id: T::PartyID,
        domains: &Domains<P::ScalarField>,
        proof: &Round4Proof<P>,
        challenges: &Round5Challenges<P>,
        data: &PlonkData<P, T>,
        polys: &FinalPolys<P, T>,
    ) -> Vec<T::ArithmeticShare> {
        tracing::debug!("computing r polynomial...");
        let zkey = &data.zkey;
        let public_inputs = &data.witness.public_inputs;
        let (l, xin) = plonk_utils::calculate_lagrange_evaluations::<P>(
            data.zkey.pow,
            data.zkey.n_public,
            &challenges.xi,
            domains,
        );
        let zh = xin - P::ScalarField::one();

        let l0 = &l[0];
        let eval_pi = plonk_utils::calculate_pi::<P>(public_inputs, &l);

        let coef_ab = proof.eval_a * proof.eval_b;
        let betaxi = challenges.beta * challenges.xi;
        let e2a = proof.eval_a + betaxi + challenges.gamma;
        let e2b = proof.eval_b + betaxi * zkey.verifying_key.k1 + challenges.gamma;
        let e2c = proof.eval_c + betaxi * zkey.verifying_key.k2 + challenges.gamma;
        let e2 = e2a * e2b * e2c * challenges.alpha;

        let e3a = proof.eval_a + challenges.beta * proof.eval_s1 + challenges.gamma;
        let e3b = proof.eval_b + challenges.beta * proof.eval_s2 + challenges.gamma;
        let e3 = e3a * e3b * proof.eval_zw * challenges.alpha;

        let e4 = challenges.alpha.square() * l0;
        let e24 = e2 + e4;

        let mut poly_r = zkey.qm_poly.coeffs.clone();
        #[allow(unused_mut)]
        for mut coeff in poly_r.iter_mut() {
            *coeff *= coef_ab;
        }
        Self::add_factor_poly(&mut poly_r.coeffs, &zkey.ql_poly.coeffs, proof.eval_a);
        Self::add_factor_poly(&mut poly_r.coeffs, &zkey.qr_poly.coeffs, proof.eval_b);
        Self::add_factor_poly(&mut poly_r.coeffs, &zkey.qo_poly.coeffs, proof.eval_c);
        Self::add_poly(&mut poly_r.coeffs, &zkey.qc_poly.coeffs);
        Self::add_factor_poly(
            &mut poly_r.coeffs,
            &zkey.s3_poly.coeffs,
            -(e3 * challenges.beta),
        );

        let len = zkey.domain_size + 6;

        let mut poly_r_shared = vec![T::ArithmeticShare::default(); len];

        #[allow(unused_mut)]
        for (mut inout, add) in poly_r_shared
            .iter_mut()
            .zip(polys.z.poly.clone().into_iter())
        {
            *inout = T::mul_with_public(add, e24);
        }

        #[allow(unused_mut)]
        for (mut inout, add) in poly_r_shared.iter_mut().zip(poly_r.iter()) {
            *inout = T::add_with_public(party_id, *inout, *add);
        }

        let mut tmp_poly = vec![T::ArithmeticShare::default(); len];
        let xin2 = xin.square();
        #[allow(unused_mut)]
        for (mut inout, add) in tmp_poly.iter_mut().zip(polys.t3.clone().into_iter()) {
            *inout = T::mul_with_public(add, xin2);
        }
        #[allow(unused_mut)]
        for (mut inout, add) in tmp_poly.iter_mut().zip(polys.t2.clone().into_iter()) {
            let tmp = T::mul_with_public(add, xin);
            *inout = T::add(*inout, tmp);
        }
        #[allow(unused_mut)]
        for (mut inout, add) in tmp_poly.iter_mut().zip(polys.t1.clone().into_iter()) {
            *inout = T::add(*inout, add);
        }
        #[allow(unused_mut)]
        for mut inout in tmp_poly.iter_mut() {
            *inout = T::mul_with_public(*inout, zh);
        }

        #[allow(unused_mut)]
        for (mut inout, sub) in poly_r_shared.iter_mut().zip(tmp_poly.iter()) {
            *inout = T::sub(*inout, *sub);
        }

        let r0 = eval_pi - (e3 * (proof.eval_c + challenges.gamma)) - e4;

        poly_r_shared[0] = T::add_with_public(party_id, poly_r_shared[0], r0);
        tracing::debug!("computing r polynomial done!");
        poly_r_shared
    }

    // The opening proof polynomial W_xi(X) (see https://eprint.iacr.org/2019/953.pdf)
    fn compute_wxi(
        party_id: T::PartyID,
        proof: &Round4Proof<P>,
        challenges: &Round5Challenges<P>,
        data: &PlonkData<P, T>,
        polys: &FinalPolys<P, T>,
        poly_r: &[T::ArithmeticShare],
    ) -> Vec<T::ArithmeticShare> {
        tracing::debug!("computing wxi polynomial...");
        let s1_poly_coeffs = &data.zkey.s1_poly.coeffs;
        let s2_poly_coeffs = &data.zkey.s2_poly.coeffs;
        let mut res = vec![T::ArithmeticShare::default(); data.zkey.domain_size + 6];

        // R
        for (inout, add) in res.iter_mut().zip(poly_r.iter()) {
            *inout = *add;
        }
        // A
        for (inout, add) in res.iter_mut().zip(polys.a.poly.clone().into_iter()) {
            let tmp = T::mul_with_public(add, challenges.v[0]);
            *inout = T::add(tmp, *inout);
        }
        // B
        for (inout, add) in res.iter_mut().zip(polys.b.poly.clone().into_iter()) {
            let tmp = T::mul_with_public(add, challenges.v[1]);
            *inout = T::add(tmp, *inout);
        }
        // C
        for (inout, add) in res.iter_mut().zip(polys.c.poly.clone().into_iter()) {
            let tmp = T::mul_with_public(add, challenges.v[2]);
            *inout = T::add(tmp, *inout);
        }
        // Sigma1
        for (inout, add) in res.iter_mut().zip(s1_poly_coeffs.iter()) {
            *inout = T::add_with_public(party_id, *inout, challenges.v[3] * add);
        }
        // Sigma2
        for (inout, add) in res.iter_mut().zip(s2_poly_coeffs.iter()) {
            *inout = T::add_with_public(party_id, *inout, challenges.v[4] * add);
        }

        res[0] = T::add_with_public(party_id, res[0], -challenges.v[0] * proof.eval_a);
        res[0] = T::add_with_public(party_id, res[0], -challenges.v[1] * proof.eval_b);
        res[0] = T::add_with_public(party_id, res[0], -challenges.v[2] * proof.eval_c);
        res[0] = T::add_with_public(party_id, res[0], -challenges.v[3] * proof.eval_s1);
        res[0] = T::add_with_public(party_id, res[0], -challenges.v[4] * proof.eval_s2);

        Self::div_by_zerofier(&mut res, 1, challenges.xi);

        tracing::debug!("computing wxi polynomial done!");
        res
    }

    // The opening proof polynomial W_xiw(X) (see https://eprint.iacr.org/2019/953.pdf)
    fn compute_wxiw(
        driver: &mut T,
        domains: &Domains<P::ScalarField>,
        proof: &Round4Proof<P>,
        challenges: &Round5Challenges<P>,
        polys: &FinalPolys<P, T>,
    ) -> Vec<T::ArithmeticShare> {
        tracing::debug!("computing wxiw polynomial...");
        let xiw = challenges.xi * domains.root_of_unity_pow;

        let mut res = polys.z.poly.clone().into_iter().collect::<Vec<_>>();
        res[0] = T::add_with_public(driver.get_party_id(), res[0], -proof.eval_zw);
        Self::div_by_zerofier(&mut res, 1, xiw);

        tracing::debug!("computing wxiw polynomial done!");
        res
    }

    // Round 5 of https://eprint.iacr.org/2019/953.pdf (page 30)
    pub(super) async fn round5(self) -> PlonkProofResult<PlonkProof<P>> {
        let Self {
            mut driver,
            domains,
            challenges,
            proof,
            polys,
            data,
        } = self;
        tracing::debug!("building challenges for round5 with Keccak256..");
        let mut transcript = Keccak256Transcript::<P>::default();
        // STEP 5.1 - Compute evaluation challenge v \in F_p
        transcript.add_scalar(challenges.xi);
        transcript.add_scalar(proof.eval_a);
        transcript.add_scalar(proof.eval_b);
        transcript.add_scalar(proof.eval_c);
        transcript.add_scalar(proof.eval_s1);
        transcript.add_scalar(proof.eval_s2);
        transcript.add_scalar(proof.eval_zw);

        let mut v = [P::ScalarField::one(); 5];
        v[0] = transcript.get_challenge();
        for i in 1..5 {
            v[i] = v[i - 1] * v[0];
        }
        tracing::debug!("v[0]: {}", v[0]);
        tracing::debug!("v[1]: {}", v[1]);
        tracing::debug!("v[2]: {}", v[2]);
        tracing::debug!("v[3]: {}", v[3]);
        tracing::debug!("v[4]: {}", v[4]);
        let challenges = Round5Challenges::new(challenges, v);
        let party_id = driver.get_party_id();

        // STEP 5.2 Compute linearisation polynomial r(X)
        let r = Self::compute_r(party_id, &domains, &proof, &challenges, &data, &polys);
        //STEP 5.3 Compute opening proof polynomial Wxi(X)
        let wxi = Self::compute_wxi(party_id, &proof, &challenges, &data, &polys, &r);

        //STEP 5.4 Compute opening proof polynomial Wxiw(X)
        let wxiw = Self::compute_wxiw(&mut driver, &domains, &proof, &challenges, &polys);
        // Fifth output of the prover is ([Wxi]_1, [Wxiw]_1)

        let p_tau = &data.zkey.p_tau;
        let commit_wxi = T::msm_public_points_g1(&p_tau[..wxi.len()], &wxi);
        let commit_wxiw = T::msm_public_points_g1(&p_tau[..wxiw.len()], &wxiw);

        let opened = driver.open_point_vec_g1(&[commit_wxi, commit_wxiw]).await?;

        let commit_wxi: P::G1 = opened[0];
        let commit_wxiw: P::G1 = opened[1];
        tracing::debug!(
            "Round5Proof(commit_wxi: {}, commit_wxiw: {})",
            commit_wxi.into_affine(),
            commit_wxiw.into_affine()
        );
        driver.close_network().await?;
        Ok(proof.into_final_proof(commit_wxi, commit_wxiw))
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
    macro_rules! g1_from_xy {
        ($x: expr,$y: expr) => {
            <ark_bn254::Bn254 as Pairing>::G1Affine::new(
                ark_bn254::Fq::from_str($x).unwrap(),
                ark_bn254::Fq::from_str($y).unwrap(),
            )
        };
    }

    use ark_ec::pairing::Pairing;
    use std::str::FromStr;
    #[tokio::test]
    async fn test_round5_multiplier2() {
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
        let proof = round5.round5().await.unwrap();
        assert_eq!(
            proof.wxi,
            g1_from_xy!(
                "17714933343167283383757911844657193439824158284537335005582807825912982308761",
                "10956622068891399683012461981563789956666325407769410657364052444385845871778"
            )
        );
        assert_eq!(
            proof.wxiw,
            g1_from_xy!(
                "11975595019949715918668172153793336705506375746143971491421022814159658028345",
                "21836122222240321064812409945656239690711148338716835775906941056446809090474"
            )
        );
    }
}
