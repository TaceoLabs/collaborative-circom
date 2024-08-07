use crate::{
    round1::{Round1Challenges, Round1Proof},
    round2::{Round2Challenges, Round2Polys, Round2Proof},
    types::{Keccak256Transcript, PolyEval},
    Domains, FieldShareVec, PlonkData, PlonkProofError, PlonkProofResult, Round,
};
use ark_ec::pairing::Pairing;
use ark_ff::Field;
use ark_poly::GeneralEvaluationDomain;
use circom_types::{
    groth16::{public_input, zkey},
    plonk::ZKey,
};
use collaborative_groth16::groth16::CollaborativeGroth16;
use mpc_core::traits::EcMpcProtocol;
use mpc_core::traits::{
    FFTPostProcessing, FFTProvider, MSMProvider, MontgomeryField, MpcToMontgomery,
    PairingEcMpcProtocol, PrimeFieldMpcProtocol,
};
use num_traits::One;
use num_traits::Zero;

// TODO parallelize these?
macro_rules! mul4vec {
    ($driver: expr, $a: expr,$b: expr,$c: expr,$d: expr,$ap: expr,$bp: expr,$cp: expr,$dp: expr, $domain: expr) => {{
        let a_b = $driver.mul_vec($a, $b)?;
        let a_bp = $driver.mul_vec($a, $bp)?;
        let ap_b = $driver.mul_vec($ap, $b)?;
        let ap_bp = $driver.mul_vec($ap, $bp)?;

        let c_d = $driver.mul_vec($c, $d)?;
        let c_dp = $driver.mul_vec($c, $dp)?;
        let cp_d = $driver.mul_vec($cp, $d)?;
        let cp_dp = $driver.mul_vec($cp, $dp)?;

        let r = $driver.mul_vec(&a_b, &c_d)?;

        let mut a0 = $driver.mul_vec(&ap_b, &c_d)?;
        a0 = $driver.add_mul_vec(&a0, &a_bp, &c_d)?;
        a0 = $driver.add_mul_vec(&a0, &a_b, &cp_d)?;
        a0 = $driver.add_mul_vec(&a0, &a_b, &c_dp)?;

        let mut a1 = $driver.mul_vec(&ap_bp, &c_d)?;
        a1 = $driver.add_mul_vec(&a1, &ap_b, &cp_d)?;
        a1 = $driver.add_mul_vec(&a1, &ap_b, &c_dp)?;
        a1 = $driver.add_mul_vec(&a1, &a_bp, &cp_d)?;
        a1 = $driver.add_mul_vec(&a1, &a_bp, &c_dp)?;
        a1 = $driver.add_mul_vec(&a1, &a_b, &cp_dp)?;

        let mut a2 = $driver.mul_vec(&a_bp, &cp_dp)?;
        a2 = $driver.add_mul_vec(&a2, &ap_b, &cp_dp)?;
        a2 = $driver.add_mul_vec(&a2, &ap_bp, &c_dp)?;
        a2 = $driver.add_mul_vec(&a2, &ap_bp, &cp_d)?;

        let a3 = $driver.mul_vec(&ap_bp, &cp_dp)?;
        [r, a0, a1, a2, a3]
    }};
}

macro_rules! mul4vec_post {
    ($driver: expr, $a: expr,$b: expr,$c: expr,$d: expr,$i: expr, $z1: expr, $z2: expr, $z3: expr) => {{
        let mod_i = $i % 4;
        let mut rz = T::index_sharevec(&$a, $i);
        if mod_i != 0 {
            let b = T::index_sharevec(&$b, $i);
            let c = T::index_sharevec(&$c, $i);
            let d = T::index_sharevec(&$d, $i);
            let tmp = $driver.mul_with_public(&$z1[mod_i], &b);
            rz = $driver.add(&rz, &tmp);
            let tmp = $driver.mul_with_public(&$z2[mod_i], &c);
            rz = $driver.add(&rz, &tmp);
            let tmp = $driver.mul_with_public(&$z3[mod_i], &d);
            rz = $driver.add(&rz, &tmp);
        }
        rz
    }};
}

pub(super) struct Round3Proof<P: Pairing> {
    pub(crate) commit_a: P::G1,
    pub(crate) commit_b: P::G1,
    pub(crate) commit_c: P::G1,
    pub(crate) commit_z: P::G1,
    pub(crate) commit_t1: P::G1,
    pub(crate) commit_t2: P::G1,
    pub(crate) commit_t3: P::G1,
}

impl<P: Pairing> Round3Proof<P> {
    fn new(
        round2_proof: Round2Proof<P>,
        commit_t1: P::G1,
        commit_t2: P::G1,
        commit_t3: P::G1,
    ) -> Self {
        Self {
            commit_a: round2_proof.commit_a,
            commit_b: round2_proof.commit_b,
            commit_c: round2_proof.commit_c,
            commit_z: round2_proof.commit_z,
            commit_t1,
            commit_t2,
            commit_t3,
        }
    }
}
pub(super) struct Round3Challenges<T, P: Pairing>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>,
{
    pub(crate) b: [T::FieldShare; 11],
    pub(crate) beta: P::ScalarField,
    pub(crate) gamma: P::ScalarField,
    pub(crate) alpha: P::ScalarField,
    pub(crate) alpha2: P::ScalarField,
}

pub(super) struct Round3Polys<T, P: Pairing>
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
    pub(crate) t1: FieldShareVec<T, P>,
    pub(crate) t2: FieldShareVec<T, P>,
    pub(crate) t3: FieldShareVec<T, P>,
}
impl<T, P: Pairing> Round3Polys<T, P>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>,
{
    fn new(
        polys: Round2Polys<T, P>,
        t1: FieldShareVec<T, P>,
        t2: FieldShareVec<T, P>,
        t3: FieldShareVec<T, P>,
    ) -> Self {
        Self {
            buffer_a: polys.buffer_a,
            buffer_b: polys.buffer_b,
            buffer_c: polys.buffer_c,
            poly_eval_a: polys.poly_eval_a,
            poly_eval_b: polys.poly_eval_b,
            poly_eval_c: polys.poly_eval_c,
            z: polys.z,
            t1,
            t2,
            t3,
        }
    }
}

impl<T, P: Pairing> Round3Challenges<T, P>
where
    for<'a> T: PrimeFieldMpcProtocol<P::ScalarField>,
{
    fn new(
        round2_challenges: Round2Challenges<T, P>,
        alpha: P::ScalarField,
        alpha2: P::ScalarField,
    ) -> Self {
        Self {
            b: round2_challenges.b,
            beta: round2_challenges.beta,
            gamma: round2_challenges.gamma,
            alpha,
            alpha2,
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
    fn get_z1(domains: &Domains<P>) -> [P::ScalarField; 4] {
        //TODO MOVE THIS THIS MUST BE A CONSTANT
        let zero = P::ScalarField::zero();
        let neg_1 = zero - P::ScalarField::one();
        let neg_2 = neg_1 - P::ScalarField::one();
        let root_of_unity = domains.roots_of_unity[2];
        [zero, neg_1 + root_of_unity, neg_2, neg_1 - root_of_unity]
    }

    fn get_z2(domains: &Domains<P>) -> [P::ScalarField; 4] {
        let zero = P::ScalarField::zero();
        let two = P::ScalarField::one() + P::ScalarField::one();
        let four = two.square();
        let neg_2 = zero - two;
        let root_of_unity = domains.roots_of_unity[2];
        let neg2_root_unity = neg_2 * root_of_unity;
        [
            zero,
            neg2_root_unity,
            four,
            P::ScalarField::zero() - neg2_root_unity,
        ]
    }

    fn get_z3(domains: &Domains<P>) -> [P::ScalarField; 4] {
        let zero = P::ScalarField::zero();
        let two = P::ScalarField::one() + P::ScalarField::one();
        let neg_eight = -(two.square() * two);
        let root_of_unity = domains.roots_of_unity[2];
        let two_root_unity = two * root_of_unity;
        [zero, two + two_root_unity, neg_eight, two - two_root_unity]
    }
    fn compute_t(
        driver: &mut T,
        domains: &Domains<P>,
        challenges: &Round3Challenges<T, P>,
        zkey: &ZKey<P>,
        polys: &Round2Polys<T, P>,
    ) -> PlonkProofResult<[FieldShareVec<T, P>; 3]> {
        let z1 = Self::get_z1(domains);
        let z2 = Self::get_z2(domains);
        let z3 = Self::get_z3(domains);
        let mut w = P::ScalarField::one();
        let mut ap = Vec::with_capacity(zkey.domain_size * 4);
        let mut bp = Vec::with_capacity(zkey.domain_size * 4);
        let mut cp = Vec::with_capacity(zkey.domain_size * 4);

        let pow_root_of_unity = domains.roots_of_unity[zkey.power];
        let pow_plus2_root_of_unity = domains.roots_of_unity[zkey.power + 2];
        for _ in 0..zkey.domain_size * 4 {
            //TODO use add_mul convenience method
            let ap_ = driver.mul_with_public(&w, &challenges.b[0]);
            let ap_ = driver.add(&challenges.b[1], &ap_);
            ap.push(ap_);

            let bp_ = driver.mul_with_public(&w, &challenges.b[2]);
            let bp_ = driver.add(&challenges.b[3], &bp_);
            bp.push(bp_);

            let cp_ = driver.mul_with_public(&w, &challenges.b[4]);
            let cp_ = driver.add(&challenges.b[5], &cp_);
            cp.push(cp_);

            w *= &pow_plus2_root_of_unity;
        }

        let ap_vec = ap.into();
        let bp_vec = bp.into();
        let cp_vec = cp.into();

        // TODO parallelize these?
        let a_b = driver.mul_vec(&polys.poly_eval_a.eval, &polys.poly_eval_b.eval)?;
        let a_bp = driver.mul_vec(&polys.poly_eval_a.eval, &bp_vec)?;
        let ap_b = driver.mul_vec(&polys.poly_eval_b.eval, &ap_vec)?;
        let ap_bp = driver.mul_vec(&ap_vec, &bp_vec)?;

        let mut e1 = Vec::with_capacity(zkey.domain_size * 4);
        let mut e1z = Vec::with_capacity(zkey.domain_size * 4);

        let mut e2a = Vec::with_capacity(zkey.domain_size * 4);
        let mut e2b = Vec::with_capacity(zkey.domain_size * 4);
        let mut e2c = Vec::with_capacity(zkey.domain_size * 4);
        let mut e2d = Vec::with_capacity(zkey.domain_size * 4);
        let mut zp = Vec::with_capacity(zkey.domain_size * 4);

        let mut e3a = Vec::with_capacity(zkey.domain_size * 4);
        let mut e3b = Vec::with_capacity(zkey.domain_size * 4);
        let mut e3c = Vec::with_capacity(zkey.domain_size * 4);
        let mut e3d = Vec::with_capacity(zkey.domain_size * 4);
        let mut zwp = Vec::with_capacity(zkey.domain_size * 4);
        let mut w = P::ScalarField::one();
        for i in 0..zkey.domain_size * 4 {
            let a = T::index_sharevec(&polys.poly_eval_a.eval, i);
            let b = T::index_sharevec(&polys.poly_eval_b.eval, i);
            let c = T::index_sharevec(&polys.poly_eval_c.eval, i);
            let z = T::index_sharevec(&polys.z.eval, i);
            let qm = zkey.qm_poly.evaluations[i];
            let ql = zkey.ql_poly.evaluations[i];
            let qr = zkey.qr_poly.evaluations[i];
            let qo = zkey.qo_poly.evaluations[i];
            let qc = zkey.qc_poly.evaluations[i];
            let s1 = zkey.s1_poly.evaluations[i];
            let s2 = zkey.s2_poly.evaluations[i];
            let s3 = zkey.s3_poly.evaluations[i];
            let a_bp = T::index_sharevec(&a_bp, i);
            let a_b = T::index_sharevec(&a_b, i);
            let ap_b = T::index_sharevec(&ap_b, i);
            let ap = T::index_sharevec(&ap_vec, i);
            let bp = T::index_sharevec(&bp_vec, i);

            let w2 = w.square();
            let zp_lhs = driver.mul_with_public(&w2, &challenges.b[6]);
            let zp_rhs = driver.mul_with_public(&w, &challenges.b[7]);
            let zp_ = driver.add(&zp_lhs, &zp_rhs);
            let zp_ = driver.add(&challenges.b[8], &zp_);
            zp.push(zp_);

            let w_w = w * pow_root_of_unity;
            let w_w2 = w_w.square();
            let zw = T::index_sharevec(
                &polys.z.eval,
                (zkey.domain_size * 4 + 4 + i) % (zkey.domain_size * 4),
            );
            let zwp_lhs = driver.mul_with_public(&w_w2, &challenges.b[6]);
            let zwp_rhs = driver.mul_with_public(&w_w, &challenges.b[7]);
            let zwp_ = driver.add(&zwp_lhs, &zwp_rhs);
            let zwp_ = driver.add(&challenges.b[8], &zwp_);
            zwp.push(zwp_);

            let mut a0 = driver.add(&a_bp, &ap_b);
            let mod_i = i % 4;
            if mod_i != 0 {
                let z1 = z1[mod_i];
                let ap_bp = T::index_sharevec(&ap_bp, i);
                let tmp = driver.mul_with_public(&z1, &ap_bp);
                a0 = driver.add(&a0, &tmp);
            }

            let (mut e1_, mut e1z_) = (a_b, a0);
            e1_ = driver.mul_with_public(&qm, &e1_);
            e1z_ = driver.mul_with_public(&qm, &e1z_);

            e1_ = driver.add_mul_public(&e1_, &a, &ql);
            e1z_ = driver.add_mul_public(&e1z_, &ap, &ql);

            e1_ = driver.add_mul_public(&e1_, &b, &qr);
            e1z_ = driver.add_mul_public(&e1z_, &bp, &qr);

            e1_ = driver.add_mul_public(&e1_, &c, &qo);
            e1z_ = driver.add_mul_public(&e1z_, &T::index_sharevec(&cp_vec, i), &qo);

            let mut pi = T::zero_share();
            for (j, lagrange) in zkey.lagrange.iter().enumerate() {
                let l_eval = lagrange.evaluations[i];
                let a_val = T::index_sharevec(&polys.buffer_a, j);
                let tmp = driver.mul_with_public(&l_eval, &a_val);
                pi = driver.sub(&pi, &tmp);
            }

            e1_ = driver.add(&e1_, &pi);
            e1_ = driver.add_with_public(&qc, &e1_);
            e1.push(e1_);
            e1z.push(e1z_);

            let betaw = challenges.beta * w;
            let mut e2a_ = a.clone();
            e2a_ = driver.add_with_public(&betaw, &e2a_);
            e2a_ = driver.add_with_public(&challenges.gamma, &e2a_);
            e2a.push(e2a_);

            let mut e2b_ = b.clone();
            e2b_ = driver.add_with_public(&(betaw * zkey.verifying_key.k1), &e2b_);
            e2b_ = driver.add_with_public(&challenges.gamma, &e2b_);
            e2b.push(e2b_);

            let mut e2c_ = c.clone();
            e2c_ = driver.add_with_public(&(betaw * zkey.verifying_key.k2), &e2c_);
            e2c_ = driver.add_with_public(&challenges.gamma, &e2c_);
            e2c.push(e2c_);

            let e2d_ = z;
            e2d.push(e2d_);

            let mut e3a_ = a;
            e3a_ = driver.add_with_public(&(s1 * challenges.beta), &e3a_);
            e3a_ = driver.add_with_public(&challenges.gamma, &e3a_);
            e3a.push(e3a_);

            let mut e3b_ = b;
            e3b_ = driver.add_with_public(&(s2 * challenges.beta), &e3b_);
            e3b_ = driver.add_with_public(&challenges.gamma, &e3b_);
            e3b.push(e3b_);

            let mut e3c_ = c;
            e3c_ = driver.add_with_public(&(s3 * challenges.beta), &e3c_);
            e3c_ = driver.add_with_public(&challenges.gamma, &e3c_);
            e3c.push(e3c_);

            let e3d_ = zw;
            e3d.push(e3d_);
            w *= pow_plus2_root_of_unity;
        }

        let e2a_vec = e2a.into();
        let e2b_vec = e2b.into();
        let e2c_vec = e2c.into();
        let e2d_vec = e2d.into();
        let zp_vec = zp.into();

        let [e2, e2z_0, e2z_1, e2z_2, e2z_3] = mul4vec!(
            driver, &e2a_vec, &e2b_vec, &e2c_vec, &e2d_vec, &ap_vec, &bp_vec, &cp_vec, &zp_vec,
            &domain1
        );

        let e3a_vec = e3a.into();
        let e3b_vec = e3b.into();
        let e3c_vec = e3c.into();
        let e3d_vec = e3d.into();
        let zwp_vec = zwp.into();

        let [e3, e3z_0, e3z_1, e3z_2, e3z_3] = mul4vec!(
            driver, &e3a_vec, &e3b_vec, &e3c_vec, &e3d_vec, &ap_vec, &bp_vec, &cp_vec, &zwp_vec,
            &domain1
        );

        let mut t_vec = Vec::with_capacity(zkey.domain_size * 4);
        let mut tz_vec = Vec::with_capacity(zkey.domain_size * 4);
        for i in 0..zkey.domain_size * 4 {
            let mut e2 = T::index_sharevec(&e2, i);
            let mut e2z = mul4vec_post!(driver, e2z_0, e2z_1, e2z_2, e2z_3, i, z1, z2, z3);
            let mut e3 = T::index_sharevec(&e3, i);
            let mut e3z = mul4vec_post!(driver, e3z_0, e3z_1, e3z_2, e3z_3, i, z1, z2, z3);

            let z = T::index_sharevec(&polys.z.eval, i);
            let zp = T::index_sharevec(&zp_vec, i);

            e2 = driver.mul_with_public(&challenges.alpha, &e2);
            e2z = driver.mul_with_public(&challenges.alpha, &e2z);

            e3 = driver.mul_with_public(&challenges.alpha, &e3);
            e3z = driver.mul_with_public(&challenges.alpha, &e3z);

            let mut e4 = driver.add_with_public(&-P::ScalarField::one(), &z);
            e4 = driver.mul_with_public(&zkey.lagrange[0].evaluations[i], &e4);
            e4 = driver.mul_with_public(&challenges.alpha2, &e4);

            let mut e4z = driver.mul_with_public(&zkey.lagrange[0].evaluations[i], &zp);
            e4z = driver.mul_with_public(&challenges.alpha2, &e4z);

            let mut t = driver.add(&e1[i], &e2);
            t = driver.sub(&t, &e3);
            t = driver.add(&t, &e4);

            let mut tz = driver.add(&e1z[i], &e2z);
            tz = driver.sub(&tz, &e3z);
            tz = driver.add(&tz, &e4z);

            t_vec.push(t);
            tz_vec.push(tz);
        }
        let mut coefficients_t = driver.ifft(&t_vec.into(), &domains.constraint_domain16);
        driver.neg_vec_in_place_limit(&mut coefficients_t, zkey.domain_size);

        for i in zkey.domain_size..zkey.domain_size * 4 {
            let a_lhs = T::index_sharevec(&coefficients_t, i - zkey.domain_size);
            let a_rhs = T::index_sharevec(&coefficients_t, i);
            let a = driver.sub(&a_lhs, &a_rhs);
            T::set_index_sharevec(&mut coefficients_t, a, i);
            /*
              We cannot check whether the polynomial is divisible by Zh here
            */
        }

        let coefficients_tz = driver.ifft(&tz_vec.into(), &domains.constraint_domain16);
        let t_final = driver.add_vec(&coefficients_t, &coefficients_tz);
        let mut t_final = t_final.into_iter();
        let mut t1 = Vec::with_capacity(zkey.domain_size + 1);
        let mut t2 = Vec::with_capacity(zkey.domain_size + 1);
        for _ in 0..zkey.domain_size {
            t1.push(t_final.next().unwrap());
        }
        for _ in 0..zkey.domain_size {
            t2.push(t_final.next().unwrap());
        }
        let mut t3 = t_final.collect::<Vec<_>>();
        t1.push(challenges.b[9].to_owned());

        t2[0] = driver.sub(&t2[0], &challenges.b[9]);
        t2.push(challenges.b[10].to_owned());

        t3[0] = driver.sub(&t3[0], &challenges.b[10]);

        Ok([t1.into(), t2.into(), t3.into()])
    }
    pub(super) fn round3(
        driver: &mut T,
        domains: Domains<P>,
        challenges: Round2Challenges<T, P>,
        proof: Round2Proof<P>,
        polys: Round2Polys<T, P>,
        data: PlonkData<T, P>,
    ) -> PlonkProofResult<Self> {
        let mut transcript = Keccak256Transcript::<P>::default();
        // STEP 3.1 - Compute evaluation challenge alpha ∈ F
        transcript.add_scalar(challenges.beta);
        transcript.add_scalar(challenges.gamma);
        transcript.add_point(proof.commit_z.into());

        let alpha = transcript.get_challenge();
        let alpha2 = alpha.square();
        let challenges = Round3Challenges::new(challenges, alpha, alpha2);
        let [t1, t2, t3] = Self::compute_t(driver, &domains, &challenges, &data.zkey, &polys)?;

        // Compute [T1]_1, [T2]_1, [T3]_1
        let commit_t1 = MSMProvider::<P::G1>::msm_public_points(driver, &data.zkey.p_tau, &t1);
        let commit_t2 = MSMProvider::<P::G1>::msm_public_points(driver, &data.zkey.p_tau, &t2);
        let commit_t3 = MSMProvider::<P::G1>::msm_public_points(driver, &data.zkey.p_tau, &t3);

        let opened = driver.open_point_many(&[commit_t1, commit_t2, commit_t3])?;
        debug_assert_eq!(opened.len(), 3);
        let polys = Round3Polys::new(polys, t1, t2, t3);
        let proof = Round3Proof::new(proof, opened[0], opened[1], opened[2]);
        Ok(Round::Round4 {
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
    fn test_round3_multiplier2() {
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
        let round3 = round2.next_round(&mut driver).unwrap();
        if let Round::Round4 {
            domains,
            challenges,
            proof,
            polys,
            data,
        } = round3.next_round(&mut driver).unwrap()
        {
            assert_eq!(
                proof.commit_t1,
                g1_from_xy!(
                    "19565274859171776656487339149400891418763323717194784371999925527281379783558",
                    "10386763814606525393157981088877418120953953282074496034368009898032691004276"
                )
            );
            assert_eq!(
                proof.commit_t2,
                g1_from_xy!(
                    "21373781172590655440995199489129447744768377173681683872177714389170709756453",
                    "15022438500999602781656113216961007860523642088813935888401623445602503921378"
                )
            );
            assert_eq!(
                proof.commit_t3,
                g1_from_xy!(
                    "17716310934983494559822846675349569221799734312843852303648316069451468517652",
                    "3901565218682159263339490461749079732547202222201762568846146351037318174485"
                )
            );
        } else {
            panic!("must be round4 after round3");
        }
    }
}
