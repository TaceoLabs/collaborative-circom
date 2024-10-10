//! A Groth16 proof protocol that uses a collaborative MPC protocol to generate the proof.
use ark_ec::pairing::Pairing;
use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{FftField, PrimeField};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::{ConstraintMatrices, Matrix, SynthesisError};
use circom_types::groth16::{Groth16Proof, ZKey};
use circom_types::traits::{CircomArkworksPairingBridge, CircomArkworksPrimeFieldBridge};
use co_circom_snarks::SharedWitness;
use eyre::Result;
use mpc_core::protocols::rep3::network::{IoContext, Rep3MpcNet};
use mpc_core::protocols::shamir::network::ShamirMpcNet;
use mpc_core::protocols::shamir::{ShamirPreprocessing, ShamirProtocol};
use mpc_net::config::NetworkConfig;
use num_traits::identities::One;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::oneshot;
use tracing::instrument;

use crate::mpc::plain::PlainGroth16Driver;
use crate::mpc::rep3::Rep3Groth16Driver;
use crate::mpc::shamir::ShamirGroth16Driver;
use crate::mpc::CircomGroth16Prover;

macro_rules! rayon_join {
    ($t1: expr, $t2: expr, $t3: expr) => {{
        let ((x, y), z) = rayon::join(|| rayon::join(|| $t1, || $t2), || $t3);
        (x, y, z)
    }};
}

/// The plain [`Groth16`] type.
///
/// This type is actually the [`CoGroth16`] type initialized with
/// the [`PlainDriver`], a single party (you) MPC protocol (i.e., your everyday Groth16).
/// You can use this instance to create a proof, but we recommend against it for a real use-case.
/// Have a look at the [Groth16 implementation of arkworks](https://docs.rs/ark-groth16/latest/ark_groth16/)
/// for a plain Groth16 prover.
///
/// More interesting is the [`Groth16::verify`] method. You can verify any circom Groth16 proof, be it
/// from snarkjs or one created by this project. Under the hood we use the arkwork Groth16 project for verifying.
pub type Groth16<P> = CoGroth16<P, PlainGroth16Driver>;

/// A type alias for a [CoGroth16] protocol using replicated secret sharing.
pub type Rep3CoGroth16<P, N> = CoGroth16<P, Rep3Groth16Driver<N>>;
/// A type alias for a [CoGroth16] protocol using shamir secret sharing.
pub type ShamirCoGroth16<P, N> = CoGroth16<P, ShamirGroth16Driver<<P as Pairing>::ScalarField, N>>;

/* old way of computing root of unity, does not work for bls12_381:
let root_of_unity = {
    let domain_size_double = 2 * domain_size;
    let domain_double =
        D::new(domain_size_double).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    domain_double.element(1)
};
new one is computed in the same way as in snarkjs (More precisely in ffjavascript/src/wasm_field1.js)
calculate smallest quadratic non residue q (by checking q^((p-1)/2)=-1 mod p) also calculate smallest t (F::TRACE) s.t. p-1=2^s*t, s is the two_adicity
use g=q^t (this is a 2^s-th root of unity) as (some kind of) generator and compute another domain by repeatedly squaring g, should get to 1 in the s+1-th step.
then if log2(domain_size) equals s we take as root of unity q^2, and else we take the log2(domain_size) + 1-th element of the domain created above
*/
#[instrument(level = "debug", name = "root of unity", skip_all)]
fn root_of_unity_for_groth16<F: PrimeField + FftField>(
    pow: usize,
    domain: &mut GeneralEvaluationDomain<F>,
) -> F {
    let (q, roots) = co_circom_snarks::utils::roots_of_unity::<F>();
    match domain {
        GeneralEvaluationDomain::Radix2(domain) => {
            domain.group_gen = roots[pow];
            domain.group_gen_inv = domain.group_gen.inverse().expect("can compute inverse");
        }
        GeneralEvaluationDomain::MixedRadix(domain) => {
            domain.group_gen = roots[pow];
            domain.group_gen_inv = domain.group_gen.inverse().expect("can compute inverse");
        }
    };
    if F::TWO_ADICITY.to_u64().unwrap() == domain.log_size_of_group() {
        q.square()
    } else {
        roots[domain.log_size_of_group().to_usize().unwrap() + 1]
    }
}

/// A Groth16 proof protocol that uses a collaborative MPC protocol to generate the proof.
pub struct CoGroth16<P: Pairing, T: CircomGroth16Prover<P>> {
    pub(crate) driver: T,
    phantom_data: PhantomData<P>,
}

impl<P: Pairing + CircomArkworksPairingBridge, T: CircomGroth16Prover<P>> CoGroth16<P, T>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    /// Creates a new [CoGroth16] protocol with a given MPC driver.
    pub fn new(driver: T) -> Self {
        Self {
            driver,
            phantom_data: PhantomData,
        }
    }

    /// Execute the Groth16 prover using the internal MPC driver.
    /// This version takes the Circom-generated constraint matrices as input and does not re-calculate them.
    #[instrument(level = "debug", name = "Groth16 - Proof", skip_all)]
    pub async fn prove(
        mut self,
        zkey: &ZKey<P>,
        private_witness: SharedWitness<P::ScalarField, T::ArithmeticShare>,
    ) -> Result<Groth16Proof<P>> {
        let id = self.driver.get_party_id();
        tracing::info!("Party {}: starting proof generation..", id);
        let start = Instant::now();
        let matrices = &zkey.matrices;
        let num_inputs = matrices.num_instance_variables;
        let num_constraints = matrices.num_constraints;
        let public_inputs = Arc::new(private_witness.public_inputs);
        let private_witness = Arc::new(private_witness.witness);
        let h = self
            .witness_map_from_matrices(
                zkey.pow,
                matrices,
                num_constraints,
                num_inputs,
                &public_inputs,
                &private_witness,
            )
            .await?;
        let (r, s) = (self.driver.rand()?, self.driver.rand()?);
        let proof = self
            .create_proof_with_assignment(zkey, r, s, h, public_inputs, private_witness)
            .await?;

        let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
        tracing::info!("Party {}: Proof generation took {} ms", id, duration_ms);
        Ok(proof)
    }

    fn evaluate_constraint(
        party_id: T::PartyID,
        domain_size: usize,
        matrix: &Matrix<P::ScalarField>,
        public_inputs: &[P::ScalarField],
        private_witness: &[T::ArithmeticShare],
    ) -> Vec<T::ArithmeticShare> {
        let mut result = matrix
            .par_iter()
            .with_min_len(32)
            .map(|x| T::evaluate_constraint(party_id, x, public_inputs, private_witness))
            .collect::<Vec<_>>();
        result.resize(domain_size, T::ArithmeticShare::default());
        result
    }

    #[instrument(level = "debug", name = "witness map from matrices", skip_all)]
    async fn witness_map_from_matrices(
        &mut self,
        power: usize,
        matrices: &ConstraintMatrices<P::ScalarField>,
        num_constraints: usize,
        num_inputs: usize,
        public_inputs: &[P::ScalarField],
        private_witness: &[T::ArithmeticShare],
    ) -> Result<Vec<P::ScalarField>> {
        let mut domain =
            GeneralEvaluationDomain::<P::ScalarField>::new(num_constraints + num_inputs)
                .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain_size = domain.size();
        let party_id = self.driver.get_party_id();
        let eval_constraint_span =
            tracing::debug_span!("evaluate constraints + root of unity computation").entered();
        let (roots_to_power_domain, a, b) = rayon_join!(
            {
                let root_of_unity_span =
                    tracing::debug_span!("root of unity computation").entered();
                let root_of_unity = root_of_unity_for_groth16(power, &mut domain);
                let mut roots = Vec::with_capacity(domain_size);
                let mut c = P::ScalarField::one();
                for _ in 0..domain_size {
                    roots.push(c);
                    c *= root_of_unity;
                }
                root_of_unity_span.exit();
                Arc::new(roots)
            },
            {
                let eval_constraint_span_a =
                    tracing::debug_span!("evaluate constraints - a").entered();
                let mut result = Self::evaluate_constraint(
                    party_id,
                    domain_size,
                    &matrices.a,
                    public_inputs,
                    private_witness,
                );
                let promoted_public = T::promote_to_trivial_shares(party_id, public_inputs);
                result[num_constraints..num_constraints + num_inputs]
                    .clone_from_slice(&promoted_public[..num_inputs]);
                eval_constraint_span_a.exit();
                result
            },
            {
                let eval_constraint_span_b =
                    tracing::debug_span!("evaluate constraints - a").entered();
                let result = Self::evaluate_constraint(
                    party_id,
                    domain_size,
                    &matrices.b,
                    public_inputs,
                    private_witness,
                );
                eval_constraint_span_b.exit();
                result
            }
        );

        let domain = Arc::new(domain);
        eval_constraint_span.exit();

        let (a_tx, a_rx) = oneshot::channel();
        let (b_tx, b_rx) = oneshot::channel();
        let (c_tx, c_rx) = oneshot::channel();
        let a_domain = Arc::clone(&domain);
        let b_domain = Arc::clone(&domain);
        let c_domain = Arc::clone(&domain);
        let mut a_result = a.clone();
        let mut b_result = b.clone();
        let a_roots = Arc::clone(&roots_to_power_domain);
        let b_roots = Arc::clone(&roots_to_power_domain);
        let c_roots = Arc::clone(&roots_to_power_domain);
        rayon::spawn(move || {
            let a_span = tracing::debug_span!("distribute powers mul a (fft/ifft)").entered();
            a_domain.ifft_in_place(&mut a_result);
            T::distribute_powers_and_mul_by_const(&mut a_result, &a_roots);
            a_domain.fft_in_place(&mut a_result);
            a_tx.send(a_result).expect("channel not droped");
            a_span.exit();
        });

        rayon::spawn(move || {
            let b_span = tracing::debug_span!("distribute powers mul b (fft/ifft)").entered();
            b_domain.ifft_in_place(&mut b_result);
            T::distribute_powers_and_mul_by_const(&mut b_result, &b_roots);
            b_domain.fft_in_place(&mut b_result);
            b_tx.send(b_result).expect("channel not droped");
            b_span.exit();
        });

        let local_mul_vec_span = tracing::debug_span!("c: local_mul_vec").entered();
        // this performs some heavy computation on the tokio runtime.
        // it doesn't matter because we do not perform any IO at this moment, so the thread
        // is not needed for driving IO
        let mut ab = self.driver.local_mul_vec(a, b);
        local_mul_vec_span.exit();
        rayon::spawn(move || {
            let ifft_span = tracing::debug_span!("c: ifft in dist pows").entered();
            c_domain.ifft_in_place(&mut ab);
            ifft_span.exit();
            let dist_pows_span = tracing::debug_span!("c: dist pows").entered();
            #[allow(unused_mut)]
            ab.par_iter_mut()
                .zip_eq(c_roots.par_iter())
                .with_min_len(512)
                .for_each(|(share, pow)| {
                    *share *= pow;
                });
            dist_pows_span.exit();
            let fft_span = tracing::debug_span!("c: fft in dist pows").entered();
            c_domain.fft_in_place(&mut ab);
            fft_span.exit();
            c_tx.send(ab).expect("channel not dropped");
        });

        let a = a_rx.await?;
        let b = b_rx.await?;

        let compute_ab_span = tracing::debug_span!("compute ab").entered();
        let local_ab_span = tracing::debug_span!("local part (mul and sub)").entered();
        // same as above. No IO task is run at the moment.
        let mut ab = self.driver.local_mul_vec(a, b);
        let c = c_rx.await?;
        ab.par_iter_mut()
            .zip_eq(c.par_iter())
            .with_min_len(512)
            .for_each(|(a, b)| {
                *a -= b;
            });
        local_ab_span.exit();
        compute_ab_span.exit();
        Ok(ab)
    }

    fn calculate_coeff_g1(
        id: T::PartyID,
        initial: T::PointShareG1,
        query: &[P::G1Affine],
        vk_param: P::G1Affine,
        input_assignment: &[P::ScalarField],
        aux_assignment: &[T::ArithmeticShare],
    ) -> T::PointShareG1 {
        let pub_len = input_assignment.len();

        // we block this thread of the runtime here.
        // It should not matter too much, as we have multithreaded
        // runtime.
        let (pub_acc, priv_acc) = rayon::join(
            || P::G1::msm_unchecked(&query[1..=pub_len], input_assignment),
            || T::msm_public_points_g1(&query[1 + pub_len..], aux_assignment),
        );

        let mut res = initial;
        T::add_assign_points_public_g1(id, &mut res, &query[0].into_group());
        T::add_assign_points_public_g1(id, &mut res, &vk_param.into_group());
        T::add_assign_points_public_g1(id, &mut res, &pub_acc);
        T::add_assign_points_g1(&mut res, &priv_acc);
        res
    }

    fn calculate_coeff_g2(
        id: T::PartyID,
        initial: T::PointShareG2,
        query: &[P::G2Affine],
        vk_param: P::G2Affine,
        input_assignment: &[P::ScalarField],
        aux_assignment: &[T::ArithmeticShare],
    ) -> T::PointShareG2 {
        let pub_len = input_assignment.len();

        // we block this thread of the runtime here.
        // It should not matter too much, as we have multithreaded
        // runtime.
        let (pub_acc, priv_acc) = rayon::join(
            || P::G2::msm_unchecked(&query[1..=pub_len], input_assignment),
            || T::msm_public_points_g2(&query[1 + pub_len..], aux_assignment),
        );

        let mut res = initial;
        T::add_assign_points_public_g2(id, &mut res, &query[0].into_group());
        T::add_assign_points_public_g2(id, &mut res, &vk_param.into_group());
        T::add_assign_points_public_g2(id, &mut res, &pub_acc);
        T::add_assign_points_g2(&mut res, &priv_acc);
        res
    }

    #[instrument(level = "debug", name = "create proof with assignment", skip_all)]
    async fn create_proof_with_assignment(
        mut self,
        zkey: &ZKey<P>,
        r: T::ArithmeticShare,
        s: T::ArithmeticShare,
        h: Vec<P::ScalarField>,
        input_assignment: Arc<Vec<P::ScalarField>>,
        aux_assignment: Arc<Vec<T::ArithmeticShare>>,
    ) -> Result<Groth16Proof<P>> {
        let delta_g1 = zkey.delta_g1.into_group();
        let (l_acc_tx, l_acc_rx) = oneshot::channel();
        let h_query = Arc::clone(&zkey.h_query);
        let l_query = Arc::clone(&zkey.l_query);

        let party_id = self.driver.get_party_id();
        let (r_g1_tx, r_g1_rx) = oneshot::channel();
        let (s_g1_tx, s_g1_rx) = oneshot::channel();
        let (s_g2_tx, s_g2_rx) = oneshot::channel();
        let a_query = Arc::clone(&zkey.a_query);
        let b_g1_query = Arc::clone(&zkey.b_g1_query);
        let b_g2_query = Arc::clone(&zkey.b_g2_query);
        let input_assignment1 = Arc::clone(&input_assignment);
        let input_assignment2 = Arc::clone(&input_assignment);
        let input_assignment3 = Arc::clone(&input_assignment);
        let aux_assignment1 = Arc::clone(&aux_assignment);
        let aux_assignment2 = Arc::clone(&aux_assignment);
        let aux_assignment3 = Arc::clone(&aux_assignment);
        let aux_assignment4 = Arc::clone(&aux_assignment);
        let alpha_g1 = zkey.vk.alpha_g1;
        let beta_g1 = zkey.beta_g1;
        let beta_g2 = zkey.vk.beta_g2;
        let delta_g2 = zkey.vk.delta_g2.into_group();

        rayon::spawn(move || {
            let compute_a =
                tracing::debug_span!("compute A in create proof with assignment").entered();
            // Compute A
            let r_g1 = T::scalar_mul_public_point_g1(&delta_g1, r);
            let r_g1 = Self::calculate_coeff_g1(
                party_id,
                r_g1,
                &a_query,
                alpha_g1,
                &input_assignment1[1..],
                &aux_assignment1,
            );
            r_g1_tx.send(r_g1).expect("not dropped");
            compute_a.exit();
        });

        rayon::spawn(move || {
            let compute_b =
                tracing::debug_span!("compute B/G1 in create proof with assignment").entered();
            // Compute B in G1
            // In original implementation this is skipped if r==0, however r is shared in our case
            let s_g1 = T::scalar_mul_public_point_g1(&delta_g1, s);
            let s_g1 = Self::calculate_coeff_g1(
                party_id,
                s_g1,
                &b_g1_query,
                beta_g1,
                &input_assignment2[1..],
                &aux_assignment2,
            );
            s_g1_tx.send(s_g1).expect("not dropped");
            compute_b.exit();
        });

        rayon::spawn(move || {
            let compute_b =
                tracing::debug_span!("compute B/G2 in create proof with assignment").entered();
            // Compute B in G2
            let s_g2 = T::scalar_mul_public_point_g2(&delta_g2, s);
            let s_g2 = Self::calculate_coeff_g2(
                party_id,
                s_g2,
                &b_g2_query,
                beta_g2,
                &input_assignment3[1..],
                &aux_assignment3,
            );
            s_g2_tx.send(s_g2).expect("not dropped");
            compute_b.exit();
        });

        rayon::spawn(move || {
            let msm_l_query = tracing::debug_span!("msm l_query").entered();
            let result = T::msm_public_points_g1(l_query.as_ref(), &aux_assignment4);
            l_acc_tx.send(result).expect("channel not dropped");
            msm_l_query.exit();
        });

        // TODO we can remove the networking round in msm_and_mul because we only have linear operations afterwards
        let (h_acc, rs) = self.driver.msm_and_mul(h, h_query, r, s).await?;
        let r_s_delta_g1 = T::scalar_mul_public_point_g1(&delta_g1, rs);

        let l_aux_acc = l_acc_rx.await.expect("channel not dropped");

        let calculate_coeff_span = tracing::debug_span!("calculate coeff").entered();
        let g_a = r_g1_rx.await?;
        let g1_b = s_g1_rx.await?;
        calculate_coeff_span.exit();

        let network_round = tracing::debug_span!("network round after calc coeff").entered();
        let (g_a_opened, r_g1_b) = self
            .driver
            .open_point_and_scalar_mul(&g_a, &g1_b, r)
            .await?;
        network_round.exit();

        let last_round = tracing::debug_span!("finish open two points and some adds").entered();
        let s_g_a = T::scalar_mul_public_point_g1(&g_a_opened, s);

        let mut g_c = s_g_a;
        T::add_assign_points_g1(&mut g_c, &r_g1_b);
        T::sub_assign_points_g1(&mut g_c, &r_s_delta_g1);
        T::add_assign_points_g1(&mut g_c, &l_aux_acc);
        T::add_assign_points_g1(&mut g_c, &h_acc);

        let g2_b = s_g2_rx.await?;
        let (g_c_opened, g2_b_opened) = self.driver.open_two_points(g_c, g2_b).await?;
        last_round.exit();
        self.driver.close_network().await?;

        Ok(Groth16Proof {
            pi_a: g_a_opened.into_affine(),
            pi_b: g2_b_opened.into_affine(),
            pi_c: g_c_opened.into_affine(),
            protocol: "groth16".to_owned(),
            curve: P::get_circom_name(),
        })
    }
}

impl<P: Pairing> Rep3CoGroth16<P, Rep3MpcNet>
where
    P: CircomArkworksPairingBridge,
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    /// Create a new [Rep3CoGroth16] protocol with a given network configuration.
    pub async fn with_network_config(config: NetworkConfig) -> Result<Self> {
        let mpc_net = Rep3MpcNet::new(config).await?;
        let mut io_context0 = IoContext::init(mpc_net).await?;
        let io_context1 = io_context0.fork().await?;
        let driver = Rep3Groth16Driver::new(io_context0, io_context1);
        Ok(CoGroth16 {
            driver,
            phantom_data: PhantomData,
        })
    }
}

impl<P: Pairing> ShamirCoGroth16<P, ShamirMpcNet>
where
    P: CircomArkworksPairingBridge,
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    /// Create a new [ShamirCoGroth16] protocol with a given network configuration.
    pub async fn with_network_config(
        threshold: usize,
        config: NetworkConfig,
        zkey: &ZKey<P>,
    ) -> Result<Self> {
        let domain_size = 2usize.pow(u32::try_from(zkey.pow).expect("pow fits into u32"));
        // we need domain_size + 2 + 1 number of corr rand pairs in witness_map_from_matrices (degree_reduce_vec + r and s + 1 for fork)
        let num_pairs = domain_size + 2 + 1;
        let mpc_net = ShamirMpcNet::new(config).await?;
        let preprocessing = ShamirPreprocessing::new(threshold, mpc_net, num_pairs).await?;
        let mut protocol0 = ShamirProtocol::from(preprocessing);
        // the protocol1 is only used for scalar_mul and a field_mul which need 1 pair each (ergo 2
        // pairs)
        let protocol1 = protocol0.fork_with_pairs(2).await?;
        let driver = ShamirGroth16Driver::new(protocol0, protocol1);
        Ok(CoGroth16 {
            driver,
            phantom_data: PhantomData,
        })
    }
}

impl<P: Pairing> Groth16<P>
where
    P: CircomArkworksPairingBridge,
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    /// *Locally* create a `Groth16` proof. This is just the [`CoGroth16`] prover
    /// initialized with the [`PlainDriver`].
    ///
    /// DOES NOT PERFORM ANY MPC. For a plain prover checkout the [Groth16 implementation of arkworks](https://docs.rs/ark-groth16/latest/ark_groth16/).
    pub async fn plain_prove(
        zkey: &ZKey<P>,
        private_witness: SharedWitness<P::ScalarField, P::ScalarField>,
    ) -> Result<Groth16Proof<P>> {
        let prover = Self {
            driver: PlainGroth16Driver,
            phantom_data: PhantomData,
        };
        prover.prove(zkey, private_witness).await
    }
}
