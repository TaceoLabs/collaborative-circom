pub(crate) mod co_decider;
pub(crate) mod co_oink;
pub(crate) mod mpc;
pub(crate) mod parse;
pub mod prelude;
pub(crate) mod prover;
pub(crate) mod types;

use ark_ec::pairing::Pairing;
use mpc::{plain::PlainUltraHonkDriver, NoirUltraHonkProver};
use parse::builder_variable::SharedBuilderVariable;
use ultrahonk::prelude::ProverCrs;

impl<P: Pairing> SharedBuilderVariable<PlainUltraHonkDriver, P> {
    pub fn promote_public_witness_vector(
        witness: Vec<P::ScalarField>,
    ) -> Vec<SharedBuilderVariable<PlainUltraHonkDriver, P>> {
        witness
            .into_iter()
            .map(SharedBuilderVariable::Public)
            .collect()
    }
}

pub(crate) const NUM_ALPHAS: usize = ultrahonk::NUM_ALPHAS;
// The log of the max circuit size assumed in order to achieve constant sized Honk proofs
// AZTEC TODO(https://github.com/AztecProtocol/barretenberg/issues/1046): Remove the need for const sized proofs
pub(crate) const CONST_PROOF_SIZE_LOG_N: usize = ultrahonk::CONST_PROOF_SIZE_LOG_N;
pub(crate) const N_MAX: usize = ultrahonk::N_MAX;
pub const OINK_CRAND_PAIRS_FACTOR_N: usize = co_oink::CRAND_PAIRS_FACTOR_N;
pub const OINK_CRAND_PAIRS_FACTOR_N_MINUS_ONE: usize = co_oink::CRAND_PAIRS_FACTOR_N_MINUS_ONE;
pub const OINK_CRAND_PAIRS_CONST: usize = co_oink::CRAND_PAIRS_CONST;
pub const SUMCHECK_ROUND_CRAND_PAIRS_FACTOR: usize = co_decider::relations::CRAND_PAIRS_FACTOR;
pub const MAX_PARTIAL_RELATION_LENGTH: usize = co_decider::types::MAX_PARTIAL_RELATION_LENGTH;

pub(crate) struct CoUtils {}

impl CoUtils {
    pub(crate) fn commit<T: NoirUltraHonkProver<P>, P: Pairing>(
        poly: &[T::ArithmeticShare],
        crs: &ProverCrs<P>,
    ) -> T::PointShareG1 {
        let len = poly.len();
        T::msm_public_points_g1(&crs.monomials[..len], poly)
    }

    pub(crate) async fn batch_invert<T: NoirUltraHonkProver<P>, P: Pairing>(
        driver: &mut T,
        poly: &mut [T::ArithmeticShare],
    ) -> std::io::Result<()> {
        driver.inv_many_in_place(poly).await
    }
}
