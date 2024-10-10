use core::fmt;
use std::{fmt::Debug, sync::Arc};

use ark_ec::pairing::Pairing;
use ark_poly::domain::DomainCoeff;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub(crate) mod plain;
pub(crate) mod rep3;
pub(crate) mod shamir;

pub use plain::PlainGroth16Driver;
pub use rep3::Rep3Groth16Driver;
pub use shamir::ShamirGroth16Driver;

type IoResult<T> = std::io::Result<T>;

/// This trait represents the operations used during Groth16 proof generation
#[allow(async_fn_in_trait)]
pub trait CircomGroth16Prover<P: Pairing>: Send + Sized {
    /// The arithemitc share type
    type ArithmeticShare: CanonicalSerialize
        + CanonicalDeserialize
        + Copy
        + Clone
        + Default
        + Send
        + Debug
        + DomainCoeff<P::ScalarField>
        + 'static;
    /// The G1 point share type
    type PointShareG1: Debug + Send + 'static;
    /// The G2 point share type
    type PointShareG2: Debug + Send + 'static;
    /// The party id type
    type PartyID: Send + Sync + Copy + fmt::Display + 'static;

    /// Gracefully shutdown the netowork. Waits until all data is sent and received
    async fn close_network(self) -> IoResult<()>;

    /// Generate a random arithemitc share
    fn rand(&mut self) -> IoResult<Self::ArithmeticShare>;

    /// Get the party id
    fn get_party_id(&self) -> Self::PartyID;

    /// Each value of lhs consists of a coefficient c and an index i. This function computes the sum of the coefficients times the corresponding public input or private witness. In other words, an accumulator a is initialized to 0, and for each (c, i) in lhs, a += c * public_inputs\[i\] is computed if i corresponds to a public input, or c * private_witness[i - public_inputs.len()] if i corresponds to a private witness.
    fn evaluate_constraint(
        party_id: Self::PartyID,
        lhs: &[(P::ScalarField, usize)],
        public_inputs: &[P::ScalarField],
        private_witness: &[Self::ArithmeticShare],
    ) -> Self::ArithmeticShare;

    /// Elementwise transformation of a vector of public values into a vector of shared values: \[a_i\] = a_i.
    fn promote_to_trivial_shares(
        id: Self::PartyID,
        public_values: &[P::ScalarField],
    ) -> Vec<Self::ArithmeticShare>;

    /// Performs element-wise multiplication of two vectors of shared values.
    /// Does not perform any networking.
    ///
    /// # Security
    /// You must *NOT* perform additional non-linear operations on the result of this function.
    fn local_mul_vec(
        &mut self,
        a: Vec<Self::ArithmeticShare>,
        b: Vec<Self::ArithmeticShare>,
    ) -> Vec<P::ScalarField>;

    /// Compute the msm of `h` and `h_query` and multiplication `r` * `s`.
    async fn msm_and_mul(
        &mut self,
        h: Vec<<P as Pairing>::ScalarField>,
        h_query: Arc<Vec<P::G1Affine>>,
        r: Self::ArithmeticShare,
        s: Self::ArithmeticShare,
    ) -> IoResult<(Self::PointShareG1, Self::ArithmeticShare)>;

    /// Computes the \[coeffs_i\] *= c * g^i for the coefficients in 0 <= i < coeff.len()
    fn distribute_powers_and_mul_by_const(
        coeffs: &mut [Self::ArithmeticShare],
        roots: &[P::ScalarField],
    );

    /// Perform msm between G1 `points` and `scalars`
    fn msm_public_points_g1(
        points: &[P::G1Affine],
        scalars: &[Self::ArithmeticShare],
    ) -> Self::PointShareG1;

    /// Perform msm between G2 `points` and `scalars`
    fn msm_public_points_g2(
        points: &[P::G2Affine],
        scalars: &[Self::ArithmeticShare],
    ) -> Self::PointShareG2;

    /// Multiplies a public point B to the shared point A in place: \[A\] *= B
    fn scalar_mul_public_point_g1(a: &P::G1, b: Self::ArithmeticShare) -> Self::PointShareG1;

    /// Add a shared point B in place to the shared point A: \[A\] += \[B\]
    fn add_assign_points_g1(a: &mut Self::PointShareG1, b: &Self::PointShareG1);

    /// Add a public point B in place to the shared point A
    fn add_assign_points_public_g1(id: Self::PartyID, a: &mut Self::PointShareG1, b: &P::G1);

    /// Reconstructs a shared point: A = Open(\[A\]).
    async fn open_point_g1(&mut self, a: &Self::PointShareG1) -> IoResult<P::G1>;

    /// Multiplies a share b to the shared point A: \[A\] *= \[b\]. Requires network communication.
    async fn scalar_mul_g1(
        &mut self,
        a: &Self::PointShareG1,
        b: Self::ArithmeticShare,
    ) -> IoResult<Self::PointShareG1>;

    /// Subtract a shared point B in place from the shared point A: \[A\] -= \[B\]
    fn sub_assign_points_g1(a: &mut Self::PointShareG1, b: &Self::PointShareG1);

    /// Perform scalar multiplication of point A with a shared scalar b
    fn scalar_mul_public_point_g2(a: &P::G2, b: Self::ArithmeticShare) -> Self::PointShareG2;

    /// Add a shared point B in place to the shared point A: \[A\] += \[B\]
    fn add_assign_points_g2(a: &mut Self::PointShareG2, b: &Self::PointShareG2);

    /// Add a public point B in place to the shared point A
    fn add_assign_points_public_g2(id: Self::PartyID, a: &mut Self::PointShareG2, b: &P::G2);

    /// Reconstructs a shared points: A = Open(\[A\]), B = Open(\[B\]).
    async fn open_two_points(
        &mut self,
        a: Self::PointShareG1,
        b: Self::PointShareG2,
    ) -> std::io::Result<(P::G1, P::G2)>;

    /// Reconstruct point G_a and perform scalar multiplication of G1_b and r concurrently
    async fn open_point_and_scalar_mul(
        &mut self,
        g_a: &Self::PointShareG1,
        g1_b: &Self::PointShareG1,
        r: Self::ArithmeticShare,
    ) -> std::io::Result<(P::G1, Self::PointShareG1)>;
}
