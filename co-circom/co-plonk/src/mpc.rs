use std::future::Future;

use ark_ec::pairing::Pairing;
use ark_poly::EvaluationDomain;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub(crate) mod plain;
pub(crate) mod rep3;
pub(crate) mod shamir;

pub use plain::PlainPlonkDriver;
pub use rep3::Rep3PlonkDriver;
pub use shamir::ShamirPlonkDriver;

type IoResult<T> = std::io::Result<T>;

/// This trait represents the operations used during Groth16 proof generation
#[allow(async_fn_in_trait)]
pub trait CircomPlonkProver<P: Pairing> {
    /// The arithemitc share type
    type ArithmeticShare: CanonicalSerialize + CanonicalDeserialize + Copy + Clone + Default + Send;
    /// The G1 point share type
    type PointShareG1: Send;
    /// The G2 point share type
    type PointShareG2: Send;
    /// The party id type
    type PartyID: Send + Sync + Copy + std::fmt::Display;
    /// The IoContext type
    type IoContext;

    /// Gracefully shutdown the netowork. Waits until all data is sent and received
    async fn close_network(self) -> IoResult<()>;

    /// Generate a random arithmetic share
    fn rand(&mut self) -> IoResult<Self::ArithmeticShare>;

    /// Get the party id
    fn get_party_id(&self) -> Self::PartyID;

    /// Subtract the share b from the share a: \[c\] = \[a\] - \[b\]
    fn add(a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare;

    /// Add a public value a to the share b: \[c\] = a + \[b\]
    fn add_with_public(
        party_id: Self::PartyID,
        shared: Self::ArithmeticShare,
        public: P::ScalarField,
    ) -> Self::ArithmeticShare;

    /// Subtract the share b from the share a: \[c\] = \[a\] - \[b\]
    fn sub(a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare;

    /// Negates a vector of shared values: \[b\] = -\[a\] for every element in place.
    fn neg_vec_in_place(&mut self, a: &mut [Self::ArithmeticShare]);

    /// Multiply a share b by a public value a: c = a * \[b\].
    fn mul_with_public(
        shared: Self::ArithmeticShare,
        public: P::ScalarField,
    ) -> Self::ArithmeticShare;

    /// Performs element-wise multiplication of two vectors of shared values.
    /// Does not perform any networking.
    ///
    /// # Security
    /// If you want to perform additional non-linear operations on the result of this function,
    /// you *MUST* call [`io_round_mul_vec`] first. Only then the relevant network round is performed.
    fn local_mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> Vec<P::ScalarField>;

    /// Performs networking round of `local_mul_vec`
    async fn io_round_mul_vec(
        &mut self,
        a: Vec<P::ScalarField>,
    ) -> IoResult<Vec<Self::ArithmeticShare>>;

    /// Performs element-wise multiplication of two vectors of shared values.
    ///
    /// Use this function for small vecs. For large vecs see [`local_mul_vec`]
    fn mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<Self::ArithmeticShare>>>;

    /// Performs element-wise multiplication of three vectors of shared values.
    ///
    /// Use this function for small vecs. For large vecs see [`local_mul_vec`]
    fn mul_vecs(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
        c: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<Self::ArithmeticShare>>>;

    /// Convenience method for \[a\] + \[b\] * \[c\]
    fn add_mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
        c: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<Self::ArithmeticShare>>>;

    /// Convenience method for \[a\] + \[b\] * c
    fn add_mul_public(
        &mut self,
        a: Self::ArithmeticShare,
        b: Self::ArithmeticShare,
        c: P::ScalarField,
    ) -> Self::ArithmeticShare {
        Self::add(a, Self::mul_with_public(b, c))
    }

    /// This function performs a multiplication directly followed by an opening. This safes one round of communication in some MPC protocols compared to calling `mul` and `open` separately.
    fn mul_open_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<P::ScalarField>>>;

    /// Reconstructs many shared values: a = Open(\[a\]).
    fn open_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<P::ScalarField>>>;

    /// Computes the inverse of many shared values: \[b\] = \[a\] ^ -1. Requires network communication.
    fn inv_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> impl Future<Output = IoResult<Vec<Self::ArithmeticShare>>>;

    /// Transforms a public value into a shared value: \[a\] = a.
    fn promote_to_trivial_share(
        party_id: Self::PartyID,
        public_value: P::ScalarField,
    ) -> Self::ArithmeticShare;

    /// Computes the FFT of a vector of shared field elements.
    fn fft<D: EvaluationDomain<P::ScalarField>>(
        data: &[Self::ArithmeticShare],
        domain: &D,
    ) -> Vec<Self::ArithmeticShare>;

    /// Computes the inverse FFT of a vector of shared field elements.
    fn ifft<D: EvaluationDomain<P::ScalarField>>(
        data: &[Self::ArithmeticShare],
        domain: &D,
    ) -> Vec<Self::ArithmeticShare>;

    /// Reconstructs a shared point: A = Open(\[A\]).
    fn open_point_g1(&mut self, a: Self::PointShareG1) -> impl Future<Output = IoResult<P::G1>>;

    /// Reconstructs many shared points: A = Open(\[A\]).
    fn open_point_vec_g1(
        &mut self,
        a: &[Self::PointShareG1],
    ) -> impl Future<Output = IoResult<Vec<P::G1>>>;

    /// Perform msm between G1 `points` and `scalars`
    fn msm_public_points_g1(
        points: &[P::G1Affine],
        scalars: &[Self::ArithmeticShare],
    ) -> Self::PointShareG1;

    /// Evaluate the shared polynomial at the public point.
    /// Returns the evaluation and the polynomial
    fn evaluate_poly_public(
        poly: Vec<Self::ArithmeticShare>,
        point: P::ScalarField,
    ) -> (Self::ArithmeticShare, Vec<Self::ArithmeticShare>);

    /// Perform elementwise multiplication of the three vectors of shares and then perform the array_prod_mul protocol
    async fn array_prod_mul(
        io_context: &mut Self::IoContext,
        inv: bool,
        arr1: &[Self::ArithmeticShare],
        arr2: &[Self::ArithmeticShare],
        arr3: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>>;

    /// Perform `array_prod_mul` for two sets of three inputs concurrently
    async fn array_prod_mul2(
        &mut self,
        n1: &[Self::ArithmeticShare],
        n2: &[Self::ArithmeticShare],
        n3: &[Self::ArithmeticShare],
        d1: &[Self::ArithmeticShare],
        d2: &[Self::ArithmeticShare],
        d3: &[Self::ArithmeticShare],
    ) -> IoResult<(Vec<Self::ArithmeticShare>, Vec<Self::ArithmeticShare>)>;
}
