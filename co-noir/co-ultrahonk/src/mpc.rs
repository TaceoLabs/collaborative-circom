use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub(crate) mod plain;
pub(crate) mod rep3;
pub(crate) mod shamir;

/// This trait represents the operations used during UltraHonk proof generation
#[allow(async_fn_in_trait)]
pub trait NoirUltraHonkProver<P: Pairing>: Send + Sized {
    /// The arithemitc share type
    type ArithmeticShare: CanonicalSerialize
        + CanonicalDeserialize
        + Copy
        + Clone
        + Default
        + Send
        + PartialEq
        + std::fmt::Debug
        + 'static;
    /// The G1 point share type
    type PointShareG1: std::fmt::Debug + Send + 'static;
    /// The G2 point share type
    type PointShareG2: std::fmt::Debug + Send + 'static;
    /// The party id type
    type PartyID: Copy;

    /// Generate a share of a random value. The value is thereby unknown to anyone.
    fn rand(&mut self) -> std::io::Result<Self::ArithmeticShare>;

    /// Get the party id
    fn get_party_id(&self) -> Self::PartyID;

    /// Subtract the share b from the share a: \[c\] = \[a\] - \[b\]
    fn sub(&self, a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare;

    /// Add two shares: \[c\] = \[a\] + \[b\]
    fn add(&self, a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare;

    /// Negates a shared value: \[b\] = -\[a\].
    fn neg(&mut self, a: Self::ArithmeticShare) -> Self::ArithmeticShare;

    /// Multiply a share b by a public value a: c = a * \[b\].
    fn mul_with_public(
        &self,
        public: P::ScalarField,
        shared: Self::ArithmeticShare,
    ) -> Self::ArithmeticShare;

    /// Multiply two shares: \[c\] = \[a\] * \[b\]. Requires network communication.
    async fn mul_many(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> std::io::Result<Vec<Self::ArithmeticShare>>;

    /// Add a public value a to the share b: \[c\] = a + \[b\]
    fn add_with_public(
        &self,
        public: P::ScalarField,
        shared: Self::ArithmeticShare,
    ) -> Self::ArithmeticShare;

    /// Transforms a public value into a shared value: \[a\] = a.
    fn promote_to_trivial_share(
        id: Self::PartyID,
        public_value: P::ScalarField,
    ) -> Self::ArithmeticShare;

    /// Elementwise transformation of a vector of public values into a vector of shared values: \[a_i\] = a_i.
    fn promote_to_trivial_shares(
        id: Self::PartyID,
        public_values: &[P::ScalarField],
    ) -> Vec<Self::ArithmeticShare>;

    /// Reconstructs a shared point: A = Open(\[A\]).
    async fn open_point(&mut self, a: Self::PointShareG1) -> std::io::Result<P::G1>;

    /// Reconstructs many shared points: A = Open(\[A\]).
    async fn open_point_many(&mut self, a: &[Self::PointShareG1]) -> std::io::Result<Vec<P::G1>>;

    /// Reconstructs many shared values: a = Open(\[a\]).
    async fn open_many(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> std::io::Result<Vec<P::ScalarField>>;

    /// This function performs a multiplication directly followed by an opening. This safes one round of communication in some MPC protocols compared to calling `mul` and `open` separately.
    async fn mul_open_many(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> std::io::Result<Vec<P::ScalarField>>;

    /// Computes the inverse of many shared values: \[b\] = \[a\] ^ -1. Requires network communication.
    async fn inv_many(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> std::io::Result<Vec<Self::ArithmeticShare>>;

    /// Computes the inverse of many shared values: \[a\] = \[a\] ^ -1. Requires network communication.
    /// This function ignores the case of one share to be zero and maps it to zero.
    async fn inv_many_in_place(&mut self, a: &mut [Self::ArithmeticShare]) -> std::io::Result<()>;

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
}
