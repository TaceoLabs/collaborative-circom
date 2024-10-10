use std::{fmt, io};

use ark_ff::PrimeField;
use mpc_core::lut::LookupTableProvider;

pub(super) mod plain;
pub(super) mod rep3;

/// A trait representing the MPC operations required for extending the secret-shared Noir witness in MPC.
/// The operations are generic over public and private (i.e., secret-shared) inputs.
pub trait NoirWitnessExtensionProtocol<F: PrimeField> {
    type Lookup: LookupTableProvider<F>;
    type ArithmeticShare: Clone;
    /// A type representing the values encountered during Circom compilation. It should at least contain public field elements and shared values.
    type AcvmType: Clone
        + Default
        + fmt::Debug
        + fmt::Display
        + From<Self::ArithmeticShare>
        + From<F>;

    /// Returns F::zero() as a ACVM-type. The default implementation uses the `Default` trait. If `Default` does not return 0, this function has to be overwritten.
    fn public_zero() -> Self::AcvmType {
        Self::AcvmType::default()
    }

    /// Checks whether an ACVM-type is public zero.
    fn is_public_zero(a: &Self::AcvmType) -> bool;

    /// Checks whether an ACVM-type is public one.
    fn is_public_one(a: &Self::AcvmType) -> bool;

    /// Adds a public value to an ACVM-type in place: *\[target\] += public
    fn acvm_add_assign_with_public(&mut self, public: F, target: &mut Self::AcvmType);

    /// Multiply an ACVM-types with a public value: \[c\] = public * \[secret\].
    fn acvm_mul_with_public(&mut self, public: F, secret: Self::AcvmType) -> Self::AcvmType;

    /// Multiply an ACVM-types with a public value and add_assign with result: \[result\] += q_l * \[w_l\].
    fn solve_linear_term(&mut self, q_l: F, w_l: Self::AcvmType, result: &mut Self::AcvmType);

    /// Multiply two acvm-types and a public value and stores them at target: \[*result\] = c * \[lhs\] * \[rhs\].
    fn solve_mul_term(
        &mut self,
        c: F,
        lhs: Self::AcvmType,
        rhs: Self::AcvmType,
        target: &mut Self::AcvmType,
    ) -> io::Result<()>;

    /// Solves the equation \[q_l\] * w_l + \[c\] = 0, by computing \[-c\]/\[q_l\] and returning the result.
    fn solve_equation(
        &mut self,
        q_l: Self::AcvmType,
        c: Self::AcvmType,
    ) -> eyre::Result<Self::AcvmType>;

    /// Initializes a new LUT from the provided values. The index shall be the order
    /// of the values in the `Vec`. This is wrapper around the method from the [`LookupTableProvider`] as
    /// we create the table from either public or shared values.
    fn init_lut_by_acvm_type(
        &mut self,
        values: Vec<Self::AcvmType>,
    ) -> <Self::Lookup as LookupTableProvider<F>>::SecretSharedMap;

    /// Wrapper around reading from a LUT by the [`Self::AcvmType`] as this can either be a
    /// public or a shared read.
    fn read_lut_by_acvm_type(
        &mut self,
        index: &Self::AcvmType,
        lut: &<Self::Lookup as LookupTableProvider<F>>::SecretSharedMap,
    ) -> io::Result<Self::AcvmType>;

    /// Wrapper around writing a value to a LUT. The index and the value can be shared or public.
    fn write_lut_by_acvm_type(
        &mut self,
        index: Self::AcvmType,
        value: Self::AcvmType,
        lut: &mut <Self::Lookup as LookupTableProvider<F>>::SecretSharedMap,
    ) -> io::Result<()>;

    /// Returns true if the value is shared
    fn is_shared(a: &Self::AcvmType) -> bool;

    /// Returns the share if the value is shared
    fn get_shared(a: &Self::AcvmType) -> Option<Self::ArithmeticShare>;

    // TODO do we want this here?
    fn open_many(&mut self, a: &[Self::ArithmeticShare]) -> io::Result<Vec<F>>;
}
