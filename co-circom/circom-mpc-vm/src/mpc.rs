use eyre::Result;
use std::fmt;

use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

pub(crate) mod plain;
pub(crate) mod rep3;

/// This trait represents the operations used during witness extension by the co-circom MPC-VM
pub trait VmCircomWitnessExtension<F: PrimeField> {
    /// The arithemitc share type
    type ArithmeticShare: CanonicalSerialize + CanonicalDeserialize + Clone + Default;
    /// The binary share type
    type BinaryShare;
    /// The VM type
    type VmType: Clone
        + Default
        + fmt::Debug
        + fmt::Display
        + From<F>
        + From<Self::ArithmeticShare>
        + From<Self::BinaryShare>;

    /// Add two VM-types: c = a + b.
    fn add(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Subtract the VM-type b from the VM-type a: c = a - b.
    fn sub(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Multiply two VM-types: c = a * b.
    fn mul(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Divide the VM-type a by the VM-type b: c = a / b. In finite fields, this is equivalent to multiplying a by the inverse of b.
    fn div(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Integer division of the VM-type a by the VM-type b: c = a \ b.
    fn int_div(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Compute c = a ^ b, where a and b ar VM-types.
    fn pow(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Compute the modulo of the VM-type a by the VM-type b: c = a % b.
    fn modulo(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Compute the square root of the VM-type a: c = sqrt(a).
    fn sqrt(&mut self, a: Self::VmType) -> Result<Self::VmType>;

    /// Compute the negation of the VM-type a: c = -a.
    fn neg(&mut self, a: Self::VmType) -> Result<Self::VmType>;

    /// Compute the less than operation of two VM-types: a < b. Outputs 1 if a < b, 0 otherwise.
    fn lt(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Compute the less or equal than operation of two VM-types: a <= b. Outputs 1 if a <= b, 0 otherwise.
    fn le(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Compute the greater than operation of two VM-types: a > b. Outputs 1 if a > b, 0 otherwise.
    fn gt(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Compute the greater or equal than operation of two VM-types: a >= b. Outputs 1 if a >= b, 0 otherwise.
    fn ge(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Compute the equality of two VM-types: a == b. Outputs 1 if a == b, 0 otherwise.
    fn eq(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Compute the inequality of two VM-types: a != b. Outputs 1 if a != b, 0 otherwise.
    fn neq(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Computes the bitwise shift right of the VM-type a by the VM-type b: c = a >> b.
    fn shift_r(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Computes the bitwise shift left of the VM-type a by the VM-type b: c = a << b.
    fn shift_l(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Computes the boolean NOT of the VM-type a, i.e. 1 - a. The value a is expected to either be 0 or 1.
    fn bool_not(&mut self, a: Self::VmType) -> Result<Self::VmType>;

    /// Computes the boolean AND of the VM-types a and b: c = a && b. The values a and b are expected to be either 0 or 1.
    fn bool_and(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Computes the boolean OR of the VM-types a and b: c = a || b. The values a and b are expected to be either 0 or 1.
    fn bool_or(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Computes a CMUX: If cond is 1, returns truthy, otherwise returns falsy.
    fn cmux(
        &mut self,
        cond: Self::VmType,
        truthy: Self::VmType,
        falsy: Self::VmType,
    ) -> Result<Self::VmType>;

    /// Computes the bitwise XOR of the VM-types a and b: c = a ^ b.
    fn bit_xor(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Computes the bitwise OR of the VM-types a and b: c = a | b.
    fn bit_or(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Computes the bitwise AND of the VM-types a and b: c = a & b.
    fn bit_and(&mut self, a: Self::VmType, b: Self::VmType) -> Result<Self::VmType>;

    /// Outputs whether a is zero (true) or not (false). This values is output in plain! Thus, if a is secret shared, the result is opened.
    fn is_zero(&mut self, a: Self::VmType, allow_secret_inputs: bool) -> Result<bool>;

    /// Returns whether the VM-type represents a shared value (true) or a public one (false).
    fn is_shared(&mut self, a: &Self::VmType) -> Result<bool>;

    /// Transforms a public field element into a usize if possible.
    fn to_index(&mut self, a: Self::VmType) -> Result<usize>;

    /// Opens the VM-type a. If a is secret shared, it gets reconstructed.
    fn open(&mut self, a: Self::VmType) -> Result<F>;

    /// Transforms a VM-type into a secret-shared value.
    fn to_share(&mut self, a: Self::VmType) -> Result<Self::ArithmeticShare>;

    /// Returns F::one() as a VM-type.
    fn public_one(&self) -> Self::VmType;

    /// Returns F::zero() as a VM-type. The default implementation uses the `Default` trait. If `Default` does not return 0, this function has to be overwritten.
    fn public_zero(&self) -> Self::VmType;
}
