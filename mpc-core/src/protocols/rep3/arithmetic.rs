//! Arithmetic
//!
//! This module contains operations with arithmetic shares

use core::panic;
use num_traits::cast::ToPrimitive;

use ark_ff::PrimeField;
use itertools::{izip, Itertools};
use num_bigint::BigUint;
use num_traits::Zero;
use types::Rep3PrimeFieldShare;

use crate::protocols::rep3::{detail, id::PartyID, network::Rep3Network};
use rayon::prelude::*;

use super::{
    binary, conversion, network::IoContext, rngs::Rep3CorrelatedRng, IoResult, Rep3BigUintShare,
};

/// Type alias for a [`Rep3PrimeFieldShare`]
pub type FieldShare<F> = Rep3PrimeFieldShare<F>;
/// Type alias for a [`Rep3BigUintShare`]
pub type BinaryShare<F> = Rep3BigUintShare<F>;

mod ops;
pub(super) mod types;

/// Performs addition between two shared values.
pub fn add<F: PrimeField>(a: FieldShare<F>, b: FieldShare<F>) -> FieldShare<F> {
    a + b
}

/// Performs addition between two shared values in place
pub fn add_assign<F: PrimeField>(shared: &mut FieldShare<F>, b: FieldShare<F>) {
    *shared += b;
}

/// Performs addition between a shared value and a public value.
pub fn add_public<F: PrimeField>(shared: FieldShare<F>, public: F, id: PartyID) -> FieldShare<F> {
    let mut res = shared;
    match id {
        PartyID::ID0 => res.a += public,
        PartyID::ID1 => res.b += public,
        PartyID::ID2 => {}
    }
    res
}

/// Performs addition between a shared value and a public value in place.
pub fn add_assign_public<F: PrimeField>(shared: &mut FieldShare<F>, public: F, id: PartyID) {
    match id {
        PartyID::ID0 => shared.a += public,
        PartyID::ID1 => shared.b += public,
        PartyID::ID2 => {}
    }
}

/// Performs element-wise addition of two vectors of shared values in place.
pub fn add_vec_assign<F: PrimeField>(lhs: &mut [FieldShare<F>], rhs: &[FieldShare<F>]) {
    for (a, b) in izip!(lhs.iter_mut(), rhs.iter()) {
        *a += b;
    }
}

/// Performs subtraction between two shared values, returning a - b.
pub fn sub<F: PrimeField>(a: FieldShare<F>, b: FieldShare<F>) -> FieldShare<F> {
    a - b
}

/// Performs subtraction between two shared values in place.
pub fn sub_assign<F: PrimeField>(shared: &mut FieldShare<F>, b: FieldShare<F>) {
    *shared -= b;
}

/// Performs subtraction between a shared value and a public value, returning shared - public.
pub fn sub_shared_by_public<F: PrimeField>(
    shared: FieldShare<F>,
    public: F,
    id: PartyID,
) -> FieldShare<F> {
    add_public(shared, -public, id)
}

/// Performs subtraction between a shared value and a public value, returning public - shared.
pub fn sub_public_by_shared<F: PrimeField>(
    public: F,
    shared: FieldShare<F>,
    id: PartyID,
) -> FieldShare<F> {
    add_public(-shared, public, id)
}

/// Performs multiplication of two shared values.
pub async fn mul<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let local_a = a * b + io_context.rngs.rand.masking_field_element::<F>();
    let local_b = io_context.network.reshare(local_a).await?;
    Ok(FieldShare {
        a: local_a,
        b: local_b,
    })
}

/// Performs multiplication of a shared value and a public value.
pub fn mul_public<F: PrimeField>(shared: FieldShare<F>, public: F) -> FieldShare<F> {
    shared * public
}

/// Performs multiplication of a shared value and a public value.
pub fn mul_assign_public<F: PrimeField>(shared: &mut FieldShare<F>, public: F) {
    *shared *= public;
}

/// Performs element-wise multiplication of two vectors of shared values. *DOES NOT PERFORM RESHARE*
///
/// # Security
/// If you want to perform additional non-linear operations on the result of this function,
/// you *MUST* call [`io_mul_vec`] first. Only then, a reshare is performed.
pub fn local_mul_vec<F: PrimeField>(
    lhs: &[FieldShare<F>],
    rhs: &[FieldShare<F>],
    rngs: &mut Rep3CorrelatedRng,
) -> Vec<F> {
    //squeeze all random elements at once in beginning for determinismus
    let masking_fes = rngs.rand.masking_field_elements_vec::<F>(lhs.len());

    lhs.par_iter()
        .zip_eq(rhs.par_iter())
        .zip_eq(masking_fes.par_iter())
        .with_min_len(1024)
        .map(|((lhs, rhs), masking)| lhs.a * rhs.a + lhs.a * rhs.b + lhs.b * rhs.a + masking)
        .collect()
}

/// Performs a reshare on all shares in the vector.
pub async fn io_mul_vec<F: PrimeField, N: Rep3Network>(
    local_a: Vec<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    let local_b = io_context.network.reshare_many(&local_a).await?;
    if local_b.len() != local_a.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of mul_vec in MPC: Invalid number of elements received",
        ));
    }
    Ok(izip!(local_a, local_b)
        .map(|(a, b)| FieldShare::new(a, b))
        .collect())
}

/// Performs element-wise multiplication of two vectors of shared values.
///
/// Use this function for small vecs. For large vecs see [`local_mul_vec`] and [`io_mul_vec`]
pub async fn mul_vec<F: PrimeField, N: Rep3Network>(
    lhs: &[FieldShare<F>],
    rhs: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    // do not use local_mul_vec here!!! We are async, this means we
    // run on a tokio runtime. local_mul_vec uses rayon and starves the
    // runtime. This method is for small multiplications of vecs.
    // If you want a larger one use local_mul_vec and then io_mul_vec.
    debug_assert_eq!(lhs.len(), rhs.len());
    let local_a = izip!(lhs.iter(), rhs.iter())
        .map(|(lhs, rhs)| {
            lhs.a * rhs.a
                + lhs.a * rhs.b
                + lhs.b * rhs.a
                + io_context.rngs.rand.masking_field_element::<F>()
        })
        .collect_vec();
    io_mul_vec(local_a, io_context).await
}

/// Performs division of two shared values, returning a / b.
pub async fn div<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    mul(a, inv(b, io_context).await?, io_context).await
}

/// Performs division of a shared value by a public value, returning shared / public.
pub fn div_shared_by_public<F: PrimeField>(
    shared: FieldShare<F>,
    public: F,
) -> eyre::Result<FieldShare<F>> {
    if public.is_zero() {
        eyre::bail!("Cannot invert zero");
    }
    let b_inv = public.inverse().unwrap();
    Ok(mul_public(shared, b_inv))
}

/// Performs division of a public value by a shared value, returning public / shared.
pub async fn div_public_by_shared<F: PrimeField, N: Rep3Network>(
    public: F,
    shared: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    Ok(mul_public(inv(shared, io_context).await?, public))
}

/// Negates a shared value.
pub fn neg<F: PrimeField>(a: FieldShare<F>) -> FieldShare<F> {
    -a
}

/// Computes the inverse of a shared value.
pub async fn inv<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let r = FieldShare::rand(io_context);
    let y = mul_open(a, r, io_context).await?;
    if y.is_zero() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of inverse in MPC: cannot compute inverse of zero",
        ));
    }
    let y_inv = y
        .inverse()
        .expect("we checked if y is zero. Must be possible to invert.");
    Ok(r * y_inv)
}

/// Computes the inverse of a vector of shared field elements
pub async fn inv_vec<F: PrimeField, N: Rep3Network>(
    a: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<FieldShare<F>>> {
    let r = (0..a.len())
        .map(|_| FieldShare::rand(io_context))
        .collect_vec();
    let y = mul_open_vec(a, &r, io_context).await?;
    if y.iter().any(|y| y.is_zero()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of inverse in MPC: cannot compute inverse of zero",
        ));
    }

    // we can unwrap as we checked that none of the y is zero
    Ok(izip!(r, y).map(|(r, y)| r * y.inverse().unwrap()).collect())
}

/// Performs the opening of a shared value and returns the equivalent public value.
pub async fn open<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<F> {
    let c = io_context.network.reshare(a.b).await?;
    Ok(a.a + a.b + c)
}

/// Performs the opening of a shared value and returns the equivalent public value.
pub async fn open_vec<F: PrimeField, N: Rep3Network>(
    a: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<F>> {
    // TODO think about something better... it is not so bad
    // because we use it exactly once in PLONK where we do it for 4
    // shares..
    let (a, b) = a
        .iter()
        .map(|share| (share.a, share.b))
        .collect::<(Vec<F>, Vec<F>)>();
    let c = io_context.network.reshare_many(&b).await?;
    Ok(izip!(a, b, c).map(|(a, b, c)| a + b + c).collect_vec())
}

/// Computes a CMUX: If cond is 1, returns truthy, otherwise returns falsy.
/// Implementations should not overwrite this method.
pub async fn cmux<F: PrimeField, N: Rep3Network>(
    cond: FieldShare<F>,
    truthy: FieldShare<F>,
    falsy: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let b_min_a = sub(truthy, falsy);
    let d = mul(cond, b_min_a, io_context).await?;
    Ok(add(falsy, d))
}

/// Convenience method for \[a\] + \[b\] * c
pub fn add_mul_public<F: PrimeField>(a: FieldShare<F>, b: FieldShare<F>, c: F) -> FieldShare<F> {
    add(a, mul_public(b, c))
}

/// Convenience method for \[a\] + \[b\] * \[c\]
pub async fn add_mul<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    c: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let mul = mul(c, b, io_context).await?;
    Ok(add(a, mul))
}

/// Transforms a public value into a shared value: \[a\] = a.
pub fn promote_to_trivial_share<F: PrimeField>(id: PartyID, public_value: F) -> FieldShare<F> {
    match id {
        PartyID::ID0 => Rep3PrimeFieldShare::new(public_value, F::zero()),
        PartyID::ID1 => Rep3PrimeFieldShare::new(F::zero(), public_value),
        PartyID::ID2 => Rep3PrimeFieldShare::zero_share(),
    }
}

/// This function performs a multiplication directly followed by an opening. This safes one round of communication in some MPC protocols compared to calling `mul` and `open` separately.
pub async fn mul_open<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<F> {
    let a = a * b + io_context.rngs.rand.masking_field_element::<F>();
    let (b, c) = io_context.network.broadcast(a).await?;
    Ok(a + b + c)
}

/// This function performs a multiplication directly followed by an opening. This safes one round of communication in some MPC protocols compared to calling `mul` and `open` separately.
pub async fn mul_open_vec<F: PrimeField, N: Rep3Network>(
    a: &[FieldShare<F>],
    b: &[FieldShare<F>],
    io_context: &mut IoContext<N>,
) -> IoResult<Vec<F>> {
    let mut a = izip!(a, b)
        .map(|(a, b)| a * b + io_context.rngs.rand.masking_field_element::<F>())
        .collect_vec();
    let (b, c) = io_context.network.broadcast_many(&a).await?;
    izip!(a.iter_mut(), b, c).for_each(|(a, b, c)| *a += b + c);
    Ok(a)
}

/// Generate a random [`FieldShare`].
pub fn rand<F: PrimeField, N: Rep3Network>(io_context: &mut IoContext<N>) -> FieldShare<F> {
    let (a, b) = io_context.random_fes();
    FieldShare::new(a, b)
}

/// Computes the square root of a shared value.
pub async fn sqrt<F: PrimeField, N: Rep3Network>(
    share: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let r_squ = rand(io_context);
    let r_inv = rand(io_context);

    let rr = mul(r_squ, r_squ, io_context).await?;

    // parallel mul of rr with a and r_squ with r_inv
    let lhs = vec![rr, r_squ];
    let rhs = vec![share, r_inv];
    let mul = mul_vec(&lhs, &rhs, io_context).await?;

    // Open mul
    io_context
        .network
        .send_next_many(&mul.iter().map(|s| s.b.to_owned()).collect_vec())
        .await?;
    let c = io_context.network.recv_prev_many::<F>().await?;
    if c.len() != 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of square root in MPC: invalid number of elements received",
        ));
    }
    let y_sq = (mul[0].a + mul[0].b + c[0]).sqrt();
    let y_inv = mul[1].a + mul[1].b + c[1];

    // postprocess the square and inverse
    let y_sq = match y_sq {
        Some(y) => y,
        None => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "During execution of square root in MPC: cannot compute square root",
            ));
        }
    };

    if y_inv.is_zero() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "During execution of square root in MPC: cannot compute inverse of zero",
        ));
    }
    let y_inv = y_inv.inverse().unwrap();

    let r_squ_inv = r_inv * y_inv;
    let a_sqrt = r_squ_inv * y_sq;

    Ok(a_sqrt)
}

/// Performs a pow operation using a shared value as base and a public value as exponent.
pub async fn pow_public<F: PrimeField, N: Rep3Network>(
    shared: FieldShare<F>,
    public: F,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3PrimeFieldShare<F>> {
    // TODO: are negative exponents allowed in circom?
    let mut res = promote_to_trivial_share(io_context.id, F::one());
    let mut public: BigUint = public.into_bigint().into();
    let mut shared: FieldShare<F> = shared;
    while !public.is_zero() {
        if public.bit(0) {
            public -= 1u64;
            res = mul(res, shared, io_context).await?;
        }
        shared = mul(shared, shared, io_context).await?;
        public >>= 1;
    }
    mul(res, shared, io_context).await
}

/// Returns 1 if lhs < rhs and 0 otherwise. Checks if one shared value is less than another shared value. The result is a shared value that has value 1 if the first shared value is less than the second shared value and 0 otherwise.
pub async fn lt<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // a < b is equivalent to !(a >= b)
    let tmp = ge(lhs, rhs, io_context).await?;
    Ok(sub_public_by_shared(F::one(), tmp, io_context.id))
}

/// Returns 1 if lhs < rhs and 0 otherwise. Checks if a shared value is less than the public value. The result is a shared value that has value 1 if the shared value is less than the public value and 0 otherwise.
pub async fn lt_public<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // a < b is equivalent to !(a >= b)
    let tmp = ge_public(lhs, rhs, io_context).await?;
    Ok(sub_public_by_shared(F::one(), tmp, io_context.id))
}

/// Returns 1 if lhs <= rhs and 0 otherwise. Checks if one shared value is less than or equal to another shared value. The result is a shared value that has value 1 if the first shared value is less than or equal to the second shared value and 0 otherwise.
pub async fn le<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // a <= b is equivalent to b >= a
    ge(rhs, lhs, io_context).await
}

/// Returns 1 if lhs <= rhs and 0 otherwise. Checks if a shared value is less than or equal to a public value. The result is a shared value that has value 1 if the shared value is less than or equal to the public value and 0 otherwise.
pub async fn le_public<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let res = detail::unsigned_ge_const_lhs(rhs, lhs, io_context).await?;
    conversion::bit_inject(&res, io_context).await
}

/// Returns 1 if lhs > rhs and 0 otherwise. Checks if one shared value is greater than another shared value. The result is a shared value that has value 1 if the first shared value is greater than the second shared value and 0 otherwise.
pub async fn gt<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // a > b is equivalent to !(a <= b)
    let tmp = le(lhs, rhs, io_context).await?;
    Ok(sub_public_by_shared(F::one(), tmp, io_context.id))
}

/// Returns 1 if lhs > rhs and 0 otherwise. Checks if a shared value is greater than the public value. The result is a shared value that has value 1 if the shared value is greater than the public value and 0 otherwise.
pub async fn gt_public<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    // a > b is equivalent to !(a <= b)
    let tmp = le_public(lhs, rhs, io_context).await?;
    Ok(sub_public_by_shared(F::one(), tmp, io_context.id))
}

/// Returns 1 if lhs >= rhs and 0 otherwise. Checks if one shared value is greater than or equal to another shared value. The result is a shared value that has value 1 if the first shared value is greater than or equal to the second shared value and 0 otherwise.
pub async fn ge<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let res = detail::unsigned_ge(lhs, rhs, io_context).await?;
    conversion::bit_inject(&res, io_context).await
}

/// Returns 1 if lhs >= rhs and 0 otherwise. Checks if a shared value is greater than or equal to a public value. The result is a shared value that has value 1 if the shared value is greater than or equal to the public value and 0 otherwise.
pub async fn ge_public<F: PrimeField, N: Rep3Network>(
    lhs: FieldShare<F>,
    rhs: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let res = detail::unsigned_ge_const_rhs(lhs, rhs, io_context).await?;
    conversion::bit_inject(&res, io_context).await
}

//TODO FN REMARK - I think we can skip the bit_inject.
//Circom has dedicated op codes for bool ops so we would know
//for bool_and/bool_or etc that we are a boolean value (and therefore
//bit len 1).
//
//We leave it like that and come back to that later. Maybe it doesn't matter...

/// Checks if two shared values are equal. The result is a shared value that has value 1 if the two shared values are equal and 0 otherwise.
pub async fn eq<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let diff = sub(a, b);
    let bits = conversion::a2b(diff, io_context).await?;
    let is_zero = binary::is_zero(&bits, io_context).await?;
    let res = conversion::bit_inject(&is_zero, io_context).await?;
    Ok(res)
}

/// Checks if a shared value is equal to a public value. The result is a shared value that has value 1 if the two values are equal and 0 otherwise.
pub async fn eq_public<F: PrimeField, N: Rep3Network>(
    shared: FieldShare<F>,
    public: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let public = promote_to_trivial_share(io_context.id, public);
    eq(shared, public, io_context).await
}

/// Same as eq but without using bit_inject on the result. Checks whether two prime field shares are equal and return a binary share of 0 or 1. 1 means they are equal.
pub async fn eq_bit<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let diff = sub(a, b);
    let bits = conversion::a2b(diff, io_context).await?;
    let is_zero = binary::is_zero(&bits, io_context).await?;
    Ok(is_zero)
}

/// Checks if two shared values are not equal. The result is a shared value that has value 1 if the two values are not equal and 0 otherwise.
pub async fn neq<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    b: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let eq = eq(a, b, io_context).await?;
    Ok(sub_public_by_shared(F::one(), eq, io_context.id))
}

/// Checks if a shared value is not equal to a public value. The result is a shared value that has value 1 if the two values are not equal and 0 otherwise.
pub async fn neq_public<F: PrimeField, N: Rep3Network>(
    shared: FieldShare<F>,
    public: F,
    io_context: &mut IoContext<N>,
) -> IoResult<FieldShare<F>> {
    let public = promote_to_trivial_share(io_context.id, public);
    neq(shared, public, io_context).await
}

/// Outputs whether a shared value is zero (true) or not (false).
pub async fn is_zero<F: PrimeField, N: Rep3Network>(
    a: FieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<bool> {
    let zero_share = FieldShare::default();
    let res = eq(zero_share, a, io_context).await?;
    let x = open(res, io_context).await?;
    Ok(x.is_one())
}

/// Computes `shared*2^public`. This is the same as `shared << public`.
///
/// #Panics
/// If public is larger than the bit size of the modulus of the underlying `PrimeField`.
pub fn pow_2_public<F: PrimeField>(shared: FieldShare<F>, public: F) -> FieldShare<F> {
    if public.is_zero() {
        shared
    } else {
        let shift: BigUint = public.into();
        let shift = shift.to_u32().expect("can cast shift operand to u32");
        if shift >= F::MODULUS_BIT_SIZE {
            panic!(
                "Expected left shift to be maximal {}, but was {}",
                F::MODULUS_BIT_SIZE,
                shift
            );
        } else {
            mul_public(shared, F::from(2u64).pow(public.into_bigint()))
        }
    }
}

/// computes XOR using arithmetic operations, only valid when x and y are known to be 0 or 1.
pub(crate) async fn arithmetic_xor<F: PrimeField, N: Rep3Network>(
    x: Rep3PrimeFieldShare<F>,
    y: Rep3PrimeFieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3PrimeFieldShare<F>> {
    let d = mul(x, y, io_context).await?;
    let d = add(d, d);
    let e = add(x, y);
    let d = sub(e, d);
    Ok(d)
}
