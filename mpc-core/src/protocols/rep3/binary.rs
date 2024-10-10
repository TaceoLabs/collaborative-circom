//! Binary
//!
//! This module contains operations with binary shares

use ark_ff::{One, PrimeField};
use futures::{stream::FuturesOrdered, StreamExt};
use itertools::izip;
use num_bigint::BigUint;
use types::Rep3BigUintShare;

use crate::protocols::rep3::{
    arithmetic::{self},
    conversion,
    id::PartyID,
    network::Rep3Network,
};

use super::{network::IoContext, Rep3PrimeFieldShare};
use num_traits::cast::ToPrimitive;

mod ops;
pub(super) mod types;

type ArithmeticShare<F> = Rep3PrimeFieldShare<F>;
type BinaryShare<F> = Rep3BigUintShare<F>;
type IoResult<T> = std::io::Result<T>;

/// Performs a bitwise XOR operation on two shared values.
pub fn xor<F: PrimeField>(a: &BinaryShare<F>, b: &BinaryShare<F>) -> BinaryShare<F> {
    a ^ b
}

/// Performs a bitwise XOR operation on a shared value and a public value.
pub fn xor_public<F: PrimeField>(
    shared: &BinaryShare<F>,
    public: &BigUint,
    id: PartyID,
) -> BinaryShare<F> {
    let mut res = shared.to_owned();
    match id {
        PartyID::ID0 => res.a ^= public,
        PartyID::ID1 => res.b ^= public,
        PartyID::ID2 => {}
    }
    res
}

/// Performs a bitwise OR operation on two shared values.
pub async fn or<F: PrimeField, N: Rep3Network>(
    a: &BinaryShare<F>,
    b: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let xor = a ^ b;
    let and = and(a, b, io_context).await?;
    Ok(xor ^ and)
}

/// Performs a bitwise OR operation on a shared value and a public value.
pub fn or_public<F: PrimeField>(
    shared: &BinaryShare<F>,
    public: &BigUint,
    id: PartyID,
) -> BinaryShare<F> {
    let tmp = shared & public;
    let xor = xor_public(shared, public, id);
    xor ^ tmp
}

/// Performs a bitwise AND operation on two shared values.
pub async fn and<F: PrimeField, N: Rep3Network>(
    a: &BinaryShare<F>,
    b: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    debug_assert!(a.a.bits() <= u64::from(F::MODULUS_BIT_SIZE));
    debug_assert!(b.a.bits() <= u64::from(F::MODULUS_BIT_SIZE));
    let (mut mask, mask_b) = io_context
        .rngs
        .rand
        .random_biguint(usize::try_from(F::MODULUS_BIT_SIZE).expect("u32 fits into usize"));
    mask ^= mask_b;
    let local_a = (a & b) ^ mask;
    let local_b = io_context.network.reshare(local_a.clone()).await?;
    Ok(BinaryShare::new(local_a, local_b))
}

/// Performs a bitwise AND operation on a shared value and a public value.
pub fn and_with_public<F: PrimeField>(shared: &BinaryShare<F>, public: &BigUint) -> BinaryShare<F> {
    shared & public
}

/// Shifts a share by a public value `F` to the right.
///
/// # Panics
/// This method panics if `public` is larger than the of bits of
/// the underlying `PrimeField`'s modulus'.
pub fn shift_r_public<F: PrimeField>(shared: &BinaryShare<F>, public: F) -> BinaryShare<F> {
    // some special casing
    if public.is_zero() {
        return shared.to_owned();
    }
    let shift: BigUint = public.into();
    let shift = shift.to_usize().expect("can cast shift operand to usize");
    shared >> shift
}

/// Shifts a share by a public value `F` to the left.
///
/// # Panics
/// This method panics if `public` is larger than the of bits of
/// the underlying `PrimeField`'s modulus'.
pub fn shift_l_public<F: PrimeField>(shared: &BinaryShare<F>, public: F) -> BinaryShare<F> {
    // some special casing
    if public.is_zero() {
        return shared.to_owned();
    }
    let shift: BigUint = public.into();
    let shift = shift.to_usize().expect("can cast shift operand to usize");
    shared << shift
}

/// Shifts a public value `F` by a share to the left.
pub async fn shift_l_public_by_shared<F: PrimeField, N: Rep3Network>(
    public: F,
    shared: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<ArithmeticShare<F>> {
    // This case is equivalent to a*2^b
    // Strategy: limit size of b to k bits
    // bit-decompose b into bits b_i

    // TODO: this sucks... we need something better here...
    let io_0 = io_context.fork().await?;
    let io_1 = io_context.fork().await?;
    let io_2 = io_context.fork().await?;
    let io_3 = io_context.fork().await?;
    let io_4 = io_context.fork().await?;
    let io_5 = io_context.fork().await?;
    let io_6 = io_context.fork().await?;
    let io_7 = io_context.fork().await?;
    let mut contexts = [io_0, io_1, io_2, io_3, io_4, io_5, io_6, io_7];
    let party_id = io_context.id;
    let mut futures_ordered = FuturesOrdered::new();
    for (i, context) in izip!((0..8), contexts.iter_mut()) {
        let bit = Rep3BigUintShare::new(
            (shared.a.clone() >> i) & BigUint::one(),
            (shared.b.clone() >> i) & BigUint::one(),
        );
        futures_ordered.push_back(conversion::b2a_consume(bit, context));
    }
    let mut individual_bit_shares = Vec::with_capacity(8);
    while let Some(result) = futures_ordered.next().await {
        individual_bit_shares.push(result?);
    }
    // v_i = 2^2^i * <b_i> + 1 - <b_i>
    let mut vs: Vec<_> = individual_bit_shares
        .into_iter()
        .enumerate()
        .map(|(i, b_i)| {
            let two = F::from(2u64);
            // i is 8 at most there `as u32` is ok
            let two_to_two_to_i = two.pow([2u64.pow(i as u32)]);
            let v = arithmetic::mul_public(b_i, two_to_two_to_i);
            let v = arithmetic::add_public(v, F::one(), party_id);
            arithmetic::sub(v, b_i)
        })
        .collect();

    // v = \prod v_i
    // TODO: This should be done in a multiplication tree
    let mut v = vs.pop().unwrap();
    for v_i in vs {
        v = arithmetic::mul(v, v_i, io_context).await?;
    }
    // TODO could use try_fold from futures::stream
    // let last = vs.pop().unwrap();
    // let v = self.runtime.block_on(
    //     futures::stream::iter(vs.into_iter().map(|v| Ok(v)))
    //         .try_fold(last, |a, b| async move {
    //             arithmetic::mul(a, b, &mut self.io_context).await
    //         }),
    // )?;
    Ok(arithmetic::mul_public(v, public))
}

//pub async fn and_vec(
//    a: &FieldShareVec<F>,
//    b: &FieldShareVec<F>,
//    io_context: &mut IoContext<N>,
//) -> IoResult<FieldShareVec<F>> {
//    //debug_assert_eq!(a.len(), b.len());
//    let local_a = izip!(a.a.iter(), a.b.iter(), b.a.iter(), b.b.iter())
//        .map(|(aa, ab, ba, bb)| {
//            *aa * ba + *aa * bb + *ab * ba + io_context.rngs.rand.masking_field_element::<F>()
//        })
//        .collect_vec();
//    io_context.network.send_next_many(&local_a)?;
//    let local_b = io_context.network.recv_prev_many()?;
//    if local_b.len() != local_a.len() {
//        return Err(std::io::Error::new(
//            std::io::ErrorKind::InvalidData,
//            "During execution of mul_vec in MPC: Invalid number of elements received",
//        ));
//    }
//    Ok(FieldShareVec::new(local_a, local_b))
//}

/// Performs the opening of a shared value and returns the equivalent public value.
pub async fn open<F: PrimeField, N: Rep3Network>(
    a: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BigUint> {
    let c = io_context.network.reshare(a.b.clone()).await?;
    Ok(&a.a ^ &a.b ^ c)
}

/// Transforms a public value into a shared value: \[a\] = a.
pub fn promote_to_trivial_share<F: PrimeField>(
    id: PartyID,
    public_value: &BigUint,
) -> BinaryShare<F> {
    match id {
        PartyID::ID0 => BinaryShare::new(public_value.to_owned(), BigUint::ZERO),
        PartyID::ID1 => BinaryShare::new(BigUint::ZERO, public_value.to_owned()),
        PartyID::ID2 => BinaryShare::zero_share(),
    }
}

/// Computes a CMUX: If `c` is `1`, returns `x_t`, otherwise returns `x_f`.
pub async fn cmux<F: PrimeField, N: Rep3Network>(
    c: &BinaryShare<F>,
    x_t: &BinaryShare<F>,
    x_f: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let xor = x_f ^ x_t;
    let mut and = and(c, &xor, io_context).await?;
    and ^= x_f;
    Ok(and)
}

//TODO most likely the inputs here are only one bit therefore we
//do not have to perform an or over the whole length of prime field
//but only one bit.
//Do we want that to be configurable? Semms like a waste?
/// Compute a OR tree of the input vec
pub async fn or_tree<F: PrimeField, N: Rep3Network>(
    mut inputs: Vec<BinaryShare<F>>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let mut num = inputs.len();

    tracing::debug!("starting or tree over {} elements", inputs.len());
    while num > 1 {
        tracing::trace!("binary tree still has {} elements", num);
        let mod_ = num & 1;
        num >>= 1;

        let (a_vec, tmp) = inputs.split_at(num);
        let (b_vec, leftover) = tmp.split_at(num);

        let mut res = Vec::with_capacity(num);
        // TODO WE WANT THIS BATCHED!!!
        // THIS IS SUPER BAD
        for (a, b) in izip!(a_vec.iter(), b_vec.iter()) {
            res.push(or(a, b, io_context).await?);
        }

        res.extend_from_slice(leftover);
        inputs = res;

        num += mod_;
    }
    let result = inputs[0].clone();
    tracing::debug!("we did it!");
    Ok(result)
}

/// Computes a binary circuit to check whether the replicated binary-shared input x is zero or not. The output is a binary sharing of one bit.
pub async fn is_zero<F: PrimeField, N: Rep3Network>(
    x: &BinaryShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<BinaryShare<F>> {
    let bit_len = F::MODULUS_BIT_SIZE as usize;
    let mask = (BigUint::from(1u64) << bit_len) - BigUint::one();

    // negate
    let mut x = x ^ &mask;

    // do ands in a tree
    // TODO: Make and tree more communication efficient, ATM we send the full element for each level, even though they halve in size
    let mut len = bit_len;
    while len > 1 {
        if len % 2 == 1 {
            len += 1;
            // pad with a 1 (= 1 xor 1 xor 1) in MSB position
            // since this is publicly known we just set the bit in each party's share and its replication
            x.a.set_bit(len as u64 - 1, true);
            x.b.set_bit(len as u64 - 1, true);
        }
        len /= 2;
        let mask = (BigUint::from(1u64) << len) - BigUint::one();
        let y = &x >> len;
        x = and(&(&x & &mask), &(&y & &mask), io_context).await?;
    }
    // extract LSB
    Ok(x & BigUint::one())
}
