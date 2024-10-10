use std::marker::PhantomData;

use ark_ff::One;
use ark_ff::PrimeField;
use ark_ff::Zero;
use num_bigint::BigUint;

use crate::protocols::rep3::network::Rep3Network;

use super::binary;
use super::conversion;
use super::network::IoContext;
use super::Rep3BigUintShare;
use super::Rep3PrimeFieldShare;

type IoResult<T> = std::io::Result<T>;

pub(super) async fn low_depth_binary_add_mod_p<F: PrimeField, N: Rep3Network>(
    x1: &Rep3BigUintShare<F>,
    x2: &Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
    bitlen: usize,
) -> IoResult<Rep3BigUintShare<F>> {
    let x = low_depth_binary_add(x1, x2, io_context, bitlen).await?;
    low_depth_sub_p_cmux::<F, N>(&x, io_context, bitlen + 1).await
}

async fn low_depth_binary_add<F: PrimeField, N: Rep3Network>(
    x1: &Rep3BigUintShare<F>,
    x2: &Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
    bitlen: usize,
) -> IoResult<Rep3BigUintShare<F>> {
    // Add x1 + x2 via a packed Kogge-Stone adder
    let p = x1 ^ x2;
    let g = binary::and(x1, x2, io_context).await?;
    kogge_stone_inner(&p, &g, io_context, bitlen).await
}

async fn kogge_stone_inner<F: PrimeField, N: Rep3Network>(
    p: &Rep3BigUintShare<F>,
    g: &Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
    bitlen: usize,
) -> IoResult<Rep3BigUintShare<F>> {
    let d = ceil_log2(bitlen);
    let s_ = p.to_owned();
    let mut p = p.to_owned();
    let mut g = g.to_owned();
    for i in 0..d {
        let shift = 1 << i;
        let mut p_ = p.to_owned();
        let mut g_ = g.to_owned();
        let mask = (BigUint::from(1u64) << (bitlen - shift)) - BigUint::one();
        p_ &= &mask;
        g_ &= &mask;
        let p_shift = &p >> shift;

        // TODO: Make and more communication efficient, ATM we send the full element for each level, even though they reduce in size
        // maybe just input the mask into AND?
        let (r1, r2) = and_twice(&p_shift, &g_, &p_, io_context, bitlen - shift).await?;
        p = r2 << shift;
        g ^= &(r1 << shift);
    }
    g <<= 1;
    g ^= &s_;
    Ok(g)
}

async fn low_depth_sub_p_cmux<F: PrimeField, N: Rep3Network>(
    x: &Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
    bitlen: usize,
) -> IoResult<Rep3BigUintShare<F>> {
    let original_bitlen = bitlen - 1; // before the potential overflow after an addition
    let mask = (BigUint::from(1u64) << original_bitlen) - BigUint::one();
    let x_msb = x >> original_bitlen;
    let x = x & &mask;
    let mut y = low_depth_binary_sub_p::<F, N>(&x, io_context, bitlen).await?;
    let y_msb = &y >> (bitlen);
    y &= &mask;

    // Spread the ov share to the whole biguint
    let ov_a = (x_msb.a.iter_u64_digits().next().unwrap_or_default()
        ^ y_msb.a.iter_u64_digits().next().unwrap_or_default())
        & 1;
    let ov_b = (x_msb.b.iter_u64_digits().next().unwrap_or_default()
        ^ y_msb.b.iter_u64_digits().next().unwrap_or_default())
        & 1;

    let ov_a = if ov_a == 1 {
        mask.to_owned()
    } else {
        BigUint::zero()
    };
    let ov_b = if ov_b == 1 { mask } else { BigUint::zero() };
    let ov = Rep3BigUintShare::<F>::new(ov_a, ov_b);

    // one big multiplexer
    let res = binary::cmux(&ov, &y, &x, io_context).await?;
    Ok(res)
}

// Calculates 2^k + x1 - x2
async fn low_depth_binary_sub<F: PrimeField, N: Rep3Network>(
    x1: &Rep3BigUintShare<F>,
    x2: &Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
    bitlen: usize,
) -> IoResult<Rep3BigUintShare<F>> {
    // Let x2' = be the bit_not of x2
    // Add x1 + x2' via a packed Kogge-Stone adder, where carry_in = 1
    // This is equivalent to x1 - x2 = x1 + two's complement of x2
    let mask = (BigUint::from(1u64) << bitlen) - BigUint::one();
    // bitnot of x2
    let x2 = binary::xor_public(x2, &mask, io_context.id);
    // Now start the Kogge-Stone adder
    let p = x1 ^ &x2;
    let mut g = binary::and(x1, &x2, io_context).await?;
    // Since carry_in = 1, we need to XOR the LSB of x1 and x2 to g (i.e., xor the LSB of p)
    g ^= &(&p & &BigUint::one());

    let res = kogge_stone_inner(&p, &g, io_context, bitlen).await?;
    let res = binary::xor_public(&res, &BigUint::one(), io_context.id); // cin=1
    Ok(res)
}

fn ceil_log2(x: usize) -> usize {
    let mut y = 0;
    let mut x = x - 1;
    while x > 0 {
        x >>= 1;
        y += 1;
    }
    y
}

async fn and_twice<F: PrimeField, N: Rep3Network>(
    a: &Rep3BigUintShare<F>,
    b1: &Rep3BigUintShare<F>,
    b2: &Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
    bitlen: usize,
) -> IoResult<(Rep3BigUintShare<F>, Rep3BigUintShare<F>)> {
    debug_assert!(a.a.bits() <= bitlen as u64);
    debug_assert!(b1.a.bits() <= bitlen as u64);
    debug_assert!(b2.a.bits() <= bitlen as u64);
    let (mut mask1, mask_b) = io_context.rngs.rand.random_biguint(bitlen);
    mask1 ^= mask_b;

    let (mut mask2, mask_b) = io_context.rngs.rand.random_biguint(bitlen);
    mask2 ^= mask_b;

    let local_a1 = (b1 & a) ^ mask1;
    let local_a2 = (a & b2) ^ mask2;
    io_context.network.send_next(local_a1.to_owned()).await?;
    io_context.network.send_next(local_a2.to_owned()).await?;
    let local_b1 = io_context.network.recv_prev().await?;
    let local_b2 = io_context.network.recv_prev().await?;

    let r1 = Rep3BigUintShare {
        a: local_a1,
        b: local_b1,
        phantom: PhantomData,
    };
    let r2 = Rep3BigUintShare {
        a: local_a2,
        b: local_b2,
        phantom: PhantomData,
    };

    Ok((r1, r2))
}

async fn low_depth_binary_sub_p<F: PrimeField, N: Rep3Network>(
    x: &Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
    bitlen: usize,
) -> IoResult<Rep3BigUintShare<F>> {
    let p_ = (BigUint::from(1u64) << bitlen) - F::MODULUS.into();

    // Add x1 + p_ via a packed Kogge-Stone adder
    let g = x & &p_;
    let p = binary::xor_public(x, &p_, io_context.id);
    kogge_stone_inner(&p, &g, io_context, bitlen).await
}

/// Computes a binary circuit to compare two shared values \[x\] > \[y\]. Thus, the inputs x and y are transformed from arithmetic to binary sharings using [Rep3Protocol::a2b] first. The output is a binary sharing of one bit.
pub(crate) async fn unsigned_ge<F: PrimeField, N: Rep3Network>(
    x: Rep3PrimeFieldShare<F>,
    y: Rep3PrimeFieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3BigUintShare<F>> {
    let a_bits = conversion::a2b(x, io_context).await?;
    let b_bits = conversion::a2b(y, io_context).await?;
    let diff =
        low_depth_binary_sub(&a_bits, &b_bits, io_context, F::MODULUS_BIT_SIZE as usize).await?;

    Ok(&(&diff >> F::MODULUS_BIT_SIZE as usize) & &BigUint::one())
}

/// Computes a binary circuit to compare the shared value y to the public value x, i.e., x > \[y\]. Thus, the input y is transformed from arithmetic to binary sharings using [Rep3Protocol::a2b] first. The output is a binary sharing of one bit.
pub(crate) async fn unsigned_ge_const_lhs<F: PrimeField, N: Rep3Network>(
    x: F,
    y: Rep3PrimeFieldShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3BigUintShare<F>> {
    let b_bits = conversion::a2b(y, io_context).await?;
    let diff = low_depth_binary_sub_from_const(&x.into(), &b_bits, io_context).await?;

    Ok(&(&diff >> F::MODULUS_BIT_SIZE as usize) & &BigUint::one())
}

/// Computes a binary circuit to compare the shared value x to the public value y, i.e., \[x\] > y. Thus, the input x is transformed from arithmetic to binary sharings using [Rep3Protocol::a2b] first. The output is a binary sharing of one bit.
pub(crate) async fn unsigned_ge_const_rhs<F: PrimeField, N: Rep3Network>(
    x: Rep3PrimeFieldShare<F>,
    y: F,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3BigUintShare<F>> {
    let a_bits = conversion::a2b(x, io_context).await?;
    let diff = low_depth_binary_sub_by_const(&a_bits, &y.into(), io_context).await?;

    Ok(&(&diff >> F::MODULUS_BIT_SIZE as usize) & &BigUint::one())
}

// Calculates 2^k + x1 - x2
async fn low_depth_binary_sub_by_const<F: PrimeField, N: Rep3Network>(
    x1: &Rep3BigUintShare<F>,
    x2: &BigUint,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3BigUintShare<F>> {
    // two's complement
    let x2_ = (BigUint::from(1u64) << F::MODULUS_BIT_SIZE as usize) - x2;

    // Add x1 + x2_ via a packed Kogge-Stone adder
    let p = binary::xor_public(x1, &x2_, io_context.id);
    let g = x1 & &x2_;

    let res = kogge_stone_inner(&p, &g, io_context, F::MODULUS_BIT_SIZE as usize).await?;
    Ok(res)
}

// Calculates 2^k + x1 - x2
async fn low_depth_binary_sub_from_const<F: PrimeField, N: Rep3Network>(
    x1: &BigUint,
    x2: &Rep3BigUintShare<F>,
    io_context: &mut IoContext<N>,
) -> IoResult<Rep3BigUintShare<F>> {
    // Let x2' = be the bit_not of x2
    // Add x1 + x2' via a packed Kogge-Stone adder, where carry_in = 1
    // This is equivalent to x1 - x2 = x1 + two's complement of x2
    let mask = (BigUint::from(1u64) << F::MODULUS_BIT_SIZE as usize) - BigUint::one();
    // bitnot of x2
    let x2 = binary::xor_public(x2, &mask, io_context.id);
    // Now start the Kogge-Stone adder
    let p = binary::xor_public(&x2, x1, io_context.id);
    let mut g = &x2 & x1;
    // Since carry_in = 1, we need to XOR the LSB of x1 and x2 to g (i.e., xor the LSB of p)
    g ^= &p & &BigUint::one();

    let res = kogge_stone_inner(&p, &g, io_context, F::MODULUS_BIT_SIZE as usize).await?;
    let res = binary::xor_public(&res, &BigUint::one(), io_context.id);
    Ok(res)
}
