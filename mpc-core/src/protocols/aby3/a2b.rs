use crate::traits::PrimeFieldMpcProtocol;

use super::{id::PartyID, network::Aby3Network, Aby3PrimeFieldShare, Aby3Protocol, IoResult};
use ark_ff::{One, PrimeField, Zero};
use num_bigint::BigUint;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Aby3BigUintShare {
    pub(crate) a: BigUint,
    pub(crate) b: BigUint,
}

impl Aby3BigUintShare {
    pub(crate) fn new(a: BigUint, b: BigUint) -> Self {
        Self { a, b }
    }

    pub fn get_a(self) -> BigUint {
        self.a
    }

    pub(crate) fn xor_with_public(&self, a: &BigUint, id: PartyID) -> Aby3BigUintShare {
        let mut res = self.to_owned();
        match id {
            PartyID::ID0 => res.a ^= a,
            PartyID::ID1 => res.b ^= a,
            PartyID::ID2 => {}
        }
        res
    }
}

impl std::ops::BitXor for &Aby3BigUintShare {
    type Output = Aby3BigUintShare;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::Output {
            a: &self.a ^ &rhs.a,
            b: &self.b ^ &rhs.b,
        }
    }
}

impl std::ops::BitXor<&BigUint> for &Aby3BigUintShare {
    type Output = Aby3BigUintShare;

    fn bitxor(self, rhs: &BigUint) -> Self::Output {
        Self::Output {
            a: &self.a ^ rhs,
            b: &self.b ^ rhs,
        }
    }
}

impl std::ops::BitXorAssign<&Self> for Aby3BigUintShare {
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.a ^= &rhs.a;
        self.b ^= &rhs.b;
    }
}

impl std::ops::BitXorAssign for Aby3BigUintShare {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.a ^= rhs.a;
        self.b ^= rhs.b;
    }
}

impl std::ops::BitAnd for Aby3BigUintShare {
    type Output = BigUint;

    // Local part of AND only
    fn bitand(self, rhs: Self) -> Self::Output {
        (&self.a & &rhs.a) ^ (self.a & rhs.b) ^ (self.b & rhs.a)
    }
}

impl std::ops::BitAndAssign<&BigUint> for Aby3BigUintShare {
    fn bitand_assign(&mut self, rhs: &BigUint) {
        self.a &= rhs;
        self.b &= rhs;
    }
}

impl std::ops::BitAnd<&BigUint> for &Aby3BigUintShare {
    type Output = Aby3BigUintShare;

    fn bitand(self, rhs: &BigUint) -> Self::Output {
        Aby3BigUintShare {
            a: &self.a & rhs,
            b: &self.b & rhs,
        }
    }
}

impl std::ops::BitAnd<&Self> for Aby3BigUintShare {
    type Output = BigUint;

    // Local part of AND only
    fn bitand(self, rhs: &Self) -> Self::Output {
        (&self.a & &rhs.a) ^ (self.a & &rhs.b) ^ (self.b & &rhs.a)
    }
}

impl std::ops::ShlAssign<usize> for Aby3BigUintShare {
    fn shl_assign(&mut self, rhs: usize) {
        self.a <<= rhs;
        self.b <<= rhs;
    }
}

impl std::ops::Shl<usize> for Aby3BigUintShare {
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output {
        Aby3BigUintShare {
            a: &self.a << rhs,
            b: &self.b << rhs,
        }
    }
}

impl std::ops::Shr<usize> for &Aby3BigUintShare {
    type Output = Aby3BigUintShare;

    fn shr(self, rhs: usize) -> Self::Output {
        Aby3BigUintShare {
            a: &self.a >> rhs,
            b: &self.b >> rhs,
        }
    }
}

impl<F: PrimeField, N: Aby3Network> Aby3Protocol<F, N> {
    const BITLEN: usize = F::MODULUS_BIT_SIZE as usize;

    pub(crate) fn and(
        &mut self,
        a: Aby3BigUintShare,
        b: Aby3BigUintShare,
    ) -> IoResult<Aby3BigUintShare> {
        let (mut mask, mask_b) = self.rngs.rand.random_biguint::<F>();
        mask ^= mask_b;
        let local_a = (a & b) ^ mask;
        self.network.send_next(local_a.to_owned())?;
        let local_b = self.network.recv_prev()?;
        Ok(Aby3BigUintShare {
            a: local_a,
            b: local_b,
        })
    }

    fn and_twice(
        &mut self,
        a: Aby3BigUintShare,
        b1: Aby3BigUintShare,
        b2: Aby3BigUintShare,
    ) -> IoResult<(Aby3BigUintShare, Aby3BigUintShare)> {
        let (mut mask1, mask_b) = self.rngs.rand.random_biguint::<F>();
        mask1 ^= mask_b;

        let (mut mask2, mask_b) = self.rngs.rand.random_biguint::<F>();
        mask2 ^= mask_b;

        let local_a1 = (b1 & &a) ^ mask1;
        let local_a2 = (a & b2) ^ mask2;
        self.network.send_next(local_a1.to_owned())?;
        self.network.send_next(local_a2.to_owned())?;
        let local_b1 = self.network.recv_prev()?;
        let local_b2 = self.network.recv_prev()?;

        let r1 = Aby3BigUintShare {
            a: local_a1,
            b: local_b1,
        };
        let r2 = Aby3BigUintShare {
            a: local_a2,
            b: local_b2,
        };

        Ok((r1, r2))
    }

    fn low_depth_binary_add(
        &mut self,
        x1: Aby3BigUintShare,
        x2: Aby3BigUintShare,
    ) -> IoResult<Aby3BigUintShare> {
        // Add x1 + x2 via a packed Kogge-Stone adder
        let p = &x1 ^ &x2;
        let g = self.and(x1, x2)?;
        self.kogge_stone_inner(p, g, Self::BITLEN + 1)
    }

    // Calculates 2^k + x1 - x2
    fn low_depth_binary_sub(
        &mut self,
        x1: Aby3BigUintShare,
        mut x2: Aby3BigUintShare,
    ) -> IoResult<Aby3BigUintShare> {
        // Let x2' = be the bit_not of x2
        // Add x1 + x2' via a packed Kogge-Stone adder, where carry_in = 1
        // This is equivalent to x1 - x2 = x1 + two's complement of x2
        let mask = (BigUint::from(1u64) << Self::BITLEN) - BigUint::one();
        // bitnot of x2
        x2.a ^= &mask;
        x2.b ^= &mask;
        // Now start the Kogge-Stone adder
        let p = &x1 ^ &x2;
        let mut g = self.and(x1.to_owned(), x2.to_owned())?;
        // Since carry_in = 1, we need to XOR the LSB of x1 and x2 to g (i.e., xor the LSB of p)
        g ^= &p & &BigUint::one();

        let res = self.kogge_stone_inner(p, g, Self::BITLEN + 1)?;
        let res = res.xor_with_public(&BigUint::one(), self.network.get_id()); // cin=1
        Ok(res)
    }

    // Calculates 2^k + x1 - x2
    fn low_depth_binary_sub_by_const(
        &mut self,
        x1: Aby3BigUintShare,
        x2: BigUint,
    ) -> IoResult<Aby3BigUintShare> {
        // two's complement
        let x2_ = (BigUint::from(1u64) << Self::BITLEN) - x2;

        // Add x1 + x2_ via a packed Kogge-Stone adder
        let p = x1.xor_with_public(&x2_, self.network.get_id());
        let g = &x1 & &x2_;

        let res = self.kogge_stone_inner(p, g, Self::BITLEN + 1)?;
        Ok(res)
    }

    // Calculates 2^k + x1 - x2
    fn low_depth_binary_sub_from_const(
        &mut self,
        x1: BigUint,
        mut x2: Aby3BigUintShare,
    ) -> IoResult<Aby3BigUintShare> {
        // Let x2' = be the bit_not of x2
        // Add x1 + x2' via a packed Kogge-Stone adder, where carry_in = 1
        // This is equivalent to x1 - x2 = x1 + two's complement of x2
        let mask = (BigUint::from(1u64) << Self::BITLEN) - BigUint::one();
        // bitnot of x2
        x2.a ^= &mask;
        x2.b ^= &mask;
        // Now start the Kogge-Stone adder
        let p = x2.xor_with_public(&x1, self.network.get_id());
        let mut g = &x2 & &x1;
        // Since carry_in = 1, we need to XOR the LSB of x1 and x2 to g (i.e., xor the LSB of p)
        g ^= &p & &BigUint::one();

        let res = self.kogge_stone_inner(p, g, Self::BITLEN + 1)?;
        let res = res.xor_with_public(&BigUint::one(), self.network.get_id()); // cin=1
        Ok(res)
    }

    fn low_depth_binary_sub_p(&mut self, x: &Aby3BigUintShare) -> IoResult<Aby3BigUintShare> {
        let p_ = (BigUint::from(1u64) << (Self::BITLEN + 1)) - F::MODULUS.into();

        // Add x1 + p_ via a packed Kogge-Stone adder
        let p = x.xor_with_public(&p_, self.network.get_id());
        let g = x & &p_;
        self.kogge_stone_inner(p, g, Self::BITLEN + 2)
    }

    fn kogge_stone_inner(
        &mut self,
        mut p: Aby3BigUintShare,
        mut g: Aby3BigUintShare,
        bit_len: usize,
    ) -> IoResult<Aby3BigUintShare> {
        let d = usize::ilog2(bit_len);
        let s_ = p.to_owned();

        for i in 0..d {
            let shift = 1 << i;
            let mut p_ = p.to_owned();
            let mut g_ = g.to_owned();
            let mask = (BigUint::from(1u64) << (bit_len - shift)) - BigUint::one();
            p_ &= &mask;
            g_ &= &mask;
            let p_shift = &p >> shift;

            // TODO: Make and more communication efficient, ATM we send the full element for each level, even though they reduce in size
            // maybe just input the mask into AND?
            let (r1, r2) = self.and_twice(p_shift, g_, p_)?;
            p = r2 << shift;
            g ^= r1 << shift;
        }
        g <<= 1;
        g ^= s_;
        Ok(g)
    }

    fn cmux(
        &mut self,
        c: Aby3BigUintShare,
        x_t: Aby3BigUintShare,
        x_f: Aby3BigUintShare,
    ) -> IoResult<Aby3BigUintShare> {
        let mut xor = x_t;
        xor ^= &x_f;
        let mut and = self.and(c, xor)?;
        and ^= x_f;
        Ok(and)
    }

    fn low_depth_sub_p_cmux(&mut self, mut x: Aby3BigUintShare) -> IoResult<Aby3BigUintShare> {
        let mask = (BigUint::from(1u64) << Self::BITLEN) - BigUint::one();
        let x_msb = &x >> Self::BITLEN;
        x &= &mask;
        let mut y = self.low_depth_binary_sub_p(&x)?;
        let y_msb = &y >> (Self::BITLEN + 1);
        y &= &mask;

        // Spread the ov share to the whole biguint
        let ov_a = (x_msb.a.iter_u64_digits().next().unwrap()
            ^ y_msb.a.iter_u64_digits().next().unwrap())
            & 1;
        let ov_b = (x_msb.b.iter_u64_digits().next().unwrap()
            ^ y_msb.b.iter_u64_digits().next().unwrap())
            & 1;

        let ov_a = if ov_a == 1 {
            mask.to_owned()
        } else {
            BigUint::zero()
        };
        let ov_b = if ov_b == 1 { mask } else { BigUint::zero() };
        let ov = Aby3BigUintShare::new(ov_a, ov_b);

        // one big multiplexer
        let res = self.cmux(ov, y, x)?;
        Ok(res)
    }

    fn low_depth_binary_add_mod_p(
        &mut self,
        x1: Aby3BigUintShare,
        x2: Aby3BigUintShare,
    ) -> IoResult<Aby3BigUintShare> {
        let x = self.low_depth_binary_add(x1, x2)?;
        self.low_depth_sub_p_cmux(x)
    }

    pub fn a2b(&mut self, x: &Aby3PrimeFieldShare<F>) -> IoResult<Aby3BigUintShare> {
        let mut x01 = Aby3BigUintShare::default();
        let mut x2 = Aby3BigUintShare::default();

        let (mut r, r2) = self.rngs.rand.random_biguint::<F>();
        r ^= r2;

        match self.network.get_id() {
            PartyID::ID0 => {
                x01.a = r;
                x2.b = x.b.into();
            }
            PartyID::ID1 => {
                let val: BigUint = (x.a + x.b).into();
                x01.a = val ^ r;
            }
            PartyID::ID2 => {
                x01.a = r;
                x2.a = x.a.into();
            }
        }

        // Reshare x01
        self.network.send_next(x01.a.to_owned())?;
        let local_b = self.network.recv_prev()?;
        x01.b = local_b;

        self.low_depth_binary_add_mod_p(x01, x2)
    }

    pub fn unsigned_ge(
        &mut self,
        x: Aby3PrimeFieldShare<F>,
        y: Aby3PrimeFieldShare<F>,
    ) -> IoResult<Aby3BigUintShare> {
        let a_bits = self.a2b(&x)?;
        let b_bits = self.a2b(&y)?;
        let diff = self.low_depth_binary_sub(a_bits, b_bits)?;

        Ok(&(&diff >> Self::BITLEN) & &BigUint::one())
    }

    pub fn unsigned_ge_const_lhs(
        &mut self,
        x: F,
        y: Aby3PrimeFieldShare<F>,
    ) -> IoResult<Aby3BigUintShare> {
        let a_bigint = x.into();
        let b_bits = self.a2b(&y)?;
        let diff = self.low_depth_binary_sub_from_const(a_bigint, b_bits)?;

        Ok(&(&diff >> Self::BITLEN) & &BigUint::one())
    }

    pub fn unsigned_ge_const_rhs(
        &mut self,
        x: Aby3PrimeFieldShare<F>,
        y: F,
    ) -> IoResult<Aby3BigUintShare> {
        let a_bits = self.a2b(&x)?;
        let b_bigint = y.into();
        let diff = self.low_depth_binary_sub_by_const(a_bits, b_bigint)?;

        Ok(&(&diff >> Self::BITLEN) & &BigUint::one())
    }

    // Keep in mind: Only works if input is actually a binary sharing of a valid field element
    // If the input has the correct number of bits, but is >= P, then either x can be reduced with self.low_depth_sub_p_cmux(x) first, or self.low_depth_binary_add_2_mod_p(x, y) is extended to subtract 2P in parallel as well. The second solution requires another multiplexer in the end.
    pub fn b2a(&mut self, x: Aby3BigUintShare) -> IoResult<Aby3PrimeFieldShare<F>> {
        let mut y = Aby3BigUintShare::default();
        let mut res = Aby3PrimeFieldShare::default();

        let (mut r, r2) = self.rngs.rand.random_biguint::<F>();
        r ^= r2;

        match self.network.get_id() {
            PartyID::ID0 => {
                let k3 = self.rngs.bitcomp2.random_fes_3keys::<F>();

                res.b = (k3.0 + k3.1 + k3.2).neg();
                y.a = r;
            }
            PartyID::ID1 => {
                let k2 = self.rngs.bitcomp1.random_fes_3keys::<F>();

                res.a = (k2.0 + k2.1 + k2.2).neg();
                y.a = r;
            }
            PartyID::ID2 => {
                let k2 = self.rngs.bitcomp1.random_fes_3keys::<F>();
                let k3 = self.rngs.bitcomp2.random_fes_3keys::<F>();

                let k2_comp = k2.0 + k2.1 + k2.2;
                let k3_comp = k3.0 + k3.1 + k3.2;
                let val: BigUint = (k2_comp + k3_comp).into();
                y.a = val ^ r;
                res.a = k3_comp.neg();
                res.b = k2_comp.neg();
            }
        }

        // Reshare y
        self.network.send_next(y.a.to_owned())?;
        let local_b = self.network.recv_prev()?;
        y.b = local_b;

        let z = self.low_depth_binary_add_mod_p(x, y)?;

        match self.network.get_id() {
            PartyID::ID0 => {
                self.network.send_next(z.b.to_owned())?;
                let rcv: BigUint = self.network.recv_prev()?;
                res.a = (z.a ^ z.b ^ rcv).into();
            }
            PartyID::ID1 => {
                let rcv: BigUint = self.network.recv_prev()?;
                res.b = (z.a ^ z.b ^ rcv).into();
            }
            PartyID::ID2 => {
                self.network.send_next(z.b)?;
            }
        }
        Ok(res)
    }

    pub fn is_zero(&mut self, x: Aby3BigUintShare) -> IoResult<Aby3BigUintShare> {
        let mask = (BigUint::from(1u64) << Self::BITLEN) - BigUint::one();

        // negate
        let mut x = &x ^ &mask;

        // do ands in a tree
        // TODO: Make and tree more communication efficient, ATM we send the full element for each level, even though they halve in size
        let mut len = Self::BITLEN;
        while len > 1 {
            if len % 2 == 1 {
                len += 1;
                // pad with a 1 (= 1 xor 1 xor 1) in MSB position
                // since this is publicly known we just set the bit in each party's share and its replication
                x.a.set_bit(len as u64 - 1, true);
                x.b.set_bit(len as u64 - 1, true);
            }
            len /= 2;
            let y = &x >> len;
            x = self.and(x, y)?;
        }
        // extract LSB
        let x = &x & &BigUint::one();
        Ok(x)
    }

    pub fn bit_inject(&mut self, x: Aby3BigUintShare) -> IoResult<Aby3PrimeFieldShare<F>> {
        // standard bitinject
        assert!(x.a.bits() <= 1);

        let mut b0 = Aby3PrimeFieldShare::default();
        let mut b1 = Aby3PrimeFieldShare::default();
        let mut b2 = Aby3PrimeFieldShare::default();

        match self.network.get_id() {
            PartyID::ID0 => {
                b0.a = x.a.into();
                b2.b = x.b.into();
            }
            PartyID::ID1 => {
                b1.a = x.a.into();
                b0.b = x.b.into();
            }
            PartyID::ID2 => {
                b2.a = x.a.into();
                b1.b = x.b.into();
            }
        };

        let d = self.arithmetic_xor(b0, b1)?;
        let e = self.arithmetic_xor(d, b2)?;
        Ok(e)
    }

    fn arithmetic_xor(
        &mut self,
        x: Aby3PrimeFieldShare<F>,
        y: Aby3PrimeFieldShare<F>,
    ) -> IoResult<Aby3PrimeFieldShare<F>> {
        let d = self.mul(&x, &y)?;
        let d = self.add(&d, &d);
        let e = self.add(&x, &y);
        let d = self.sub(&e, &d);
        Ok(d)
    }

    #[allow(dead_code)]
    pub(crate) fn open_bit_share(&mut self, a: &Aby3BigUintShare) -> IoResult<BigUint> {
        self.network.send_next(a.b.clone())?;
        let c = self.network.recv_prev::<BigUint>()?;
        Ok(&a.a ^ &a.b ^ c)
    }
}
