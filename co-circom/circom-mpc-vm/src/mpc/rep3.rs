use std::io;

use ark_ff::{One, PrimeField};
use eyre::{bail, eyre};
use mpc_core::protocols::rep3::{
    arithmetic, binary, conversion,
    network::{IoContext, Rep3Network},
    Rep3BigUintShare, Rep3PrimeFieldShare,
};
use num_bigint::BigUint;
use num_traits::cast::ToPrimitive;

use super::{
    plain::{to_usize, CircomPlainVmWitnessExtension},
    VmCircomWitnessExtension,
};
use tokio::runtime::{self};

type ArithmeticShare<F> = Rep3PrimeFieldShare<F>;
type BinaryShare<F> = Rep3BigUintShare<F>;

/// This type represents a public, arithmetic share, or binary share type used in the co-cricom MPC-VM
#[derive(Clone)]
pub enum Rep3VmType<F: PrimeField> {
    /// The public variant
    Public(F),
    /// The arithemtic share variant
    Arithmetic(ArithmeticShare<F>),
    /// The binary share variant
    Binary(BinaryShare<F>),
}

impl<F: PrimeField> From<F> for Rep3VmType<F> {
    fn from(value: F) -> Self {
        Self::Public(value)
    }
}

impl<F: PrimeField> From<ArithmeticShare<F>> for Rep3VmType<F> {
    fn from(value: ArithmeticShare<F>) -> Self {
        Self::Arithmetic(value)
    }
}

impl<F: PrimeField> From<BinaryShare<F>> for Rep3VmType<F> {
    fn from(value: BinaryShare<F>) -> Self {
        Self::Binary(value)
    }
}

impl<F: PrimeField> Default for Rep3VmType<F> {
    fn default() -> Self {
        Self::Public(F::zero())
    }
}

pub struct CircomRep3VmWitnessExtension<F: PrimeField, N: Rep3Network> {
    io_context0: IoContext<N>,
    io_context1: IoContext<N>,
    runtime: runtime::Runtime,
    plain: CircomPlainVmWitnessExtension<F>,
}

impl<F: PrimeField, N: Rep3Network> CircomRep3VmWitnessExtension<F, N> {
    pub fn from_network(network: N) -> io::Result<Self> {
        let runtime = runtime::Builder::new_multi_thread().enable_all().build()?;
        let mut io_context = runtime.block_on(IoContext::init(network))?;
        let io_context_fork = runtime.block_on(io_context.fork())?;
        Ok(Self {
            io_context0: io_context,
            io_context1: io_context_fork,
            runtime,
            plain: CircomPlainVmWitnessExtension::default(),
        })
    }

    /// Normally F is split into positive and negative numbers in the range [0, p/2] and [p/2 + 1, p)
    /// However, for comparisons, we want the negative numbers to be "lower" than the positive ones.
    /// Therefore we shift the input by p/2 + 1 to the left, which results in a mapping of [negative, 0, positive] into F.
    /// We can then compare the numbers as if they were unsigned.
    /// While this could be done easier by just comparing the numbers as BigInt, we do it this way because this is easier to replicate in MPC later.
    #[inline(always)]
    fn val(&mut self, z: ArithmeticShare<F>) -> ArithmeticShare<F> {
        let modulus: BigUint = F::MODULUS.into();
        let one = BigUint::one();
        let two = BigUint::from(2u64);
        let p_half_plus_one = F::from(modulus / two + one);
        arithmetic::sub_shared_by_public(z, p_half_plus_one, self.io_context0.id)
    }

    pub fn close_network(self) -> eyre::Result<()> {
        self.runtime.block_on(self.io_context0.network.shutdown())?;
        self.runtime.block_on(self.io_context1.network.shutdown())?;
        Ok(())
    }
}

impl<F: PrimeField, N: Rep3Network> VmCircomWitnessExtension<F>
    for CircomRep3VmWitnessExtension<F, N>
{
    type ArithmeticShare = ArithmeticShare<F>;

    type BinaryShare = BinaryShare<F>;

    type VmType = Rep3VmType<F>;

    fn add(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.add(a, b)?.into()),
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                Ok(arithmetic::add_public(a, b, self.io_context0.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                Ok(arithmetic::add(a, b).into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                Ok(arithmetic::add_public(a, b, self.io_context0.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                Ok(arithmetic::add(a, b).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                Ok(arithmetic::add(a?, b?).into())
            }
        }
    }

    fn sub(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.sub(a, b)?.into()),
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                Ok(arithmetic::sub_shared_by_public(a, b, self.io_context0.id).into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Arithmetic(b)) => {
                Ok(arithmetic::sub_public_by_shared(a, b, self.io_context0.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                Ok(arithmetic::sub(a, b).into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                Ok(arithmetic::sub_public_by_shared(a, b, self.io_context0.id).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                Ok(arithmetic::sub_shared_by_public(a, b, self.io_context0.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                Ok(arithmetic::sub(a, b).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Arithmetic(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                Ok(arithmetic::sub(a, b).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                Ok(arithmetic::sub(a?, b?).into())
            }
        }
    }

    fn mul(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.mul(a, b)?.into()),
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                Ok(arithmetic::mul_public(a, b).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => Ok(self
                .runtime
                .block_on(arithmetic::mul(a, b, &mut self.io_context0))?
                .into()),
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                Ok(arithmetic::mul_public(a, b).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                Ok(self
                    .runtime
                    .block_on(arithmetic::mul(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                Ok(self
                    .runtime
                    .block_on(arithmetic::mul(a?, b?, &mut self.io_context0))?
                    .into())
            }
        }
    }

    fn div(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.div(a, b)?.into()),
            (Rep3VmType::Public(a), Rep3VmType::Arithmetic(b)) => {
                let b = self
                    .runtime
                    .block_on(arithmetic::inv(b, &mut self.io_context0))?;
                Ok(arithmetic::mul_public(b, a).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                if b.is_zero() {
                    bail!("Cannot invert zero");
                }
                Ok(arithmetic::mul_public(a, b.inverse().unwrap()).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let b = self
                    .runtime
                    .block_on(arithmetic::inv(b, &mut self.io_context0))?;
                Ok(self
                    .runtime
                    .block_on(arithmetic::mul(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                let b = self
                    .runtime
                    .block_on(arithmetic::inv(b, &mut self.io_context0))?;
                Ok(arithmetic::mul_public(b, a).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                if b.is_zero() {
                    bail!("Cannot invert zero");
                }
                Ok(arithmetic::mul_public(a, b.inverse().unwrap()).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                let b = self
                    .runtime
                    .block_on(arithmetic::inv(b, &mut self.io_context0))?;
                Ok(self
                    .runtime
                    .block_on(arithmetic::mul(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Arithmetic(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        arithmetic::inv(b, &mut self.io_context1),
                    )
                });
                let a = a?;
                let b = b?;
                Ok(self
                    .runtime
                    .block_on(arithmetic::mul(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                let a = a?;
                let b = b?;
                let b = self
                    .runtime
                    .block_on(arithmetic::inv(b, &mut self.io_context0))?;

                Ok(self
                    .runtime
                    .block_on(arithmetic::mul(a, b, &mut self.io_context0))?
                    .into())
            }
        }
    }

    fn int_div(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.int_div(a, b)?.into()),
            _ => todo!("Shared int_div not implemented"),
        }
    }

    fn pow(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.pow(a, b)?.into()),
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.pow(a.into(), b.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                if b.is_zero() {
                    return Ok(Rep3VmType::Public(F::one()));
                }
                Ok(self
                    .runtime
                    .block_on(arithmetic::pow_public(a, b, &mut self.io_context0))?
                    .into())
            }
            _ => todo!("pow with shared exponent not implemented"),
        }
    }

    fn modulo(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.modulo(a, b)?.into()),
            (_, _) => todo!("Shared mod not implemented"),
        }
    }

    fn sqrt(&mut self, a: Self::VmType) -> eyre::Result<Self::VmType> {
        match a {
            Rep3VmType::Public(a) => Ok(self.plain.sqrt(a)?.into()),
            Rep3VmType::Arithmetic(a) => {
                let sqrt = self
                    .runtime
                    .block_on(arithmetic::sqrt(a, &mut self.io_context0))?;
                // Correction to give the result closest to 0
                // I.e., 2 * is_pos * sqrt - sqrt
                let sqrt_val = self.val(sqrt);
                let zero_val = self.plain.val(F::zero());
                let is_pos = self.runtime.block_on(arithmetic::ge_public(
                    sqrt_val,
                    zero_val,
                    &mut self.io_context0,
                ))?;
                let mut mul =
                    self.runtime
                        .block_on(arithmetic::mul(sqrt, is_pos, &mut self.io_context0))?;
                mul.double();
                mul -= sqrt;
                Ok(mul.into())
            }
            Rep3VmType::Binary(a) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.sqrt(a.into())
            }
        }
    }

    fn neg(&mut self, a: Self::VmType) -> eyre::Result<Self::VmType> {
        match a {
            Rep3VmType::Public(a) => Ok(self.plain.neg(a)?.into()),
            Rep3VmType::Arithmetic(a) => Ok(arithmetic::neg(a).into()),
            Rep3VmType::Binary(a) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                Ok(arithmetic::neg(a).into())
            }
        }
    }

    fn lt(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.lt(a, b)?.into()),
            (Rep3VmType::Public(a), Rep3VmType::Arithmetic(b)) => {
                let a = self.plain.val(a);
                let b = self.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::ge_public(b, a, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let a = self.val(a);
                let b = self.plain.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::lt_public(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let a = self.val(a);
                let b = self.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::lt(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.lt(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.lt(a.into(), b.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.lt(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Arithmetic(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.lt(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                self.lt(a?.into(), b?.into())
            }
        }
    }

    fn le(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.le(a, b)?.into()),
            (Rep3VmType::Public(a), Rep3VmType::Arithmetic(b)) => {
                let a = self.plain.val(a);
                let b = self.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::gt_public(b, a, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let a = self.val(a);
                let b = self.plain.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::le_public(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let a = self.val(a);
                let b = self.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::le(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.le(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.le(a.into(), b.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.le(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Arithmetic(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.le(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                self.le(a?.into(), b?.into())
            }
        }
    }

    fn gt(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.gt(a, b)?.into()),
            (Rep3VmType::Public(a), Rep3VmType::Arithmetic(b)) => {
                let a = self.plain.val(a);
                let b = self.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::le_public(b, a, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let a = self.val(a);
                let b = self.plain.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::gt_public(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let a = self.val(a);
                let b = self.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::gt(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.gt(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.gt(a.into(), b.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.gt(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Arithmetic(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.gt(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                self.gt(a?.into(), b?.into())
            }
        }
    }

    fn ge(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.ge(a, b)?.into()),
            (Rep3VmType::Public(a), Rep3VmType::Arithmetic(b)) => {
                let a = self.plain.val(a);
                let b = self.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::lt_public(b, a, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let a = self.val(a);
                let b = self.plain.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::ge_public(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let a = self.val(a);
                let b = self.val(b);
                Ok(self
                    .runtime
                    .block_on(arithmetic::ge(a, b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Public(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.ge(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.ge(a.into(), b.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.ge(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Arithmetic(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.ge(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                self.ge(a?.into(), b?.into())
            }
        }
    }

    fn eq(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.eq(a, b)?.into()),
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => Ok(self
                .runtime
                .block_on(arithmetic::eq_public(a, b, &mut self.io_context0))?
                .into()),
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => Ok(self
                .runtime
                .block_on(arithmetic::eq(a, b, &mut self.io_context0))?
                .into()),
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.eq(a.into(), b.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.eq(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                self.eq(a?.into(), b?.into())
            }
        }
    }

    fn neq(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.neq(a, b)?.into()),
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => Ok(self
                .runtime
                .block_on(arithmetic::neq_public(a, b, &mut self.io_context0))?
                .into()),
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => Ok(self
                .runtime
                .block_on(arithmetic::neq(a, b, &mut self.io_context0))?
                .into()),
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.neq(a.into(), b.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.neq(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                self.neq(a?.into(), b?.into())
            }
        }
    }

    fn shift_r(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.shift_r(a, b)?.into()),
            (Rep3VmType::Public(a), Rep3VmType::Arithmetic(_))
            | (Rep3VmType::Public(a), Rep3VmType::Binary(_)) => {
                // some special casing
                if a == F::zero() {
                    return Ok(Rep3VmType::Public(F::zero()));
                }
                todo!("Shared shift_right (public by shared) not implemented");
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let bits = self
                    .runtime
                    .block_on(conversion::a2b(a, &mut self.io_context0))?;
                self.shift_r(bits.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                Ok(binary::shift_r_public(&a, b).into())
            }
            (_, _) => todo!("Shared shift_right not implemented"),
        }
    }

    fn shift_l(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.shift_l(a, b)?.into()),
            (Rep3VmType::Public(a), Rep3VmType::Arithmetic(b)) => {
                // some special casing
                if a == F::zero() {
                    Ok(Rep3VmType::Public(F::zero()))
                } else {
                    let b = self
                        .runtime
                        .block_on(conversion::a2b(b, &mut self.io_context0))?;
                    self.shift_l(a.into(), b.into())
                }
            }
            (Rep3VmType::Public(a), Rep3VmType::Binary(b)) => {
                // some special casing
                if a == F::zero() {
                    Ok(Rep3VmType::Public(F::zero()))
                } else {
                    let res = self.runtime.block_on(binary::shift_l_public_by_shared(
                        a,
                        &b,
                        &mut self.io_context0,
                    ))?;
                    Ok(res.into())
                }
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                Ok(arithmetic::pow_2_public(a, b).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                Ok(binary::shift_l_public(&a, b).into())
            }
            (_, _) => todo!("Shared shift_right not implemented"),
        }
    }

    fn bool_not(&mut self, a: Self::VmType) -> eyre::Result<Self::VmType> {
        match a {
            Rep3VmType::Public(a) => Ok(self.plain.bool_not(a)?.into()),
            Rep3VmType::Arithmetic(a) => {
                let neg_a = arithmetic::neg(a);
                let not_a = arithmetic::add_public(neg_a, F::one(), self.io_context0.id);
                Ok(not_a.into())
            }
            Rep3VmType::Binary(a) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.bool_not(a.into())
            }
        }
    }

    fn bool_and(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.bool_and(a, b)?.into()),
            (a, b) => self.mul(a, b),
        }
    }

    fn bool_or(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.bool_or(a, b)?.into()),
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let mul = arithmetic::mul_public(a, b);
                let add = arithmetic::add_public(a, b, self.io_context0.id);
                let sub = arithmetic::sub(add, mul);
                Ok(sub.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let mul = self
                    .runtime
                    .block_on(arithmetic::mul(a, b, &mut self.io_context0))?;
                let add = arithmetic::add(a, b);
                let sub = arithmetic::sub(add, mul);
                Ok(sub.into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.bool_or(a.into(), b.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let b = self
                    .runtime
                    .block_on(conversion::b2a(&b, &mut self.io_context0))?;
                self.bool_or(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::b2a(&a, &mut self.io_context0),
                        conversion::b2a(&b, &mut self.io_context1),
                    )
                });
                self.bool_or(a?.into(), b?.into())
            }
        }
    }

    fn cmux(
        &mut self,
        cond: Self::VmType,
        truthy: Self::VmType,
        falsy: Self::VmType,
    ) -> eyre::Result<Self::VmType> {
        match (cond, truthy, falsy) {
            (Rep3VmType::Public(cond), truthy, falsy) => {
                assert!(cond.is_one() || cond.is_zero());
                if cond.is_one() {
                    Ok(truthy)
                } else {
                    Ok(falsy)
                }
            }
            (Rep3VmType::Arithmetic(cond), truthy, falsy) => {
                let b_min_a = self.sub(truthy, falsy.clone())?;
                let d = self.mul(cond.into(), b_min_a)?;
                self.add(falsy, d)
            }
            (Rep3VmType::Binary(cond), truthy, falsy) => {
                let cond = self
                    .runtime
                    .block_on(conversion::b2a(&cond, &mut self.io_context0))?;
                self.cmux(cond.into(), truthy, falsy)
            }
        }
    }

    fn bit_xor(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.bit_xor(a, b)?.into()),
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::a2b(a, &mut self.io_context0))?;
                Ok(binary::xor_public(&a, &b.into_bigint().into(), self.io_context0.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::a2b(a, &mut self.io_context0),
                        conversion::a2b(b, &mut self.io_context1),
                    )
                });
                Ok(binary::xor(&a?, &b?).into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                Ok(binary::xor_public(&a, &b.into_bigint().into(), self.io_context0.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let a = self
                    .runtime
                    .block_on(conversion::a2b(a, &mut self.io_context0))?;
                Ok(binary::xor(&a, &b).into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => Ok(binary::xor(&a, &b).into()),
        }
    }

    fn bit_or(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.bit_or(a, b)?.into()),
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::a2b(a, &mut self.io_context0))?;
                self.bit_or(a.into(), b.into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::a2b(a, &mut self.io_context0),
                        conversion::a2b(b, &mut self.io_context1),
                    )
                });
                self.bit_or(a?.into(), b?.into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                Ok(binary::or_public(&a, &b.into_bigint().into(), self.io_context0.id).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let a = self
                    .runtime
                    .block_on(conversion::a2b(a, &mut self.io_context0))?;
                self.bit_or(a.into(), b.into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => Ok(self
                .runtime
                .block_on(binary::or(&a, &b, &mut self.io_context0))?
                .into()),
        }
    }

    fn bit_and(&mut self, a: Self::VmType, b: Self::VmType) -> eyre::Result<Self::VmType> {
        match (a, b) {
            (Rep3VmType::Public(a), Rep3VmType::Public(b)) => Ok(self.plain.bit_and(a, b)?.into()),
            (Rep3VmType::Public(b), Rep3VmType::Arithmetic(a))
            | (Rep3VmType::Arithmetic(a), Rep3VmType::Public(b)) => {
                let a = self
                    .runtime
                    .block_on(conversion::a2b(a, &mut self.io_context0))?;
                Ok(binary::and_with_public(&a, &b.into_bigint().into()).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Arithmetic(b)) => {
                let (a, b) = self.runtime.block_on(async {
                    tokio::join!(
                        conversion::a2b(a, &mut self.io_context0),
                        conversion::a2b(b, &mut self.io_context1),
                    )
                });
                Ok(self
                    .runtime
                    .block_on(binary::and(&a?, &b?, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Public(b), Rep3VmType::Binary(a))
            | (Rep3VmType::Binary(a), Rep3VmType::Public(b)) => {
                Ok(binary::and_with_public(&a, &b.into_bigint().into()).into())
            }
            (Rep3VmType::Arithmetic(a), Rep3VmType::Binary(b))
            | (Rep3VmType::Binary(b), Rep3VmType::Arithmetic(a)) => {
                let a = self
                    .runtime
                    .block_on(conversion::a2b(a, &mut self.io_context0))?;
                Ok(self
                    .runtime
                    .block_on(binary::and(&a, &b, &mut self.io_context0))?
                    .into())
            }
            (Rep3VmType::Binary(a), Rep3VmType::Binary(b)) => Ok(self
                .runtime
                .block_on(binary::and(&a, &b, &mut self.io_context0))?
                .into()),
        }
    }

    fn is_zero(&mut self, a: Self::VmType, allow_secret_inputs: bool) -> eyre::Result<bool> {
        if !allow_secret_inputs && self.is_shared(&a)? {
            bail!("allow_secret_inputs is false and input is shared");
        }
        match a {
            Rep3VmType::Public(a) => Ok(self.plain.is_zero(a, allow_secret_inputs)?),
            Rep3VmType::Arithmetic(a) => Ok(self
                .runtime
                .block_on(arithmetic::is_zero(a, &mut self.io_context0))?),
            Rep3VmType::Binary(a) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.is_zero(a.into(), allow_secret_inputs)
            }
        }
    }

    fn is_shared(&mut self, a: &Self::VmType) -> eyre::Result<bool> {
        match a {
            Rep3VmType::Public(_) => Ok(false),
            Rep3VmType::Arithmetic(_) => Ok(true),
            Rep3VmType::Binary(_) => Ok(true),
        }
    }

    fn to_index(&mut self, a: Self::VmType) -> eyre::Result<usize> {
        if let Rep3VmType::Public(a) = a {
            Ok(to_usize!(a))
        } else {
            bail!("ToIndex called on shared value!")
        }
    }

    fn open(&mut self, a: Self::VmType) -> eyre::Result<F> {
        match a {
            Rep3VmType::Public(a) => Ok(a),
            Rep3VmType::Arithmetic(a) => Ok(self
                .runtime
                .block_on(arithmetic::open(a, &mut self.io_context0))?),
            Rep3VmType::Binary(a) => Ok(self
                .runtime
                .block_on(binary::open(&a, &mut self.io_context0))?
                .into()),
        }
    }

    fn to_share(&mut self, a: Self::VmType) -> eyre::Result<Self::ArithmeticShare> {
        match a {
            Rep3VmType::Public(a) => {
                Ok(arithmetic::promote_to_trivial_share(self.io_context0.id, a))
            }
            Rep3VmType::Arithmetic(a) => Ok(a),
            Rep3VmType::Binary(a) => {
                let a = self
                    .runtime
                    .block_on(conversion::b2a(&a, &mut self.io_context0))?;
                self.to_share(a.into())
            }
        }
    }

    fn public_one(&self) -> Self::VmType {
        F::one().into()
    }

    fn public_zero(&self) -> Self::VmType {
        F::zero().into()
    }
}

impl<F: PrimeField> std::fmt::Debug for Rep3VmType<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public(field) => f.debug_tuple("Public").field(field).finish(),
            Self::Arithmetic(share) => f.debug_tuple("Arithmetic").field(share).finish(),
            Self::Binary(binary) => f.debug_tuple("Binary").field(binary).finish(),
        }
    }
}

impl<F: PrimeField> std::fmt::Display for Rep3VmType<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public(field) => f.write_str(&format!("Public ({field})")),
            Self::Arithmetic(arithmetic) => {
                let (a, b) = arithmetic.ab();
                f.write_str(&format!("Arithmetic (a: {}, b: {})", a, b))
            }
            Self::Binary(binary) => {
                let (a, b) = binary.clone().ab();
                f.write_str(&format!("Binary (a: {}, b: {})", a, b))
            }
        }
    }
}
