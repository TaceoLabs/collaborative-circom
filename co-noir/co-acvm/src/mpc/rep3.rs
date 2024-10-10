use std::marker::PhantomData;

use ark_ff::PrimeField;
use itertools::{izip, Itertools};
use mpc_core::protocols::rep3::arithmetic;
use mpc_core::{
    lut::LookupTableProvider,
    protocols::rep3::{
        lut::NaiveRep3LookupTable,
        network::{IoContext, Rep3Network},
        Rep3PrimeFieldShare,
    },
};

use super::plain::PlainAcvmSolver;
use super::NoirWitnessExtensionProtocol;
use tokio::runtime::{self, Runtime};
type ArithmeticShare<F> = Rep3PrimeFieldShare<F>;

pub struct Rep3AcvmSolver<F: PrimeField, N: Rep3Network> {
    runtime: Runtime,
    lut_provider: NaiveRep3LookupTable<N>,
    io_context: IoContext<N>,
    plain_solver: PlainAcvmSolver<F>,
    phantom_data: PhantomData<F>,
}

impl<F: PrimeField, N: Rep3Network> Rep3AcvmSolver<F, N> {
    pub(crate) fn new(network: N) -> Self {
        let runtime = runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let plain_solver = PlainAcvmSolver::<F>::default();
        let mut io_context = runtime.block_on(IoContext::init(network)).unwrap();
        let forked = runtime.block_on(io_context.fork()).unwrap();
        Self {
            runtime,
            lut_provider: NaiveRep3LookupTable::new(forked),
            io_context,
            plain_solver,
            phantom_data: PhantomData,
        }
    }
}

// TODO maybe we want to merge that with the Rep3VmType?? Atm we do not need
// binary shares so maybe it is ok..
#[derive(Clone)]
pub enum Rep3AcvmType<F: PrimeField> {
    Public(F),
    Shared(ArithmeticShare<F>),
}

impl<F: PrimeField> std::fmt::Debug for Rep3AcvmType<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public(field) => f.debug_tuple("Public").field(field).finish(),
            Self::Shared(share) => f.debug_tuple("Arithmetic").field(share).finish(),
        }
    }
}

impl<F: PrimeField> std::fmt::Display for Rep3AcvmType<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public(field) => f.write_str(&format!("Public ({field})")),
            Self::Shared(arithmetic) => {
                let (a, b) = arithmetic.ab();
                f.write_str(&format!("Arithmetic (a: {}, b: {})", a, b))
            }
        }
    }
}

impl<F: PrimeField> Default for Rep3AcvmType<F> {
    fn default() -> Self {
        Self::Public(F::zero())
    }
}

impl<F: PrimeField> From<F> for Rep3AcvmType<F> {
    fn from(value: F) -> Self {
        Self::Public(value)
    }
}

impl<F: PrimeField> From<ArithmeticShare<F>> for Rep3AcvmType<F> {
    fn from(value: ArithmeticShare<F>) -> Self {
        Self::Shared(value)
    }
}

impl<F: PrimeField, N: Rep3Network> NoirWitnessExtensionProtocol<F> for Rep3AcvmSolver<F, N> {
    type Lookup = NaiveRep3LookupTable<N>;

    type ArithmeticShare = Rep3PrimeFieldShare<F>;

    type AcvmType = Rep3AcvmType<F>;

    fn is_public_zero(a: &Self::AcvmType) -> bool {
        if let Rep3AcvmType::Public(x) = a {
            x.is_zero()
        } else {
            false
        }
    }

    fn is_public_one(a: &Self::AcvmType) -> bool {
        if let Rep3AcvmType::Public(x) = a {
            x.is_one()
        } else {
            false
        }
    }

    fn acvm_add_assign_with_public(&mut self, public: F, target: &mut Self::AcvmType) {
        let id = self.io_context.id;
        let result = match target.to_owned() {
            Rep3AcvmType::Public(secret) => Rep3AcvmType::Public(public + secret),
            Rep3AcvmType::Shared(secret) => {
                Rep3AcvmType::Shared(arithmetic::add_public(secret, public, id))
            }
        };
        *target = result;
    }

    fn acvm_mul_with_public(&mut self, public: F, secret: Self::AcvmType) -> Self::AcvmType {
        match secret {
            Rep3AcvmType::Public(secret) => Rep3AcvmType::Public(public * secret),
            Rep3AcvmType::Shared(secret) => {
                Rep3AcvmType::Shared(arithmetic::mul_public(secret, public))
            }
        }
    }

    fn solve_linear_term(&mut self, q_l: F, w_l: Self::AcvmType, target: &mut Self::AcvmType) {
        let id = self.io_context.id;
        let result = match (w_l, target.to_owned()) {
            (Rep3AcvmType::Public(w_l), Rep3AcvmType::Public(result)) => {
                Rep3AcvmType::Public(q_l * w_l + result)
            }
            (Rep3AcvmType::Public(w_l), Rep3AcvmType::Shared(result)) => {
                Rep3AcvmType::Shared(arithmetic::add_public(result, q_l * w_l, id))
            }
            (Rep3AcvmType::Shared(w_l), Rep3AcvmType::Public(result)) => {
                let mul = arithmetic::mul_public(w_l, q_l);
                Rep3AcvmType::Shared(arithmetic::add_public(mul, result, id))
            }
            (Rep3AcvmType::Shared(w_l), Rep3AcvmType::Shared(result)) => {
                let mul = arithmetic::mul_public(w_l, q_l);
                Rep3AcvmType::Shared(arithmetic::add(mul, result))
            }
        };
        *target = result;
    }

    fn solve_mul_term(
        &mut self,
        c: F,
        lhs: Self::AcvmType,
        rhs: Self::AcvmType,
        target: &mut Self::AcvmType,
    ) -> std::io::Result<()> {
        let result = match (lhs, rhs) {
            (Rep3AcvmType::Public(lhs), Rep3AcvmType::Public(rhs)) => {
                Rep3AcvmType::Public(lhs * rhs * c)
            }
            (Rep3AcvmType::Public(public), Rep3AcvmType::Shared(shared))
            | (Rep3AcvmType::Shared(shared), Rep3AcvmType::Public(public)) => {
                Rep3AcvmType::Shared(arithmetic::mul_public(shared, public))
            }
            (Rep3AcvmType::Shared(lhs), Rep3AcvmType::Shared(rhs)) => {
                let future = arithmetic::mul(lhs, rhs, &mut self.io_context);
                let shared_mul = self.runtime.block_on(future)?;
                Rep3AcvmType::Shared(arithmetic::mul_public(shared_mul, c))
            }
        };
        *target = result;
        Ok(())
    }

    fn solve_equation(
        &mut self,
        q_l: Self::AcvmType,
        c: Self::AcvmType,
    ) -> eyre::Result<Self::AcvmType> {
        //-c/q_l
        let io_context = &mut self.io_context;
        let result = match (q_l, c) {
            (Rep3AcvmType::Public(q_l), Rep3AcvmType::Public(c)) => {
                Rep3AcvmType::Public(self.plain_solver.solve_equation(q_l, c)?)
            }
            (Rep3AcvmType::Public(q_l), Rep3AcvmType::Shared(c)) => {
                Rep3AcvmType::Shared(arithmetic::div_shared_by_public(arithmetic::neg(c), q_l)?)
            }
            (Rep3AcvmType::Shared(q_l), Rep3AcvmType::Public(c)) => {
                let future = arithmetic::div_public_by_shared(-c, q_l, io_context);
                let result = self.runtime.block_on(future)?;
                Rep3AcvmType::Shared(result)
            }
            (Rep3AcvmType::Shared(q_l), Rep3AcvmType::Shared(c)) => {
                let future = arithmetic::div(arithmetic::neg(c), q_l, io_context);
                let result = self.runtime.block_on(future)?;
                Rep3AcvmType::Shared(result)
            }
        };
        Ok(result)
    }

    fn init_lut_by_acvm_type(
        &mut self,
        values: Vec<Self::AcvmType>,
    ) -> <Self::Lookup as mpc_core::lut::LookupTableProvider<F>>::SecretSharedMap {
        let id = self.io_context.id;
        let values = values.into_iter().enumerate().map(|(idx, value)| {
            let idx = F::from(u64::try_from(idx).expect("usize fits into u64"));
            let value = match value {
                Rep3AcvmType::Public(public) => arithmetic::promote_to_trivial_share(id, public),
                Rep3AcvmType::Shared(shared) => shared,
            };
            (arithmetic::promote_to_trivial_share(id, idx), value)
        });
        self.lut_provider.init_map(values)
    }

    fn read_lut_by_acvm_type(
        &mut self,
        index: &Self::AcvmType,
        lut: &<Self::Lookup as mpc_core::lut::LookupTableProvider<F>>::SecretSharedMap,
    ) -> std::io::Result<Self::AcvmType> {
        let value = match index {
            Rep3AcvmType::Public(public) => {
                let id = self.io_context.id;
                let promoted_key = arithmetic::promote_to_trivial_share(id, *public);
                self.lut_provider.get_from_lut(promoted_key, lut)
            }
            Rep3AcvmType::Shared(shared) => self.lut_provider.get_from_lut(*shared, lut),
        };
        let value = self.runtime.block_on(value)?;
        Ok(Rep3AcvmType::Shared(value))
    }

    fn write_lut_by_acvm_type(
        &mut self,
        index: Self::AcvmType,
        value: Self::AcvmType,
        lut: &mut <Self::Lookup as mpc_core::lut::LookupTableProvider<F>>::SecretSharedMap,
    ) -> std::io::Result<()> {
        let id = self.io_context.id;
        let future = match (index, value) {
            (Rep3AcvmType::Public(index), Rep3AcvmType::Public(value)) => {
                let index = arithmetic::promote_to_trivial_share(id, index);
                let value = arithmetic::promote_to_trivial_share(id, value);
                self.lut_provider.write_to_lut(index, value, lut)
            }
            (Rep3AcvmType::Public(index), Rep3AcvmType::Shared(value)) => {
                let index = arithmetic::promote_to_trivial_share(id, index);
                self.lut_provider.write_to_lut(index, value, lut)
            }
            (Rep3AcvmType::Shared(index), Rep3AcvmType::Public(value)) => {
                let value = arithmetic::promote_to_trivial_share(id, value);
                self.lut_provider.write_to_lut(index, value, lut)
            }
            (Rep3AcvmType::Shared(index), Rep3AcvmType::Shared(value)) => {
                self.lut_provider.write_to_lut(index, value, lut)
            }
        };
        self.runtime.block_on(future)?;
        Ok(())
    }

    fn is_shared(a: &Self::AcvmType) -> bool {
        matches!(a, Rep3AcvmType::Shared(_))
    }

    fn get_shared(a: &Self::AcvmType) -> Option<Self::ArithmeticShare> {
        match a {
            Rep3AcvmType::Shared(shared) => Some(*shared),
            _ => None,
        }
    }

    fn open_many(&mut self, a: &[Self::ArithmeticShare]) -> std::io::Result<Vec<F>> {
        let bs = a.iter().map(|x| x.b).collect_vec();
        self.runtime
            .block_on(self.io_context.network.send_next(bs))?;
        let mut cs = self
            .runtime
            .block_on(self.io_context.network.recv_prev::<Vec<F>>())?;

        izip!(a, cs.iter_mut()).for_each(|(x, c)| *c += x.a + x.b);

        Ok(cs)
    }
}
