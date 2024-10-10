use std::collections::HashMap;
use std::io;
use std::marker::PhantomData;

use ark_ff::PrimeField;
use mpc_core::lut::{LookupTableProvider, PlainLookupTableProvider};

use super::NoirWitnessExtensionProtocol;

#[derive(Default)]
pub struct PlainAcvmSolver<F: PrimeField> {
    plain_lut: PlainLookupTableProvider<F>,
    phantom_data: PhantomData<F>,
}

impl<F: PrimeField> NoirWitnessExtensionProtocol<F> for PlainAcvmSolver<F> {
    type Lookup = PlainLookupTableProvider<F>;
    type ArithmeticShare = F;
    type AcvmType = F;

    fn is_public_zero(a: &Self::AcvmType) -> bool {
        a.is_zero()
    }

    fn is_public_one(a: &Self::AcvmType) -> bool {
        a.is_one()
    }

    fn acvm_add_assign_with_public(&mut self, public: F, secret: &mut Self::AcvmType) {
        *secret += public;
    }

    fn acvm_mul_with_public(&mut self, public: F, secret: Self::AcvmType) -> Self::AcvmType {
        secret * public
    }

    fn solve_linear_term(&mut self, q_l: F, w_l: Self::AcvmType, result: &mut Self::AcvmType) {
        *result += q_l * w_l;
    }

    fn solve_mul_term(
        &mut self,
        c: F,
        lhs: Self::AcvmType,
        rhs: Self::AcvmType,
        target: &mut Self::AcvmType,
    ) -> io::Result<()> {
        *target = c * lhs * rhs;
        Ok(())
    }

    fn solve_equation(
        &mut self,
        q_l: Self::AcvmType,
        c: Self::AcvmType,
    ) -> eyre::Result<Self::AcvmType> {
        Ok(-c / q_l)
    }

    fn init_lut_by_acvm_type(&mut self, values: Vec<Self::AcvmType>) -> HashMap<F, F> {
        self.plain_lut
            .init_map(values.into_iter().enumerate().map(|(idx, value)| {
                let promoted_idx = F::from(u64::try_from(idx).expect("usize fits into u64"));
                (promoted_idx, value)
            }))
    }

    fn read_lut_by_acvm_type(
        &mut self,
        index: &Self::AcvmType,
        lut: &HashMap<F, F>,
    ) -> io::Result<F> {
        futures::executor::block_on(self.plain_lut.get_from_lut(*index, lut))
    }

    fn write_lut_by_acvm_type(
        &mut self,
        index: Self::AcvmType,
        value: Self::AcvmType,
        map: &mut HashMap<F, F>,
    ) -> io::Result<()> {
        futures::executor::block_on(self.plain_lut.write_to_lut(index, value, map))
    }

    fn is_shared(_: &Self::AcvmType) -> bool {
        false
    }

    fn get_shared(_: &Self::AcvmType) -> Option<Self::ArithmeticShare> {
        None
    }

    fn open_many(&mut self, a: &[Self::ArithmeticShare]) -> io::Result<Vec<F>> {
        Ok(a.to_vec())
    }
}
