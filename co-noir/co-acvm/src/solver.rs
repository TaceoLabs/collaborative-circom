use acir::{
    acir_field::GenericFieldElement,
    circuit::{Circuit, ExpressionWidth, Opcode},
    native_types::{WitnessMap, WitnessStack},
    FieldElement,
};
use ark_ff::PrimeField;
use intmap::IntMap;
use mpc_core::{lut::LookupTableProvider, protocols::rep3::network::Rep3Network};
use noirc_abi::{input_parser::Format, Abi, MAIN_RETURN_NAME};
use noirc_artifacts::program::ProgramArtifact;
use std::{collections::BTreeMap, io, path::PathBuf};

use crate::mpc::{plain::PlainAcvmSolver, rep3::Rep3AcvmSolver, NoirWitnessExtensionProtocol};
/// The default expression width defined used by the ACVM.
pub(crate) const CO_EXPRESSION_WIDTH: ExpressionWidth = ExpressionWidth::Bounded { width: 4 };

mod assert_zero_solver;
mod memory_solver;
pub mod partial_abi;

pub type PlainCoSolver<F> = CoSolver<PlainAcvmSolver<F>, F>;
pub type Rep3CoSolver<F, N> = CoSolver<Rep3AcvmSolver<F, N>, F>;

type CoAcvmResult<T> = std::result::Result<T, CoAcvmError>;

pub(crate) mod solver_utils {
    use acir::native_types::Expression;

    pub(crate) fn expr_to_string<F: std::fmt::Display>(expr: &Expression<F>) -> String {
        let mul_terms = expr
            .mul_terms
            .iter()
            .map(|(q_m, w_l, w_r)| format!("({q_m} * _{w_l:?} * _{w_r:?})"))
            .collect::<Vec<String>>()
            .join(" + ");
        let linear_terms = expr
            .linear_combinations
            .iter()
            .map(|(coef, w)| format!("({coef} * _{w:?})"))
            .collect::<Vec<String>>()
            .join(" + ");
        format!("EXPR [({mul_terms}) + ({linear_terms}) + {}]", expr.q_c)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CoAcvmError {
    #[error(transparent)]
    IOError(#[from] io::Error),
    #[error(transparent)]
    UnrecoverableError(#[from] eyre::Report),
}

pub struct CoSolver<T, F>
where
    T: NoirWitnessExtensionProtocol<F>,
    F: PrimeField,
{
    driver: T,
    abi: Abi,
    functions: Vec<Circuit<GenericFieldElement<F>>>,
    // maybe this can be an array. lets see..
    witness_map: Vec<WitnessMap<T::AcvmType>>,
    // there will a more fields added as we add functionality
    function_index: usize,
    // the memory blocks
    memory_access: IntMap<<T::Lookup as LookupTableProvider<F>>::SecretSharedMap>,
}

impl<T> CoSolver<T, ark_bn254::Fr>
where
    T: NoirWitnessExtensionProtocol<ark_bn254::Fr>,
{
    pub fn read_abi_bn254_fieldelement<P>(
        path: P,
        abi: &Abi,
    ) -> eyre::Result<WitnessMap<FieldElement>>
    where
        PathBuf: From<P>,
    {
        if abi.is_empty() {
            Ok(WitnessMap::default())
        } else {
            let input_string = std::fs::read_to_string(PathBuf::from(path))?;
            let mut input_map = Format::Toml.parse(&input_string, abi)?;
            let return_value = input_map.remove(MAIN_RETURN_NAME);
            // TACEO TODO the return value can be none for the witness extension
            // do we want to keep it like that? Seems not necessary but maybe
            // we need it for proving/verifying
            Ok(abi.encode(&input_map, return_value.clone())?)
        }
    }

    // This is the same as read_abi_bn254_fieldelement, but only warns if parameters are missing instead of throwing an error and returns a map with strings instead
    pub fn partially_read_abi_bn254_fieldelement<P>(
        path: P,
        abi: &Abi,
    ) -> eyre::Result<BTreeMap<String, FieldElement>>
    where
        PathBuf: From<P>,
    {
        if abi.is_empty() {
            Ok(BTreeMap::default())
        } else {
            let input_string = std::fs::read_to_string(PathBuf::from(path))?;
            let abi_ = Self::create_partial_abi(&input_string, abi)?;
            let mut input_map = Format::Toml.parse(&input_string, &abi_)?;
            let return_value = input_map.remove(MAIN_RETURN_NAME);
            // TACEO TODO the return value can be none for the witness extension
            // do we want to keep it like that? Seems not necessary but maybe
            // we need it for proving/verifying
            let encoded = abi_.encode(&input_map, return_value.clone())?;
            Ok(Self::create_string_map(&abi_, encoded)?)
        }
    }

    pub fn read_abi_bn254<P>(path: P, abi: &Abi) -> eyre::Result<WitnessMap<T::AcvmType>>
    where
        PathBuf: From<P>,
    {
        let initial_witness = Self::read_abi_bn254_fieldelement(path, abi)?;
        let mut witnesses = WitnessMap::<T::AcvmType>::default();
        for (witness, v) in initial_witness.into_iter() {
            witnesses.insert(witness, T::AcvmType::from(v.into_repr()));
        }
        Ok(witnesses)
    }

    pub fn new_bn254<P>(
        driver: T,
        compiled_program: ProgramArtifact,
        prover_path: P,
    ) -> eyre::Result<Self>
    where
        PathBuf: From<P>,
    {
        let mut witness_map =
            vec![WitnessMap::default(); compiled_program.bytecode.functions.len()];
        witness_map[0] = Self::read_abi_bn254(prover_path, &compiled_program.abi)?;
        Ok(Self {
            driver,
            abi: compiled_program.abi,
            functions: compiled_program
                .bytecode
                .functions
                .into_iter()
                // ignore the transformation mapping for now
                .map(|function| acvm::compiler::transform(function, CO_EXPRESSION_WIDTH).0)
                .collect::<Vec<_>>(),
            witness_map,
            function_index: 0,
            memory_access: IntMap::new(),
        })
    }

    pub fn new_bn254_with_witness(
        driver: T,
        compiled_program: ProgramArtifact,
        witness: WitnessMap<T::AcvmType>,
    ) -> eyre::Result<Self> {
        let mut witness_map =
            vec![WitnessMap::default(); compiled_program.bytecode.functions.len()];
        witness_map[0] = witness;
        Ok(Self {
            driver,
            abi: compiled_program.abi,
            functions: compiled_program
                .bytecode
                .functions
                .into_iter()
                // ignore the transformation mapping for now
                .map(|function| acvm::compiler::transform(function, CO_EXPRESSION_WIDTH).0)
                .collect::<Vec<_>>(),
            witness_map,
            function_index: 0,
            memory_access: IntMap::new(),
        })
    }
}

impl<N: Rep3Network> Rep3CoSolver<ark_bn254::Fr, N> {
    pub fn from_network<P>(
        network: N,
        compiled_program: ProgramArtifact,
        prover_path: P,
    ) -> eyre::Result<Self>
    where
        PathBuf: From<P>,
    {
        Self::new_bn254(Rep3AcvmSolver::new(network), compiled_program, prover_path)
    }

    pub fn from_network_with_witness(
        network: N,
        compiled_program: ProgramArtifact,
        witness: WitnessMap<
            <Rep3AcvmSolver<ark_bn254::Fr, N> as NoirWitnessExtensionProtocol::<ark_bn254::Fr>>::AcvmType,
        >,
    ) -> eyre::Result<Self> {
        Self::new_bn254_with_witness(Rep3AcvmSolver::new(network), compiled_program, witness)
    }
}

impl<F: PrimeField> PlainCoSolver<F> {
    pub fn convert_to_plain_acvm_witness(
        mut shared_witness: WitnessStack<F>,
    ) -> WitnessStack<GenericFieldElement<F>> {
        let length = shared_witness.length();
        let mut vec = Vec::with_capacity(length);
        for _ in 0..length {
            let stack_item = shared_witness.pop().unwrap();
            vec.push((
                stack_item.index,
                stack_item
                    .witness
                    .into_iter()
                    .map(|(k, v)| (k, GenericFieldElement::from_repr(v)))
                    .collect::<BTreeMap<_, _>>(),
            ))
        }
        let mut witness = WitnessStack::default();
        //push again in reverse order
        for (index, witness_map) in vec.into_iter().rev() {
            witness.push(index, WitnessMap::from(witness_map));
        }
        witness
    }
}

impl PlainCoSolver<ark_bn254::Fr> {
    pub fn init_plain_driver<P>(
        compiled_program: ProgramArtifact,
        prover_path: P,
    ) -> eyre::Result<Self>
    where
        PathBuf: From<P>,
    {
        Self::new_bn254(PlainAcvmSolver::default(), compiled_program, prover_path)
    }

    pub fn solve_and_print_output(self) {
        let abi = self.abi.clone();
        let result = self.solve().unwrap();
        let mut result = Self::convert_to_plain_acvm_witness(result);
        let main_witness = result.pop().unwrap();
        let (_, ret_val) = abi.decode(&main_witness.witness).unwrap();
        if let Some(ret_val) = ret_val {
            println!("circuit produced: {ret_val:?}");
        } else {
            println!("no output for circuit")
        }
    }
}

impl<T, F> CoSolver<T, F>
where
    T: NoirWitnessExtensionProtocol<F>,
    F: PrimeField,
{
    #[inline(always)]
    fn witness(&mut self) -> &mut WitnessMap<T::AcvmType> {
        &mut self.witness_map[self.function_index]
    }
}

impl<T, F> CoSolver<T, F>
where
    T: NoirWitnessExtensionProtocol<F>,
    F: PrimeField,
{
    fn open_results(&mut self, function: &Circuit<GenericFieldElement<F>>) -> CoAcvmResult<()> {
        let witness_map = &mut self.witness_map[self.function_index];

        let mut vec = Vec::with_capacity(function.return_values.0.len());
        for index in function.return_values.0.iter() {
            let val = witness_map.get(index).expect("witness should be present");
            if let Some(val) = T::get_shared(val) {
                vec.push(val.clone());
            };
        }
        let mut opened = self.driver.open_many(&vec)?;
        for index in function.return_values.0.iter().rev() {
            let val = witness_map.get(index).expect("witness should be present");
            if T::is_shared(val) {
                let opened_val = opened.pop().expect("opened value should be present");
                witness_map.insert(*index, T::AcvmType::from(opened_val));
            }
        }
        Ok(())
    }

    pub fn solve(mut self) -> CoAcvmResult<WitnessStack<T::AcvmType>> {
        let functions = std::mem::take(&mut self.functions);

        for opcode in functions[self.function_index].opcodes.iter() {
            match opcode {
                Opcode::AssertZero(expr) => self.solve_assert_zero(expr)?,
                Opcode::MemoryInit {
                    block_id,
                    init,
                    block_type: _, // apparently not used
                } => self.solve_memory_init_block(*block_id, init)?,
                Opcode::MemoryOp {
                    block_id,
                    op,
                    predicate,
                } => self.solve_memory_op(*block_id, op, predicate.to_owned())?,
                _ => todo!("non assert zero opcode detected, not supported yet"),
            }
        }
        tracing::trace!("we are done! Opening results...");
        self.open_results(&functions[self.function_index])?;
        tracing::trace!("Done! Wrap things up.");
        let mut witness_stack = WitnessStack::default();
        for (idx, witness) in self.witness_map.into_iter().rev().enumerate() {
            witness_stack.push(u32::try_from(idx).expect("usize fits into u32"), witness);
        }
        Ok(witness_stack)
    }
}
