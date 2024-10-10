use std::{collections::HashMap, sync::Arc};

use ark_ff::PrimeField;
use mpc_core::protocols::rep3::network::{Rep3MpcNet, Rep3Network};
use mpc_net::config::NetworkConfig;

use crate::{
    accelerator::MpcAccelerator,
    mpc::plain::CircomPlainVmWitnessExtension,
    mpc_vm::{PlainWitnessExtension, Rep3WitnessExtension, VMConfig, WitnessExtension},
    op_codes::CodeBlock,
};
use eyre::Result;

/// A template declaration.
///
/// Stores all necessary information to create a component, including the [`CodeBlock`],
/// the number of input signals, sub-components, and vars.
///
/// > **Warning**: Users should usually not interact directly with this struct. It is only public because the
/// > compiler requires these declarations, and the compiler is a separate crate due to licensing constraints.
#[derive(Clone)]
pub struct TemplateDecl {
    pub(crate) symbol: String,
    pub(crate) input_signals: usize,
    pub(crate) sub_components: usize,
    pub(crate) vars: usize,
    pub(crate) mappings: Vec<usize>,
    pub(crate) body: Arc<CodeBlock>,
}

impl TemplateDecl {
    /// Creates a new template declaration. Only the MPC-compiler should use this method!
    pub fn new(
        symbol: String,
        input_signals: usize,
        sub_components: usize,
        vars: usize,
        mappings: Vec<usize>,
        body: CodeBlock,
    ) -> Self {
        Self {
            symbol,
            input_signals,
            sub_components,
            vars,
            mappings,
            body: Arc::new(body),
        }
    }
}

/// An unconstrained function declaration.
///
/// Stores all necessary information to call the function, including the [`CodeBlock`],
/// the number of params, and vars.
///
/// > **Warning**: Users should usually not interact directly with this struct. It is only public because the
/// > compiler requires these declarations, and the compiler is a separate crate due to licensing constraints.
#[derive(Clone)]
pub struct FunDecl {
    pub(crate) num_params: usize,
    pub(crate) vars: usize,
    pub(crate) body: Arc<CodeBlock>,
}

impl FunDecl {
    /// Creates a new function declaration. Only the MPC-compiler should use this method!
    pub fn new(num_params: usize, vars: usize, body: CodeBlock) -> Self {
        Self {
            num_params,
            vars,
            body: Arc::new(body),
        }
    }
}

/// A type that stores the name of an output signal and maps it to
/// the respective offset in the witness.
///
/// String -> (offset, size)
pub type OutputMapping = HashMap<String, (usize, usize)>;

pub(crate) type InputList = Vec<(String, usize, usize)>;

/// The state of the compiler after it parsed the circom file.
///
/// The struct provides certain methods to consume it and create an
/// [MPC-VM](WitnessExtension).
#[derive(Clone)]
pub struct CoCircomCompilerParsed<F: PrimeField> {
    pub(crate) main: String,
    pub(crate) amount_signals: usize,
    pub(crate) constant_table: Vec<F>,
    pub(crate) string_table: Vec<String>,
    pub(crate) fun_decls: HashMap<String, FunDecl>,
    pub(crate) templ_decls: HashMap<String, TemplateDecl>,
    pub(crate) signal_to_witness: Vec<usize>,
    pub(crate) main_inputs: usize,
    pub(crate) main_outputs: usize,
    pub(crate) main_input_list: InputList,
    pub(crate) output_mapping: OutputMapping,
}

impl<F: PrimeField> CoCircomCompilerParsed<F> {
    /// > **Warning**: DO NOT CALL THIS DIRECTLY! This struct is intended for internal use by the compiler crate
    /// > and should not be instantiated directly. It is publicly visible due to requirements imposed by licensing constraints.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        main: String,
        amount_signals: usize,
        constant_table: Vec<F>,
        string_table: Vec<String>,
        fun_decls: HashMap<String, FunDecl>,
        templ_decls: HashMap<String, TemplateDecl>,
        signal_to_witness: Vec<usize>,
        main_inputs: usize,
        main_outputs: usize,
        main_input_list: InputList,
        output_mapping: OutputMapping,
    ) -> Self {
        Self {
            main,
            amount_signals,
            constant_table,
            string_table,
            fun_decls,
            templ_decls,
            signal_to_witness,
            main_inputs,
            main_outputs,
            main_input_list,
            output_mapping,
        }
    }
}

//TODO: Add another builder step here?
//ParserCompiler -> into Rep3/Shamir -> build
impl<F: PrimeField> CoCircomCompilerParsed<F> {
    /// Consumes `self` and constructs an instance of [`PlainWitnessExtension`].
    ///
    /// The plain witness extension allows local execution of the witness extension without
    /// using MPC. Be cautious when using this method, as the resulting
    /// witness and input will not be protected. Do not share sensitive data when using this feature.
    ///
    /// This method is primarily intended for testing purposes.
    pub fn to_plain_vm(
        self,
        vm_config: VMConfig,
    ) -> WitnessExtension<F, CircomPlainVmWitnessExtension<F>> {
        PlainWitnessExtension::new(self, vm_config)
    }

    /// Consumes `self` and a [`NetworkConfig`], and constructs an instance of [`Rep3WitnessExtension`].
    ///
    /// # Arguments
    /// - `network_config`: A network configuration specifying how to connect to the other two parties.
    ///
    /// # Returns
    /// - `Ok(Rep3WitnessExtension)`: The MPC-VM capable of performing the witness extension using the Rep3 protocol.
    /// - `Err(err)`: An error indicating a failure, such as inability to connect to the other parties.
    pub fn to_rep3_vm(
        self,
        network_config: NetworkConfig,
        vm_config: VMConfig,
    ) -> Result<Rep3WitnessExtension<F, Rep3MpcNet>> {
        Rep3WitnessExtension::new(
            self,
            network_config,
            MpcAccelerator::full_mpc_accelerator(),
            vm_config,
        )
    }

    /// Consumes `self` and an already established [`Rep3Network`], and constructs an instance of [`Rep3WitnessExtension`].
    ///
    /// # Arguments
    /// - `network`: Am already established [`Rep3Network`].
    ///
    /// # Returns
    /// - `Ok(Rep3WitnessExtension)`: The MPC-VM capable of performing the witness extension using the Rep3 protocol.
    /// - `Err(err)`: An error indicating a failure.
    pub fn to_rep3_vm_with_network<N: Rep3Network>(
        self,
        network: N,
        vm_config: VMConfig,
    ) -> Result<Rep3WitnessExtension<F, N>> {
        Rep3WitnessExtension::from_network(
            self,
            network,
            MpcAccelerator::full_mpc_accelerator(),
            vm_config,
        )
    }
}
