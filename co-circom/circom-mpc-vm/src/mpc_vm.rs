use crate::mpc::plain::CircomPlainVmWitnessExtension;
use crate::mpc::rep3::{CircomRep3VmWitnessExtension, Rep3VmType};
use crate::types::{CoCircomCompilerParsed, FunDecl, InputList, OutputMapping, TemplateDecl};

use super::accelerator::MpcAccelerator;
use super::{
    op_codes::{self, CodeBlock},
    stack::Stack,
};
use ark_ff::PrimeField;
use co_circom_snarks::{SharedInput, SharedWitness};
use eyre::{bail, eyre, Result};
use itertools::{izip, Itertools};
use mpc_core::protocols::rep3::network::{Rep3MpcNet, Rep3Network};
use mpc_net::config::NetworkConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::mpc::VmCircomWitnessExtension;

/// The mpc-vm configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct VMConfig {
    /// Allow leaking of secret values in logs
    #[serde(default)]
    pub allow_leaky_logs: bool,
}

/// The MPC-VM that performs the witness extension.
///
/// This struct can only be instantiated by constructing it with a [`CoCircomCompilerParsed`].
/// There are two main functions of interest:
///
/// - [`run()`](WitnessExtension::run): Executes the witness extension.
/// - [`run_with_flat()`](WitnessExtension::run_with_flat): Executes the witness extension with flattened inputs.
pub struct WitnessExtension<F: PrimeField, C: VmCircomWitnessExtension<F>> {
    main: String,
    ctx: WitnessExtensionCtx<F, C>,
    signal_to_witness: Vec<usize>,
    main_inputs: usize,
    main_outputs: usize,
    main_input_list: InputList,
    output_mapping: OutputMapping,
    driver: C,
    config: VMConfig,
}

/// Shorthand type for an instance of the MPC-VM that runs locally on a single machine without MPC.
///
/// This type is mostly used for testing purposes, so use with care in production environments.
pub type PlainWitnessExtension<F> = WitnessExtension<F, CircomPlainVmWitnessExtension<F>>;

/// Shorthand type for the MPC-VM instantiated with the [`Rep3Protocol`].
///
/// This is the only supported protocol at the moment.
pub type Rep3WitnessExtension<F, N> = WitnessExtension<F, CircomRep3VmWitnessExtension<F, N>>;

type ConsumedFunCtx<T> = (usize, usize, Vec<T>, Arc<CodeBlock>, Vec<(T, Vec<T>)>);

#[derive(Default, Clone)]
struct IfCtxStack<F: PrimeField, C: VmCircomWitnessExtension<F>>(Vec<IfCtx<F, C>>);

#[derive(Default, Clone)]
struct Component<F: PrimeField, C: VmCircomWitnessExtension<F>> {
    symbol: String,
    amount_vars: usize,
    provided_input_signals: usize,
    input_signals: usize,
    current_return_vals: usize,
    /// the offset inside the signals array
    my_offset: usize,
    field_stack: Stack<C::VmType>,
    index_stack: Stack<usize>,
    if_stack: IfCtxStack<F, C>,
    functions_ctx: Stack<FunctionCtx<C::VmType>>,
    mappings: Vec<usize>,
    sub_components: Vec<Component<F, C>>,
    component_body: Arc<CodeBlock>,
    log_buf: String,
}

struct WitnessExtensionCtx<F: PrimeField, C: VmCircomWitnessExtension<F>> {
    signals: Vec<C::VmType>,
    fun_decls: HashMap<String, FunDecl>,
    templ_decls: HashMap<String, TemplateDecl>,
    constant_table: Vec<C::VmType>,
    string_table: Vec<String>,
    mpc_accelerator: MpcAccelerator<F, C>,
}

#[derive(Clone)]
enum IfCtx<F: PrimeField, C: VmCircomWitnessExtension<F>> {
    Public,
    Shared(C::VmType, C::VmType, C::VmType),
}

impl<F: PrimeField, C: VmCircomWitnessExtension<F>> IfCtxStack<F, C> {
    fn new() -> Self {
        Self(vec![])
    }

    fn is_shared(&self) -> bool {
        self.0
            .iter()
            .any(|cond| matches!(cond, IfCtx::Shared(_, _, _)))
    }

    fn get_shared_condition(&self) -> C::VmType {
        if let Some(IfCtx::Shared(_, acc_condition, _)) = self
            .0
            .iter()
            .rev()
            .find(|c| matches!(c, IfCtx::Shared(_, _, _)))
        {
            acc_condition.clone()
        } else {
            panic!("must be there");
        }
    }

    fn peek(&self) -> &IfCtx<F, C> {
        self.0.last().expect("must be here")
    }

    fn pop(&mut self) {
        self.0.pop().expect("must be here");
    }

    fn push_shared(&mut self, protocol: &mut C, cond: C::VmType) -> Result<()> {
        //find last shared
        if let Some(IfCtx::Shared(_, acc_condition, _)) = self
            .0
            .iter()
            .rev()
            .find(|c| matches!(c, IfCtx::Shared(_, _, _)))
        {
            let combined = protocol.bool_and(acc_condition.clone(), cond.clone())?;
            self.0
                .push(IfCtx::Shared(acc_condition.to_owned(), combined, cond));
        } else {
            //first shared - set last condition to 1
            let last_condition = protocol.public_one();
            let acc_condition = protocol.bool_and(last_condition.clone(), cond.clone())?;
            self.0
                .push(IfCtx::Shared(last_condition, acc_condition, cond));
        }
        Ok(())
    }

    fn toggle_last_shared(&mut self, protocol: &mut C) -> Result<()> {
        if let Some(IfCtx::Shared(last_condition, acc_condition, current_cond)) = self
            .0
            .iter_mut()
            .rev()
            .find(|c| matches!(c, IfCtx::Shared(_, _, _)))
        {
            let toggled_current_cond = protocol.bool_not(current_cond.to_owned())?;
            *acc_condition = protocol.bool_and(last_condition.to_owned(), toggled_current_cond)?;
        } else {
            panic!("last must be shared");
        }
        Ok(())
    }

    fn push_public(&mut self) {
        self.0.push(IfCtx::Public);
    }
}

impl<F: PrimeField, C: VmCircomWitnessExtension<F>> WitnessExtensionCtx<F, C> {
    fn new(
        signals: Vec<C::VmType>,
        constant_table: Vec<C::VmType>,
        fun_decls: HashMap<String, FunDecl>,
        templ_decls: HashMap<String, TemplateDecl>,
        string_table: Vec<String>,
        mpc_accelerator: MpcAccelerator<F, C>,
    ) -> Self {
        Self {
            signals,
            constant_table,
            fun_decls,
            templ_decls,
            string_table,
            mpc_accelerator,
        }
    }
}

#[derive(Clone)]
struct FunctionCtx<T> {
    ip: usize,
    return_vals: usize,
    vars: Vec<T>,
    body: Arc<CodeBlock>,
    shared_return_vals: Vec<(T, Vec<T>)>,
}

impl<T> FunctionCtx<T> {
    fn new(
        ip: usize,
        return_vals: usize,
        vars: Vec<T>,
        body: Arc<CodeBlock>,
        shared_return_vals: Vec<(T, Vec<T>)>,
    ) -> Self {
        Self {
            ip,
            return_vals,
            vars,
            body,
            shared_return_vals,
        }
    }

    fn consume(self) -> ConsumedFunCtx<T> {
        (
            self.ip,
            self.return_vals,
            self.vars,
            self.body,
            self.shared_return_vals,
        )
    }
}

impl<F: PrimeField, C: VmCircomWitnessExtension<F>> Component<F, C> {
    fn init(templ_decl: &TemplateDecl, signal_offset: usize) -> Self {
        Self {
            symbol: templ_decl.symbol.clone(),
            amount_vars: templ_decl.vars,
            provided_input_signals: 0,
            input_signals: templ_decl.input_signals,
            current_return_vals: 0,
            my_offset: signal_offset,
            field_stack: Stack::default(),
            index_stack: Stack::default(),
            if_stack: IfCtxStack::new(),
            functions_ctx: Stack::default(),
            mappings: templ_decl.mappings.clone(),
            sub_components: Vec::with_capacity(templ_decl.sub_components),
            component_body: Arc::clone(&templ_decl.body),
            log_buf: String::with_capacity(1024),
        }
    }

    #[inline]
    fn push_field(&mut self, val: C::VmType) {
        self.field_stack.push(val)
    }

    #[inline(always)]
    fn pop_field(&mut self) -> C::VmType {
        self.field_stack.pop()
    }

    #[inline(always)]
    fn push_index(&mut self, val: usize) {
        self.index_stack.push(val)
    }

    #[inline(always)]
    fn pop_index(&mut self) -> usize {
        self.index_stack.pop()
    }

    #[allow(dead_code)]
    fn debug_code_block(code_block: Arc<CodeBlock>) {
        for (idx, inst) in code_block.iter().enumerate() {
            tracing::info!("{idx:0>4}|   {inst}");
        }
    }
    pub fn run(
        &mut self,
        protocol: &mut C,
        ctx: &mut WitnessExtensionCtx<F, C>,
        config: &VMConfig,
    ) -> Result<()> {
        let mut ip = 0;
        let mut current_body = Arc::clone(&self.component_body);
        let mut current_vars = vec![C::VmType::default(); self.amount_vars];
        let mut current_shared_ret_vals = vec![];
        loop {
            let inst = &current_body[ip];
            tracing::trace!("{ip:0>4}|   {inst}");
            match inst {
                op_codes::MpcOpCode::PushConstant(index) => {
                    let constant = ctx.constant_table[*index].clone();
                    tracing::debug!("pushing constant {}", constant);
                    self.push_field(constant);
                }
                op_codes::MpcOpCode::PushIndex(index) => self.push_index(*index),
                op_codes::MpcOpCode::LoadSignals(amount) => {
                    let index = self.pop_index();
                    let start = self.my_offset + index;
                    ctx.signals[start..start + amount]
                        .iter()
                        .cloned()
                        .for_each(|signal| {
                            tracing::debug!("pushing signal {signal}");
                            self.push_field(signal);
                        });
                }
                op_codes::MpcOpCode::StoreSignals(amount) => {
                    //get index
                    let index = self.pop_index();
                    if self.if_stack.is_shared() {
                        let shared_condition = self.if_stack.get_shared_condition();
                        for i in 0..*amount {
                            let old = ctx.signals[self.my_offset + index + amount - i - 1].clone();
                            let new = self.pop_field();
                            ctx.signals[self.my_offset + index + amount - i - 1] =
                                protocol.cmux(shared_condition.clone(), new, old)?;
                        }
                    } else {
                        for i in 0..*amount {
                            ctx.signals[self.my_offset + index + amount - i - 1] = self.pop_field();
                        }
                    }
                }
                op_codes::MpcOpCode::LoadVars(amount) => {
                    let index = self.pop_index();
                    current_vars[index..index + amount]
                        .iter()
                        .cloned()
                        .for_each(|signal| {
                            self.push_field(signal);
                        });
                }
                op_codes::MpcOpCode::StoreVars(amount) => {
                    let index = self.pop_index();
                    if self.if_stack.is_shared() {
                        let cond = self.if_stack.get_shared_condition();
                        for i in 0..*amount {
                            let old = current_vars[index + amount - i - 1].clone();
                            current_vars[index + amount - i - 1] =
                                protocol.cmux(cond.clone(), self.pop_field(), old)?;
                        }
                    } else {
                        for i in 0..*amount {
                            current_vars[index + amount - i - 1] = self.pop_field();
                        }
                    }
                }
                op_codes::MpcOpCode::Call(symbol, return_vals) => {
                    tracing::debug!("Calling {symbol}");
                    let fun_decl = ctx.fun_decls.get(symbol).ok_or(eyre!(
                        "{symbol} not found in function declaration. This must be a bug.."
                    ))?;
                    let to_copy = self.field_stack.frame_len() - fun_decl.num_params;
                    if ctx.mpc_accelerator.has_accelerator(symbol) {
                        tracing::debug!("calling accelerator for {symbol}");
                        //call the accelerator
                        let mut result = ctx.mpc_accelerator.run_accelerator(
                            symbol,
                            protocol,
                            &self.field_stack.peek_stack_frame()[to_copy..],
                        )?;
                        //TODO we need to perform a full ReturnFun here with shared returns and with arrays
                        //for time being we support only sqrt therefore just assert that and push on stack
                        assert_eq!(result.len(), 1);
                        self.push_field(result.pop().unwrap());
                    } else {
                        let mut func_vars = vec![C::VmType::default(); fun_decl.vars];
                        //copy the parameters
                        for (idx, param) in self.field_stack.peek_stack_frame()[to_copy..]
                            .iter()
                            .enumerate()
                        {
                            func_vars[idx] = param.clone();
                        }
                        std::mem::swap(&mut func_vars, &mut current_vars);
                        let mut next_shared_ret_vals = vec![];
                        std::mem::swap(&mut current_shared_ret_vals, &mut next_shared_ret_vals);
                        self.index_stack.push_stack_frame();
                        self.field_stack.push_stack_frame();
                        self.functions_ctx.push(FunctionCtx::new(
                            ip,
                            self.current_return_vals,
                            func_vars,
                            Arc::clone(&current_body),
                            next_shared_ret_vals,
                        ));
                        //set size of return value
                        self.current_return_vals = *return_vals;
                        current_body = Arc::clone(&fun_decl.body);
                        ip = 0;
                        continue;
                    }
                }
                op_codes::MpcOpCode::CreateCmp(symbol, amount) => {
                    let new_components = {
                        let offset_jump = self.pop_index();
                        let relative_offset = self.pop_index();
                        let templ_decl = ctx.templ_decls.get(symbol).ok_or(eyre!(
                            "{symbol} not found in template declarations. This must be a bug"
                        ))?;
                        let mut offset = self.my_offset + relative_offset;
                        (0..*amount)
                            .map(|i| {
                                if i != 0 {
                                    offset += offset_jump;
                                }
                                Component::<F, C>::init(templ_decl, offset)
                            })
                            .collect_vec()
                    };
                    //check if we can run it instantly
                    for mut component in new_components {
                        if component.input_signals == 0 {
                            component.run(protocol, ctx, config)?;
                        }
                        self.sub_components.push(component);
                    }
                }
                op_codes::MpcOpCode::OutputSubComp(mapped, signal_code, amount) => {
                    let sub_comp_index = self.pop_index();
                    let mut index = self.pop_index();
                    let component = &mut self.sub_components[sub_comp_index];
                    if *mapped {
                        index += component.mappings[*signal_code];
                    }
                    let offset_in_component = component.my_offset + index;
                    for ele in &ctx.signals[offset_in_component..offset_in_component + (*amount)] {
                        self.push_field(ele.clone());
                    }
                }
                op_codes::MpcOpCode::InputSubComp(mapped, signal_code, amount) => {
                    assert!(
                        !self.if_stack.is_shared(),
                        "Cannot be shared when providing inputs for sub component"
                    );
                    let sub_comp_index = self.pop_index();
                    let mut index = self.pop_index();
                    //we cannot borrow later therefore we need to pop from stack here and push later
                    let mut input_signals = vec![C::VmType::default(); *amount];
                    for i in 0..*amount {
                        input_signals[*amount - i - 1] = self.pop_field();
                        tracing::debug!("poping {}", input_signals.last().unwrap());
                    }

                    let component = &mut self.sub_components[sub_comp_index];
                    if *mapped {
                        index += component.mappings[*signal_code];
                    }
                    let offset_in_component = component.my_offset + index;
                    ctx.signals[offset_in_component..offset_in_component + *amount]
                        .clone_from_slice(&input_signals);
                    component.provided_input_signals += amount;
                    if component.provided_input_signals == component.input_signals {
                        component.run(protocol, ctx, config)?;
                    }
                }
                op_codes::MpcOpCode::Assert(line) => {
                    let assertion = self.pop_field();
                    if protocol.is_zero(assertion, true)? {
                        bail!(
                            "Assertion failed during execution on line {line} in component {}",
                            self.symbol
                        );
                    }
                }
                op_codes::MpcOpCode::If(jump) => {
                    let cond = self.pop_field();
                    if protocol.is_shared(&cond)? {
                        //push the new shared condition on stack
                        self.if_stack.push_shared(protocol, cond)?;
                    } else {
                        self.if_stack.push_public();
                        //not shared we can just check
                        if protocol.is_zero(cond, false)? {
                            //we need to jump
                            ip += jump;
                            continue;
                        }
                    }
                }
                op_codes::MpcOpCode::EndTruthyBranch(jump) => {
                    let if_ctx = self.if_stack.peek();
                    match if_ctx {
                        IfCtx::Public => {
                            //we need to jump and pop the element
                            ip += std::cmp::max(1, *jump);
                            self.if_stack.pop();
                            continue;
                        }
                        IfCtx::Shared(_, _, _) => {
                            if *jump == 0 {
                                //no else branch
                                self.if_stack.pop();
                            } else {
                                self.if_stack.toggle_last_shared(protocol)?;
                            }
                        }
                    }
                }
                op_codes::MpcOpCode::EndFalsyBranch => {
                    self.if_stack.pop();
                }
                op_codes::MpcOpCode::Add => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.add(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Sub => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.sub(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Mul => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.mul(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Div => {
                    let mut rhs = self.pop_field();
                    let lhs = self.pop_field();
                    if self.if_stack.is_shared() {
                        let cond = self.if_stack.get_shared_condition();
                        rhs = protocol.cmux(cond, rhs, protocol.public_one())?;
                    }
                    self.push_field(protocol.div(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Pow => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.pow(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Mod => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.modulo(lhs, rhs)?);
                }

                op_codes::MpcOpCode::Neg => {
                    let x = self.pop_field();
                    self.push_field(protocol.neg(x)?);
                }
                op_codes::MpcOpCode::IntDiv => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.int_div(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Lt => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.lt(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Le => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.le(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Gt => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.gt(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Ge => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.ge(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Eq => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.eq(lhs, rhs)?);
                }
                op_codes::MpcOpCode::Neq => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.neq(lhs, rhs)?);
                }
                op_codes::MpcOpCode::ShiftR => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.shift_r(lhs, rhs)?);
                }
                op_codes::MpcOpCode::ShiftL => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.shift_l(lhs, rhs)?);
                }
                op_codes::MpcOpCode::BoolOr => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.bool_or(lhs, rhs)?);
                }
                op_codes::MpcOpCode::BoolAnd => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.bool_and(lhs, rhs)?);
                }
                op_codes::MpcOpCode::BitOr => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.bit_or(lhs, rhs)?);
                }
                op_codes::MpcOpCode::BitAnd => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.bit_and(lhs, rhs)?);
                }
                op_codes::MpcOpCode::BitXOr => {
                    let rhs = self.pop_field();
                    let lhs = self.pop_field();
                    self.push_field(protocol.bit_xor(lhs, rhs)?);
                }
                op_codes::MpcOpCode::AddIndex => {
                    let rhs = self.pop_index();
                    let lhs = self.pop_index();
                    self.push_index(lhs + rhs);
                }
                op_codes::MpcOpCode::MulIndex => {
                    let rhs = self.pop_index();
                    let lhs = self.pop_index();
                    self.push_index(lhs * rhs);
                }
                op_codes::MpcOpCode::ToIndex => {
                    let signal = self.pop_field();
                    self.push_index(protocol.to_index(signal)?);
                }
                op_codes::MpcOpCode::JumpBack(jump_backward) => {
                    ip -= jump_backward;
                    continue;
                }

                op_codes::MpcOpCode::JumpIfFalse(jump_forward) => {
                    let jump_to = jump_forward;
                    let cond = self.pop_field();
                    if protocol.is_zero(cond, false)? {
                        ip += jump_to;
                        continue;
                    }
                }
                op_codes::MpcOpCode::Return => {
                    //we are done
                    //just return
                    break;
                }
                op_codes::MpcOpCode::ReturnFun => {
                    //check if we have multiple return values
                    if self.current_return_vals == 1 {
                        //check if we currently in an shared if
                        if self.if_stack.is_shared() {
                            //we need to store the return val for later use
                            let mut this_condition = self.if_stack.get_shared_condition();
                            for (cond, _) in current_shared_ret_vals.iter() {
                                let neg_cond = protocol.bool_not(cond.to_owned())?;
                                this_condition = protocol.bool_and(this_condition, neg_cond)?;
                            }
                            let result = self.pop_field();
                            current_shared_ret_vals
                                .push((this_condition.clone(), vec![result.clone()]));
                            //we need to continue
                            ip += 1;
                            continue;
                        } else if !current_shared_ret_vals.is_empty() {
                            // we return for sure here but we need to check if we return this value
                            // or we need to short circuit
                            let mut this_condition = protocol.public_one();
                            for (cond, _) in current_shared_ret_vals.iter() {
                                let neg_cond = protocol.bool_not(cond.to_owned())?;
                                this_condition = protocol.bool_and(this_condition, neg_cond)?;
                            }
                            current_shared_ret_vals.push((this_condition, vec![self.pop_field()]));
                            self.handle_shared_fun_return(protocol, &current_shared_ret_vals)?;
                        } else {
                            //copy the return value
                            self.field_stack
                                .pop_stack_frame()
                                .into_iter()
                                .for_each(|signal| {
                                    self.push_field(signal);
                                });
                        }
                    } else {
                        let start = self.pop_index();
                        let end = self.current_return_vals;
                        //check whether we need to pad some return values
                        //if we return an array with different sizes
                        if current_vars.len() < start + end {
                            current_vars.resize(start + end, protocol.public_zero());
                        }
                        if self.if_stack.is_shared() {
                            //we need to store the return val for later use
                            let mut this_condition = self.if_stack.get_shared_condition();
                            for (cond, _) in current_shared_ret_vals.iter() {
                                let neg_cond = protocol.bool_not(cond.to_owned())?;
                                this_condition = protocol.bool_and(this_condition, neg_cond)?;
                            }
                            current_shared_ret_vals.push((
                                this_condition,
                                current_vars[start..start + end]
                                    .iter()
                                    .cloned()
                                    .collect_vec(),
                            ));
                            //we need to continue
                            ip += 1;
                            continue;
                        } else if !current_shared_ret_vals.is_empty() {
                            // we return for sure here but we need to check if we return this value
                            // or we need to short circuit
                            let mut this_condition = protocol.public_one();
                            for (cond, _) in current_shared_ret_vals.iter() {
                                let neg_cond = protocol.bool_not(cond.to_owned())?;
                                this_condition = protocol.bool_and(this_condition, neg_cond)?;
                            }
                            current_shared_ret_vals.push((
                                this_condition,
                                current_vars[start..start + end]
                                    .iter()
                                    .cloned()
                                    .collect_vec(),
                            ));
                            self.handle_shared_fun_return(protocol, &current_shared_ret_vals)?;
                        } else {
                            self.index_stack.pop_stack_frame();
                            self.field_stack.pop_stack_frame();
                            current_vars[start..start + end]
                                .iter()
                                .cloned()
                                .for_each(|var| {
                                    self.push_field(var);
                                });
                        }
                    }
                    let (old_ip, old_return_vals, mut old_vars, old_body, shared_return_vals) =
                        self.functions_ctx.pop().consume();
                    ip = old_ip;
                    self.current_return_vals = old_return_vals;
                    current_shared_ret_vals = shared_return_vals;
                    std::mem::swap(&mut current_vars, &mut old_vars);
                    current_body = old_body;
                }
                op_codes::MpcOpCode::ReturnSharedIfFun => {
                    self.handle_shared_fun_return(protocol, &current_shared_ret_vals)?;
                    let (old_ip, old_return_vals, mut old_vars, old_body, shared_return_vals) =
                        self.functions_ctx.pop().consume();
                    ip = old_ip;
                    self.current_return_vals = old_return_vals;
                    current_shared_ret_vals = shared_return_vals;
                    std::mem::swap(&mut current_vars, &mut old_vars);
                    current_body = old_body;
                }
                op_codes::MpcOpCode::Log => {
                    if config.allow_leaky_logs {
                        let field = protocol.open(self.pop_field())?;
                        self.log_buf.push_str(&field.to_string());
                    } else {
                        self.log_buf.push_str("secret");
                    }
                    self.log_buf.push(' ');
                }
                op_codes::MpcOpCode::LogString(idx) => {
                    if *idx >= ctx.string_table.len() {
                        bail!(
                            "trying to access string on pos: {idx} but len is {}",
                            ctx.string_table.len()
                        );
                    }
                    self.log_buf.push_str(&ctx.string_table[*idx]);
                    self.log_buf.push(' ');
                }
                op_codes::MpcOpCode::LogFlush(line) => {
                    tracing::info!("line {line:0>4}: {}", self.log_buf);
                    self.log_buf.clear();
                }
            }
            ip += 1;
        }
        Ok(())
    }

    fn handle_shared_fun_return(
        &mut self,
        protocol: &mut C,
        current_shared_ret_vals: &[(C::VmType, Vec<C::VmType>)],
    ) -> Result<()> {
        let mut acc = vec![protocol.public_zero(); self.current_return_vals];
        for (cond, maybe_ret_vals) in current_shared_ret_vals {
            for (acc, x) in izip!(
                acc.iter_mut(),
                maybe_ret_vals
                    .iter()
                    .map(|v| protocol.mul(cond.clone(), v.to_owned()))
                    //we need to collect here because borrow checker is unhappy otherwise
                    //we could remove the mut borrow for a lot of MPC operations I think
                    //maybe we should do that???
                    .collect::<Result<Vec<_>, _>>()?
            ) {
                *acc = protocol.add(acc.to_owned(), x)?;
            }
        }
        for shared_ret_val in acc {
            self.push_field(shared_ret_val);
        }
        Ok(())
    }
}

impl<F: PrimeField, C: VmCircomWitnessExtension<F>> WitnessExtension<F, C> {
    fn post_processing(
        &mut self,
        amount_public_inputs: usize,
    ) -> Result<FinalizedWitnessExtension<F, C>> {
        let total_public_amount = self.main_outputs + amount_public_inputs + 1;
        let mut public_inputs = Vec::with_capacity(total_public_amount);
        let mut witness = Vec::with_capacity(self.signal_to_witness.len() - total_public_amount);
        for (count, idx) in self.signal_to_witness.iter().enumerate() {
            // the +1 here is for the constant 1 which always is at position 0.
            if count < total_public_amount {
                public_inputs.push(self.driver.open(self.ctx.signals[*idx].clone())?);
            } else {
                witness.push(self.driver.to_share(self.ctx.signals[*idx].clone())?);
            }
        }
        Ok(FinalizedWitnessExtension {
            shared_witness: SharedWitness {
                public_inputs,
                witness,
            },
            // TODO take instead of clone? or consume self again and close network in here?
            output_mapping: self.output_mapping.clone(),
        })
    }

    fn set_input_signals(
        &mut self,
        mut input_signals: SharedInput<F, C::ArithmeticShare>,
    ) -> Result<usize> {
        let mut amount_public_inputs = 0;
        for (name, offset, size) in self.main_input_list.iter() {
            let input_signals =
                if let Some(public_values) = input_signals.public_inputs.remove(name) {
                    amount_public_inputs += public_values.len();
                    public_values.into_iter().map(C::VmType::from).collect_vec()
                } else {
                    input_signals
                        .shared_inputs
                        .remove(name)
                        .ok_or(eyre!("Cannot find signal \"{name}\" in provided input"))?
                        .into_iter()
                        .map(C::VmType::from)
                        .collect_vec()
                };
            if input_signals.len() != *size {
                bail!(
                    "for input \"{name}\" expected {size} signals, got {}",
                    input_signals.len()
                );
            }
            self.ctx.signals[*offset..*offset + *size].clone_from_slice(input_signals.as_slice());
        }
        Ok(amount_public_inputs)
    }

    fn set_flat_input_signals(&mut self, input_signals: Vec<C::VmType>) {
        assert_eq!(
            self.main_inputs,
            input_signals.len(),
            "You have to provide the input signals"
        );
        self.ctx.signals[1 + self.main_outputs..1 + self.main_outputs + self.main_inputs]
            .clone_from_slice(&input_signals);
    }

    fn call_main_component(&mut self) -> Result<()> {
        let main_templ = self
            .ctx
            .templ_decls
            .get(&self.main)
            .ok_or(eyre!("cannot find main template: {}", self.main))?;
        let mut main_component = Component::init(main_templ, 1);
        main_component.run(&mut self.driver, &mut self.ctx, &self.config)?;
        Ok(())
    }

    /// Starts the execution of the MPC-VM with the provided [SharedInput] and consumes `self`.
    ///
    /// Use this method over [`run_with_flat()`](WitnessExtension::run) when ever possible.
    /// # Arguments
    ///
    /// * `input_signals` - The [SharedInput] distributed over the parties.
    ///
    /// # Returns
    ///
    /// * `Ok([SharedWitness])` - The secret-shared witness, distributed over the parties.
    /// * `Err([eyre::Result])` - An error result.
    ///
    /// # Panics
    ///
    /// Panics if any of the [`CodeBlocks`](CodeBlock) are corrupted.
    pub fn run(
        &mut self,
        input_signals: SharedInput<F, C::ArithmeticShare>,
    ) -> Result<FinalizedWitnessExtension<F, C>> {
        let amount_public_inputs = self.set_input_signals(input_signals)?;
        self.call_main_component()?;
        self.post_processing(amount_public_inputs)
    }

    /// Starts the execution of the MPC-VM with the provided input signals and consumes `self`.
    ///
    /// > **Warning:** The input signals are copied as provided, element by element, into the internal signals `Vec`.
    /// > This means there is no mapping done between the names given in the circom file and the elements.
    /// > Use this only if you are absolutely sure which signal corresponds to which index.
    ///
    /// Use [`run()`](WitnessExtension::run) whenever possible.
    ///
    /// # Arguments
    ///
    /// * `input_signals` - A `Vec` of [`CircomWitnessExtensionProtocol::VmType`].
    /// * `amount_public_inputs` - The amount of public inputs, necessary for later building the [`SharedWitness`].
    ///
    /// # Returns
    ///
    /// * `Ok([FinalizedWitnessExtension])` - The secret-shared witness, distributed over the parties.
    /// * `Err([eyre::Result])` - An error result.
    ///
    /// # Panics
    ///
    /// Panics if any of the [`CodeBlocks`](CodeBlock) are corrupted.
    pub fn run_with_flat(
        &mut self,
        input_signals: Vec<C::VmType>,
        amount_public_inputs: usize,
    ) -> Result<FinalizedWitnessExtension<F, C>> {
        self.set_flat_input_signals(input_signals);
        self.call_main_component()?;
        self.post_processing(amount_public_inputs)
    }
}

/// The result of the witness extension, storing the secret-shared witness.
///
/// This struct is a wrapper around the [`SharedWitness`] and an [`OutputMapping`]. It is useful
/// for retrieving one of the outputs of the main component of the circom file.
///
/// If you want to retrieve the shared witness, call [`into_shared_witness()`](FinalizedWitnessExtension::into_shared_witness()).
pub struct FinalizedWitnessExtension<F: PrimeField, C: VmCircomWitnessExtension<F>> {
    shared_witness: SharedWitness<F, C::ArithmeticShare>,
    output_mapping: OutputMapping,
}

impl<F: PrimeField, C: VmCircomWitnessExtension<F>> From<FinalizedWitnessExtension<F, C>>
    for SharedWitness<F, C::ArithmeticShare>
{
    fn from(value: FinalizedWitnessExtension<F, C>) -> Self {
        value.shared_witness
    }
}

impl<F: PrimeField, C: VmCircomWitnessExtension<F>> FinalizedWitnessExtension<F, C> {
    /// Consumes self and returns the [`SharedWitness`].
    pub fn into_shared_witness(self) -> SharedWitness<F, C::ArithmeticShare> {
        self.shared_witness
    }

    /// Returns the signals associated with the provided name in the circom file.
    ///
    /// # Example
    ///
    /// Imagine the following code snippet:
    /// ```c++
    /// template Main() {
    ///     signal input a;
    ///     signal input b;
    ///     signal output c;
    ///     c <== a + b;
    /// }
    /// component main = Main();
    /// ```
    ///
    /// Then you can retrieve the output `c` by calling this method.
    ///
    /// # Arguments
    /// - `name`: The name of the signal to retrieve.
    ///
    /// # Returns
    /// Returns an `Option<Vec<F>>` containing the signals associated with the requested output.
    /// Returns `None` if the name is not known.
    pub fn get_output(&self, name: &str) -> Option<Vec<F>> {
        self.output_mapping.get(name).map(|(offset, amount)| {
            self.shared_witness.public_inputs[*offset..*offset + *amount].to_vec()
        })
    }
}

impl<F: PrimeField> PlainWitnessExtension<F> {
    pub(crate) fn new(parser: CoCircomCompilerParsed<F>, config: VMConfig) -> Self {
        let mut signals = vec![F::default(); parser.amount_signals];
        signals[0] = F::one();
        Self {
            driver: CircomPlainVmWitnessExtension::default(),
            signal_to_witness: parser.signal_to_witness,
            main: parser.main,
            ctx: WitnessExtensionCtx::new(
                signals,
                parser.constant_table,
                parser.fun_decls,
                parser.templ_decls,
                parser.string_table,
                MpcAccelerator::full_mpc_accelerator(),
            ),
            main_inputs: parser.main_inputs,
            main_outputs: parser.main_outputs,
            main_input_list: parser.main_input_list,
            output_mapping: parser.output_mapping,
            config,
        }
    }
}

impl<F: PrimeField, N: Rep3Network> Rep3WitnessExtension<F, N> {
    pub(crate) fn from_network(
        parser: CoCircomCompilerParsed<F>,
        network: N,
        mpc_accelerator: MpcAccelerator<F, CircomRep3VmWitnessExtension<F, N>>,
        config: VMConfig,
    ) -> Result<Self> {
        let driver = CircomRep3VmWitnessExtension::from_network(network)?;
        let mut signals = vec![Rep3VmType::default(); parser.amount_signals];
        signals[0] = Rep3VmType::Public(F::one());
        let constant_table = parser
            .constant_table
            .into_iter()
            .map(Rep3VmType::Public)
            .collect_vec();
        Ok(Self {
            driver,
            signal_to_witness: parser.signal_to_witness,
            main: parser.main,
            ctx: WitnessExtensionCtx::new(
                signals,
                constant_table,
                parser.fun_decls,
                parser.templ_decls,
                parser.string_table,
                mpc_accelerator,
            ),
            main_inputs: parser.main_inputs,
            main_outputs: parser.main_outputs,
            main_input_list: parser.main_input_list,
            output_mapping: parser.output_mapping,
            config,
        })
    }
}

impl<F: PrimeField> Rep3WitnessExtension<F, Rep3MpcNet> {
    pub(crate) fn new(
        parser: CoCircomCompilerParsed<F>,
        network_config: NetworkConfig,
        mpc_accelerator: MpcAccelerator<F, CircomRep3VmWitnessExtension<F, Rep3MpcNet>>,
        config: VMConfig,
    ) -> Result<Self> {
        let network = futures::executor::block_on(Rep3MpcNet::new(network_config))?;
        Self::from_network(parser, network, mpc_accelerator, config)
    }

    /// Gracefully shutdown the netowork. Waits until all data is sent and received
    pub fn close_network(self) -> Result<()> {
        self.driver.close_network()
    }
}
