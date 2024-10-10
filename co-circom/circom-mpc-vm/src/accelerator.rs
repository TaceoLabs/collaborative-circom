use std::collections::HashMap;

use ark_ff::PrimeField;
use eyre::bail;

use crate::mpc::VmCircomWitnessExtension;

type AcceleratorFunction<F, C> = Box<
    dyn Fn(
            &mut C,
            &[<C as VmCircomWitnessExtension<F>>::VmType],
        ) -> eyre::Result<Vec<<C as VmCircomWitnessExtension<F>>::VmType>>
        + Send,
>;

#[derive(Default)]
pub struct MpcAccelerator<F: PrimeField, C: VmCircomWitnessExtension<F>> {
    registered_functions: HashMap<String, AcceleratorFunction<F, C>>,
}

impl<F: PrimeField, C: VmCircomWitnessExtension<F>> MpcAccelerator<F, C> {
    pub fn empty_accelerator() -> Self {
        Self {
            registered_functions: HashMap::default(),
        }
    }

    pub fn full_mpc_accelerator() -> Self {
        let mut accelerator = Self::empty_accelerator();
        accelerator.register_sqrt();
        accelerator
    }

    pub fn register_function(
        &mut self,
        name: String,
        fun: impl Fn(&mut C, &[C::VmType]) -> eyre::Result<Vec<C::VmType>> + Send + 'static,
    ) {
        self.registered_functions.insert(name, Box::new(fun));
    }

    pub(crate) fn has_accelerator(&self, name: &str) -> bool {
        self.registered_functions.contains_key(name)
    }

    fn register_sqrt(&mut self) {
        self.register_function("sqrt_0".to_owned(), |protocol, args| {
            tracing::debug!("calling pre-defined sqrt accelerator");
            if args.len() != 1 {
                bail!("Calling SQRT accelerator with more than one argument!");
            }
            Ok(vec![protocol.sqrt(args[0].to_owned())?])
        });
    }

    pub(crate) fn run_accelerator(
        &self,
        name: &str,
        protocol: &mut C,
        args: &[C::VmType],
    ) -> eyre::Result<Vec<C::VmType>> {
        let fun = self
            .registered_functions
            .get(name)
            .ok_or(eyre::eyre!("cannot find accelerator {name}"))?;
        fun(protocol, args)
    }
}
