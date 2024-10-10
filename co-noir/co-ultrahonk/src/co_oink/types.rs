use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ultrahonk::prelude::Polynomial;

use crate::{mpc::NoirUltraHonkProver, NUM_ALPHAS};

pub(crate) struct ProverMemory<T: NoirUltraHonkProver<P>, P: Pairing> {
    pub(crate) w_4: Polynomial<T::ArithmeticShare>, // column 3
    pub(crate) z_perm: Polynomial<T::ArithmeticShare>, // column 4
    pub(crate) lookup_inverses: Polynomial<T::ArithmeticShare>, // column 5
    pub(crate) public_input_delta: P::ScalarField,
    pub(crate) challenges: Challenges<P::ScalarField>,
}

pub(crate) struct Challenges<F: PrimeField> {
    pub(crate) eta_1: F,
    pub(crate) eta_2: F,
    pub(crate) eta_3: F,
    pub(crate) beta: F,
    pub(crate) gamma: F,
    pub(crate) alphas: [F; NUM_ALPHAS],
}

impl<F: PrimeField> Default for Challenges<F> {
    fn default() -> Self {
        Self {
            eta_1: Default::default(),
            eta_2: Default::default(),
            eta_3: Default::default(),
            beta: Default::default(),
            gamma: Default::default(),
            alphas: [Default::default(); NUM_ALPHAS],
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> Default for ProverMemory<T, P> {
    fn default() -> Self {
        Self {
            w_4: Default::default(),
            z_perm: Default::default(),
            lookup_inverses: Default::default(),
            public_input_delta: Default::default(),
            challenges: Default::default(),
        }
    }
}
