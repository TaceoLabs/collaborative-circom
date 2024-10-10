pub(crate) mod builder_variable;
pub(crate) mod proving_key;
pub(crate) mod types;

use ark_ec::pairing::Pairing;
use builder_variable::SharedBuilderVariable;
use ultrahonk::prelude::GenericUltraCircuitBuilder;

use crate::{
    mpc::{plain::PlainUltraHonkDriver, rep3::Rep3UltraHonkDriver},
    prelude::ShamirUltraHonkDriver,
};

pub type CoUltraCircuitBuilder<T, P> = GenericUltraCircuitBuilder<P, SharedBuilderVariable<T, P>>;

pub type PlainCoBuilder<P> = CoUltraCircuitBuilder<PlainUltraHonkDriver, P>;
pub type Rep3CoBuilder<P, N> = CoUltraCircuitBuilder<Rep3UltraHonkDriver<N>, P>;
pub type ShamirCoBuilder<P, N> =
    CoUltraCircuitBuilder<ShamirUltraHonkDriver<<P as Pairing>::ScalarField, N>, P>;
