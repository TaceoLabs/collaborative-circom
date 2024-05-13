use ark_ff::PrimeField;
use std::marker::PhantomData;

pub mod fieldshare;
pub mod pointshare;

pub struct GSZProtocol<F> {
    field: PhantomData<F>,
}

impl<F: PrimeField> GSZProtocol<F> {}
