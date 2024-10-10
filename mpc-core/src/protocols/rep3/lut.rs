//! Lookup Table
//!
//! This module contains implementation of a LUT

use ark_ff::PrimeField;
use itertools::izip;

use crate::{
    lut::LookupTableProvider,
    protocols::rep3::{arithmetic, binary, conversion},
};

use super::{
    network::{IoContext, Rep3Network},
    IoResult, Rep3PrimeFieldShare,
};

type MpcMap<F> = Vec<(F, F)>;

/// Rep3 lookup table
pub struct NaiveRep3LookupTable<N: Rep3Network> {
    io_context: IoContext<N>,
}

impl<N: Rep3Network> NaiveRep3LookupTable<N> {
    /// Construct a new [`NaiveRep3LookupTable`]
    pub fn new(io_context: IoContext<N>) -> Self {
        Self { io_context }
    }
}

impl<F: PrimeField, N: Rep3Network> LookupTableProvider<F> for NaiveRep3LookupTable<N> {
    type SecretShare = Rep3PrimeFieldShare<F>;
    type SecretSharedSet = Vec<Rep3PrimeFieldShare<F>>;

    type SecretSharedMap = MpcMap<Rep3PrimeFieldShare<F>>;

    // Maybe give every set a dedicated ID/String to re-identify in debugging
    fn init_set(
        &self,
        values: impl IntoIterator<Item = Self::SecretShare>,
    ) -> Self::SecretSharedSet {
        tracing::debug!("initiating LUT-set");
        values.into_iter().collect()
    }

    async fn contains_set(
        &mut self,
        needle: &Self::SecretShare,
        set: &Self::SecretSharedSet,
    ) -> IoResult<Self::SecretShare> {
        tracing::debug!("checking if value is in set of size {}", set.len());
        // first get a vector of true/false

        // if we can easily fork the io context we can do it like that. Even better would be a batched
        // version
        //       let equals_vec = set
        //           .iter()
        //           .map(|ele| arithmetic::equals_bit(*needle, *ele, &mut self.io_context))
        //           .collect::<FuturesOrdered<_>>()
        //           .collect::<Vec<_>>()
        //           .await
        //           .into_iter()
        //           .collect::<IoResult<Vec<_>>>()?;
        let mut equals_vec = Vec::with_capacity(set.len());
        for ele in set.iter() {
            let bit = arithmetic::eq_bit(*needle, *ele, &mut self.io_context).await?;
            equals_vec.push(bit);
        }

        tracing::debug!("got binary equals vec now or tree..");
        //or tree to get result
        let binary_result = binary::or_tree(equals_vec, &mut self.io_context).await?;
        tracing::debug!("one last conversion from binary to arithmetic...");
        conversion::b2a(&binary_result, &mut self.io_context).await
    }

    fn init_map(
        &self,
        values: impl IntoIterator<Item = (Self::SecretShare, Self::SecretShare)>,
    ) -> Self::SecretSharedMap {
        tracing::debug!("initiating LUT-map");
        values.into_iter().collect()
    }

    async fn get_from_lut(
        &mut self,
        needle: Self::SecretShare,
        map: &Self::SecretSharedMap,
    ) -> IoResult<Self::SecretShare> {
        // make some experiments which is faster
        // a single for each or multiple chained iterators...
        //
        //first iterate over keys and perform equality check
        tracing::debug!("doing read on LUT-map of size {}", map.len());
        tracing::debug!("get random zeros for blinding..");
        let mut zeros_a = Vec::with_capacity(map.len());
        zeros_a.resize_with(map.len(), || {
            self.io_context.rngs.rand.masking_field_element::<F>()
        });
        let zeros_b = self.io_context.network.reshare_many(&zeros_a).await?;
        tracing::debug!("now perform equals and cmux...");
        let mut result = Self::SecretShare::default();
        for ((key, map), zero_a, zero_b) in
            izip!(map.iter(), zeros_a.into_iter(), zeros_b.into_iter())
        {
            // this is super slow - we can batch it?
            let zero_share = Self::SecretShare::new(zero_a, zero_b);
            let equals = arithmetic::eq(needle, *key, &mut self.io_context).await?;
            let cmux = arithmetic::cmux(equals, *map, zero_share, &mut self.io_context).await?;
            result = arithmetic::add(result, cmux);
        }
        tracing::debug!("got a result!");
        Ok(result)
    }

    async fn write_to_lut(
        &mut self,
        needle: Self::SecretShare,
        value: Self::SecretShare,
        map: &mut Self::SecretSharedMap,
    ) -> IoResult<()> {
        tracing::debug!("doing write on LUT-map of size {}", map.len());
        // we do not need any zeros here
        for (key, map) in map.iter_mut() {
            // this is super slow - we can batch it?
            let equals = arithmetic::eq(needle, *key, &mut self.io_context).await?;
            let cmux = arithmetic::cmux(equals, value, *map, &mut self.io_context).await?;
            *map = cmux;
        }
        tracing::debug!("we are done");
        Ok(())
    }
}
