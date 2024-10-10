//! Lookup table provider
//!
//! This module contains the abstraction to lookup tables

use std::{
    collections::{HashMap, HashSet},
    io,
    marker::PhantomData,
};

use ark_ff::PrimeField;

#[allow(async_fn_in_trait)]
/// This is some place holder definition. This will change most likely
pub trait LookupTableProvider<F: PrimeField> {
    /// The type used in LUTs
    type SecretShare;
    /// A LUT for performing membership checks (like `HashSet`). Mostly used for range checks.
    type SecretSharedSet;
    /// An input/output LUT (like `HashMap`).
    type SecretSharedMap;

    /// Initializes a set for membership checks from the provided values.
    fn init_set(
        &self,
        values: impl IntoIterator<Item = Self::SecretShare>,
    ) -> Self::SecretSharedSet;

    /// Checks whether the needle is a member of the provided set.
    ///
    /// # Returns
    /// Returns a secret-shared value. If the reconstructed value is 1, the set
    /// contained the element. Otherwise, shall return secret-shared 0.
    ///
    /// Can fail due to networking problems.
    ///
    async fn contains_set(
        &mut self,
        needle: &Self::SecretShare,
        set: &Self::SecretSharedSet,
    ) -> io::Result<Self::SecretShare>;

    /// Initializes a map (input/output LUT) from the provided values. The keys and values are
    /// matched from their order of the iterator.
    fn init_map(
        &self,
        values: impl IntoIterator<Item = (Self::SecretShare, Self::SecretShare)>,
    ) -> Self::SecretSharedMap;

    /// Reads a value from the map associated with the provided needle. As we work over secret-shared
    /// values we can not check whether the needle is actually in the set. The caller must ensure that
    /// the key is in the map.
    ///
    /// # Returns
    /// The secret-shared value associated with the needle. A not known needle results in undefined
    /// behaviour.
    ///
    /// Can fail due to networking problems.
    ///
    async fn get_from_lut(
        &mut self,
        key: Self::SecretShare,
        map: &Self::SecretSharedMap,
    ) -> io::Result<Self::SecretShare>;

    /// Writes a value to the map.
    ///
    /// **IMPORTANT**: the implementation will NOT add
    /// the key-value pair to the map, if it is not already registered! The implementation
    /// overwrites an existing key, but a not-known key will be ignored.
    ///
    /// #Returns
    /// Can fail due to networking problems.
    async fn write_to_lut(
        &mut self,
        index: Self::SecretShare,
        value: Self::SecretShare,
        lut: &mut Self::SecretSharedMap,
    ) -> io::Result<()>;
}

/// LUT provider for public values
#[derive(Default)]
pub struct PlainLookupTableProvider<F: PrimeField> {
    phantom_data: PhantomData<F>,
}

impl<F: PrimeField> LookupTableProvider<F> for PlainLookupTableProvider<F> {
    type SecretShare = F;
    // we could check if a Vec<F> impl may be faster. Depends on the size of the LUT..
    type SecretSharedSet = HashSet<F>;

    type SecretSharedMap = HashMap<F, F>;

    fn init_set(
        &self,
        values: impl IntoIterator<Item = Self::SecretShare>,
    ) -> Self::SecretSharedSet {
        values.into_iter().collect::<HashSet<_>>()
    }

    async fn contains_set(
        &mut self,
        value: &Self::SecretShare,
        set: &Self::SecretSharedSet,
    ) -> io::Result<F> {
        if set.contains(value) {
            Ok(F::one())
        } else {
            Ok(F::zero())
        }
    }

    fn init_map(
        &self,
        values: impl IntoIterator<Item = (Self::SecretShare, Self::SecretShare)>,
    ) -> Self::SecretSharedMap {
        values.into_iter().collect::<HashMap<_, _>>()
    }

    async fn get_from_lut(&mut self, key: F, map: &Self::SecretSharedMap) -> io::Result<F> {
        Ok(map[&key])
    }

    async fn write_to_lut(
        &mut self,
        key: F,
        value: F,
        map: &mut Self::SecretSharedMap,
    ) -> io::Result<()> {
        if map.insert(key, value).is_none() {
            panic!("we cannot add new keys to the lookup table!")
        }
        Ok(())
    }
}
