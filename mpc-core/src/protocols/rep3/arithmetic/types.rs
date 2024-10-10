use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use serde::{Deserialize, Serialize};

use crate::protocols::rep3::{
    id::PartyID,
    network::{IoContext, Rep3Network},
};

/// This type represents a replicated shared value. Since a replicated share of a field element contains additive shares of two parties, this type contains two field elements.
#[derive(
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    CanonicalSerialize,
    CanonicalDeserialize,
    Serialize,
    Deserialize,
)]
pub struct Rep3PrimeFieldShare<F: PrimeField> {
    /// Share of this party
    #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
    pub a: F,
    /// Share of the prev party
    #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
    pub b: F,
}

/// Serialize an object with ark serialization, to be used with serde.
/// `#[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]`
pub(crate) fn ark_se<S, A: CanonicalSerialize>(a: &A, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut bytes = vec![];
    a.serialize_with_mode(&mut bytes, Compress::Yes)
        .map_err(serde::ser::Error::custom)?;
    s.serialize_bytes(&bytes)
}

/// Deserialize an object with ark deserialization, to be used with serde.
/// `#[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]`
pub(crate) fn ark_de<'de, D, A: CanonicalDeserialize>(data: D) -> Result<A, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    let s: Vec<u8> = serde::de::Deserialize::deserialize(data)?;
    let a = A::deserialize_with_mode(s.as_slice(), Compress::Yes, Validate::Yes);
    a.map_err(serde::de::Error::custom)
}

impl<F: PrimeField> Default for Rep3PrimeFieldShare<F> {
    fn default() -> Self {
        Self::zero_share()
    }
}

impl<F: PrimeField> Rep3PrimeFieldShare<F> {
    /// Constructs the type from two additive shares.
    pub fn new(a: F, b: F) -> Self {
        Self { a, b }
    }

    /// Constructs a zero share.
    pub fn zero_share() -> Self {
        Self {
            a: F::zero(),
            b: F::zero(),
        }
    }

    /// Unwraps the type into two additive shares.
    pub fn ab(self) -> (F, F) {
        (self.a, self.b)
    }

    /// Double the share in place
    pub fn double(&mut self) {
        self.a.double_in_place();
        self.b.double_in_place();
    }

    /// Generate a random share
    pub fn rand<N: Rep3Network>(io_context: &mut IoContext<N>) -> Self {
        let (a, b) = io_context.rngs.rand.random_fes();
        Self::new(a, b)
    }

    /// Promotes a public field element to a replicated share by setting the additive share of the party with id=0 and leaving all other shares to be 0. Thus, the replicated shares of party 0 and party 1 are set.
    pub fn promote_from_trivial(val: &F, id: PartyID) -> Self {
        match id {
            PartyID::ID0 => Self::new(*val, F::zero()),
            PartyID::ID1 => Self::new(F::zero(), *val),
            PartyID::ID2 => Self::zero_share(),
        }
    }
}
