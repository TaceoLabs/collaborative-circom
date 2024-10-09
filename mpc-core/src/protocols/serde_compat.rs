use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};

// TODO deduplicate

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
