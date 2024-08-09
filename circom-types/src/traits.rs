//! This module contains traits for serializing and deserializing field elements and curve points into and from circom files to arkworks representation.
use std::io::Read;
use std::marker::PhantomData;

use ark_ec::{pairing::Pairing, AffineRepr};
use ark_ff::{PrimeField, Zero};
use ark_serialize::SerializationError;
use serde::ser::SerializeSeq;
use serde::{de, Serializer};
use std::str::FromStr;

type IoResult<T> = Result<T, SerializationError>;

macro_rules! impl_bn256 {
    () => {
        impl_serde_for_curve!(bn254, Bn254, ark_bn254, 32, 32, "bn128");
    };
}

macro_rules! impl_bls12_381 {
    () => {
        impl_serde_for_curve!(bls12_381, Bls12_381, ark_bls12_381, 48, 32, "bls12_381");
    };
}

#[cfg(not(feature = "dangerous"))]
macro_rules! point_check {
    ($p: expr) => {{
        if !$p.is_on_curve() {
            Err(SerializationError::InvalidData)
        } else if !$p.is_in_correct_subgroup_assuming_on_curve() {
            Err(SerializationError::InvalidData)
        } else {
            Ok($p)
        }
    }};
}

macro_rules! impl_serde_for_curve {
    ($mod_name: ident, $config: ident, $curve: ident, $field_size: expr, $scalar_field_size: expr, $circom_name: expr) => {

mod $mod_name {

    use $curve::{$config, Fq, Fq2, Fr};
    use ark_ff::BigInt;
    use ark_serialize::{CanonicalDeserialize, SerializationError};
    use serde::ser::SerializeSeq;

    use super::*;

    #[cfg(feature = "dangerous")]
    #[inline(always)]
    fn test_on_curve_g1(p: $curve::G1Affine) -> Result<$curve::G1Affine, SerializationError> {
        Ok(p)
    }

    #[cfg(feature = "dangerous")]
    #[inline(always)]
    fn test_on_curve_g2(p: $curve::G2Affine) -> Result<$curve::G2Affine, SerializationError> {
        Ok(p)
    }

    #[cfg(not(feature = "dangerous"))]
    #[inline(always)]
    fn test_on_curve_g1(p: $curve::G1Affine) -> Result<$curve::G1Affine, SerializationError> {
        point_check!(p)
    }


    #[cfg(not(feature = "dangerous"))]
    #[inline(always)]
    fn test_on_curve_g2(p: $curve::G2Affine) -> Result<$curve::G2Affine, SerializationError> {
        point_check!(p)
    }

        impl CircomArkworksPrimeFieldBridge for Fr {
            const SERIALIZED_BYTE_SIZE: usize = $scalar_field_size;
            #[inline]
            fn from_reader(mut reader: impl Read) -> IoResult<Self> {
                let mut buf = [0u8; Self::SERIALIZED_BYTE_SIZE];
                reader.read_exact(&mut buf[..])?;
                Ok(Self::from_le_bytes_mod_order(&buf))
            }

            #[inline]
            fn from_reader_unchecked(mut reader: impl Read) -> IoResult<Self> {
                let mut buf = [0u8; Self::SERIALIZED_BYTE_SIZE];
                reader.read_exact(&mut buf[..])?;
                Ok(Self::new_unchecked(BigInt::deserialize_uncompressed(
                    buf.as_slice(),
                )?))
            }
            #[inline]
            fn from_reader_unchecked_for_zkey(reader: impl Read) -> IoResult<Self> {
                Ok(Self::new_unchecked(Self::from_reader_unchecked(reader)?.into_bigint()))
            }

        }
        impl CircomArkworksPrimeFieldBridge for Fq {
            const SERIALIZED_BYTE_SIZE: usize = $field_size;
            #[inline]
            fn from_reader(mut reader: impl Read) -> IoResult<Self> {
                let mut buf = [0u8; Self::SERIALIZED_BYTE_SIZE];
                reader.read_exact(&mut buf[..])?;
                Ok(Self::from_le_bytes_mod_order(&buf))
            }

            #[inline]
            fn from_reader_unchecked(mut reader: impl Read) -> IoResult<Self> {
                let mut buf = [0u8; Self::SERIALIZED_BYTE_SIZE];
                reader.read_exact(&mut buf[..])?;
                Ok(Self::new_unchecked(BigInt::deserialize_uncompressed(
                    buf.as_slice(),
                )?))
            }
            #[inline]
            fn from_reader_unchecked_for_zkey(reader: impl Read) -> IoResult<Self> {
                Ok(Self::new_unchecked(Self::from_reader_unchecked(reader)?.into_bigint()))
            }
        }

        impl CircomArkworksPairingBridge for $config {
            const G1_SERIALIZED_BYTE_SIZE_COMPRESSED: usize = $field_size;
            const G1_SERIALIZED_BYTE_SIZE_UNCOMPRESSED: usize = $field_size * 2;
            const G2_SERIALIZED_BYTE_SIZE_COMPRESSED: usize = $field_size * 2;
            const G2_SERIALIZED_BYTE_SIZE_UNCOMPRESSED: usize = $field_size * 2 * 2;
            const GT_SERIALIZED_BYTE_SIZE_COMPRESSED: usize = 0;
            const GT_SERIALIZED_BYTE_SIZE_UNCOMPRESSED: usize = 0;

            fn get_circom_name() -> String {
                $circom_name.to_owned()
            }

            //Circom serializes its field elements in montgomery form
            //therefore we use Fq::from_reader_unchecked
            fn g1_from_reader(mut reader: impl Read) -> IoResult<Self::G1Affine> {
                let mut buf = [0u8; Self::G1_SERIALIZED_BYTE_SIZE_UNCOMPRESSED];
                reader.read_exact(&mut buf)?;
                //already in montgomery form
                let x = Fq::from_reader_unchecked(&buf[..Fq::SERIALIZED_BYTE_SIZE])?;
                let y = Fq::from_reader_unchecked(&buf[Fq::SERIALIZED_BYTE_SIZE..])?;

                if x.is_zero() && y.is_zero() {
                    return Ok(Self::G1Affine::zero());
                }
                test_on_curve_g1(Self::G1Affine::new_unchecked(x, y))
            }

            fn g2_from_reader(mut reader: impl Read) -> IoResult<Self::G2Affine> {
                let mut buf = [0u8; Self::G2_SERIALIZED_BYTE_SIZE_UNCOMPRESSED];
                reader.read_exact(&mut buf)?;
                //already in montgomery form
                let x0 = Fq::from_reader_unchecked(&buf[..Fq::SERIALIZED_BYTE_SIZE])?;
                let x1 = Fq::from_reader_unchecked(
                    &buf[Fq::SERIALIZED_BYTE_SIZE..Fq::SERIALIZED_BYTE_SIZE * 2],
                )?;
                let y0 = Fq::from_reader_unchecked(
                    &buf[Fq::SERIALIZED_BYTE_SIZE * 2..Fq::SERIALIZED_BYTE_SIZE * 3],
                )?;
                let y1 = Fq::from_reader_unchecked(
                    &buf[Fq::SERIALIZED_BYTE_SIZE * 3..Fq::SERIALIZED_BYTE_SIZE * 4],
                )?;

                let x = Fq2::new(x0, x1);
                let y = Fq2::new(y0, y1);

                if x.is_zero() && y.is_zero() {
                    return Ok(Self::G2Affine::zero());
                }
                test_on_curve_g2(Self::G2Affine::new_unchecked(x, y))
            }

            fn g1_from_strings_projective(x: &str, y: &str, z: &str) -> IoResult<Self::G1Affine> {
                let x = parse_field(x)?;
                let y = parse_field(y)?;
                let z = parse_field(z)?;
                let p = Self::G1Affine::from($curve::G1Projective::new(x, y, z));
                if p.is_zero() {
                    return Ok(p);
                }
                test_on_curve_g1(p)
            }

            fn g1_to_strings_projective(p: &Self::G1Affine) -> Vec<String> {
                if let Some((x, y)) = p.xy() {
                    vec![x.to_string(), y.to_string(), "1".to_owned()]
                } else {
                    //point at infinity
                    vec!["0".to_owned(), "1".to_owned(), "0".to_owned()]
                }
            }

            fn g2_from_strings_projective(
                x0: &str,
                x1: &str,
                y0: &str,
                y1: &str,
                z0: &str,
                z1: &str,
            ) -> IoResult<Self::G2Affine> {
                let x0 = parse_field(x0)?;
                let x1 = parse_field(x1)?;
                let y0 = parse_field(y0)?;
                let y1 = parse_field(y1)?;
                let z0 = parse_field(z0)?;
                let z1 = parse_field(z1)?;

                let x = $curve::Fq2::new(x0, x1);
                let y = $curve::Fq2::new(y0, y1);
                let z = $curve::Fq2::new(z0, z1);
                let p = $curve::G2Affine::from($curve::G2Projective::new(x, y, z));
                if p.is_zero() {
                    return Ok(p);
                }
                test_on_curve_g2(p)
            }

            fn serialize_g2<S: Serializer>(p: &Self::G2Affine, ser: S) -> Result<S::Ok, S::Error> {
                let (x, y) = p.xy().unwrap();
                let mut x_seq = ser.serialize_seq(Some(3))?;
                x_seq.serialize_element(&vec![x.c0.to_string(), x.c1.to_string()])?;
                x_seq.serialize_element(&vec![y.c0.to_string(), y.c1.to_string()])?;
                x_seq.serialize_element(&vec!["1", "0"])?;
                x_seq.end()
            }
            fn serialize_gt<S: Serializer>(
                p: &Self::TargetField,
                ser: S,
            ) -> Result<S::Ok, S::Error> {
                let a = p.c0;
                let b = p.c1;
                let aa = a.c0;
                let ab = a.c1;
                let ac = a.c2;
                let ba = b.c0;
                let bb = b.c1;
                let bc = b.c2;
                let a = vec![
                    vec![aa.c0.to_string(), aa.c1.to_string()],
                    vec![ab.c0.to_string(), ab.c1.to_string()],
                    vec![ac.c0.to_string(), ac.c1.to_string()],
                ];
                let b = vec![
                    vec![ba.c0.to_string(), ba.c1.to_string()],
                    vec![bb.c0.to_string(), bb.c1.to_string()],
                    vec![bc.c0.to_string(), bc.c1.to_string()],
                ];
                let mut seq = ser.serialize_seq(Some(2))?;
                seq.serialize_element(&a)?;
                seq.serialize_element(&b)?;
                seq.end()
            }
            fn serialize_fr<S: Serializer>(p: &Self::ScalarField, ser: S) -> Result<S::Ok, S::Error> {
                ser.serialize_str(&p.to_string())
                }

            fn deserialize_gt_element<'de, D>(
                deserializer: D,
            ) -> Result<Self::TargetField, D::Error>
            where
                D: de::Deserializer<'de>,
            {
                deserializer.deserialize_seq(TargetGroupVisitor::<Self>::new())
            }

        }

    impl<'de> de::Visitor<'de> for TargetGroupVisitor<$config> {
        type Value = $curve::Fq12;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str(
                &format!("An element of {}::Fq12 represented as string with radix 10. Must be a sequence of form [[[String; 2]; 3]; 2].", stringify!($mod_name)),
            )
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let x = seq
                .next_element::<Vec<Vec<String>>>()?
                .ok_or(de::Error::custom(
                    &format!("expected elements target group in {} as sequence of sequences", stringify!($mod_name)),
                ))?;
            let y = seq
                .next_element::<Vec<Vec<String>>>()?
                .ok_or(de::Error::custom(
                    &format!("expected elements target group in {} as sequence of sequences", stringify!($mod_name)),
                ))?;
            if x.len() != 3 || y.len() != 3 {
                Err(de::Error::custom(
                    &format!("need three elements for cubic extension field in {}", stringify!($mod_name)),
                ))
            } else {
                let c0 = cubic_extension_field_from_vec(x).map_err(|_| {
                    de::Error::custom("InvalidData for target group (cubic extension field)")
                })?;
                let c1 = cubic_extension_field_from_vec(y).map_err(|_| {
                    de::Error::custom("InvalidData for target group (cubic extension field)")
                })?;
                Ok($curve::Fq12::new(c0, c1))
            }
        }
    }
    #[inline]
    fn cubic_extension_field_from_vec(strings: Vec<Vec<String>>) -> IoResult<$curve::Fq6> {
        if strings.len() != 3 {
            Err(SerializationError::InvalidData)
        } else {
            let c0 = quadratic_extension_field_from_vec(&strings[0])?;
            let c1 = quadratic_extension_field_from_vec(&strings[1])?;
            let c2 = quadratic_extension_field_from_vec(&strings[2])?;
            Ok($curve::Fq6::new(c0, c1, c2))
        }
    }
    #[inline]
    fn quadratic_extension_field_from_vec(strings: &[String]) -> IoResult<$curve::Fq2> {
        if strings.len() != 2 {
            Err(SerializationError::InvalidData)
        } else {
            let c0 = parse_field(&strings[0])?;
            let c1 = parse_field(&strings[1])?;
            Ok($curve::Fq2::new(c0, c1))
        }
    }

    #[inline]
    fn parse_field(string: &str) -> IoResult<$curve::Fq> {
        $curve::Fq::from_str(string).map_err(|_| SerializationError::InvalidData)
    }
}
    };
}
struct FrVisitor<P: Pairing + CircomArkworksPairingBridge>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    phantom_data: PhantomData<P>,
}

impl<P: Pairing + CircomArkworksPairingBridge> FrVisitor<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    fn new() -> Self {
        Self {
            phantom_data: PhantomData,
        }
    }
}

impl<'de, P: Pairing + CircomArkworksPairingBridge> de::Visitor<'de> for FrVisitor<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    type Value = P::ScalarField;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an element over a PrimeField as string with radix 10")
    }
    fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        P::ScalarField::from_str(s).map_err(|_| de::Error::custom("invalid field element"))
    }
}
struct G1Visitor<P: Pairing + CircomArkworksPairingBridge>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    phantom_data: PhantomData<P>,
}

impl<P: Pairing + CircomArkworksPairingBridge> G1Visitor<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    fn new() -> Self {
        Self {
            phantom_data: PhantomData,
        }
    }
}

impl<'de, P: Pairing + CircomArkworksPairingBridge> de::Visitor<'de> for G1Visitor<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    type Value = P::G1Affine;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence of 3 strings, representing a projective point on G1")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let x = seq.next_element::<String>()?.ok_or(de::Error::custom(
            "expected G1 projective coordinates but x coordinate missing.".to_owned(),
        ))?;
        let y = seq.next_element::<String>()?.ok_or(de::Error::custom(
            "expected G1 projective coordinates but y coordinate missing.".to_owned(),
        ))?;
        let z = seq.next_element::<String>()?.ok_or(de::Error::custom(
            "expected G1 projective coordinates but z coordinate missing.".to_owned(),
        ))?;
        //check if there are no more elements
        if seq.next_element::<String>()?.is_some() {
            Err(de::Error::invalid_length(4, &self))
        } else {
            P::g1_from_strings_projective(&x, &y, &z)
                .map_err(|_| de::Error::custom("Invalid projective point on G1.".to_owned()))
        }
    }
}

struct G2Visitor<P: Pairing + CircomArkworksPairingBridge>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    phantom_data: PhantomData<P>,
}

impl<P: Pairing + CircomArkworksPairingBridge> TargetGroupVisitor<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    fn new() -> Self {
        Self {
            phantom_data: PhantomData,
        }
    }
}

impl<'de, P: Pairing + CircomArkworksPairingBridge> de::Visitor<'de> for G2Visitor<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    type Value = P::G2Affine;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter
            .write_str("a sequence of 3 sequences, representing a projective point on G2. The 3 sequences each consist of two strings")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: de::SeqAccess<'de>,
    {
        let x = seq.next_element::<Vec<String>>()?.ok_or(de::Error::custom(
            "expected G1 projective coordinates but x coordinate missing.".to_owned(),
        ))?;
        let y = seq.next_element::<Vec<String>>()?.ok_or(de::Error::custom(
            "expected G2 projective coordinates but y coordinate missing.".to_owned(),
        ))?;
        let z = seq.next_element::<Vec<String>>()?.ok_or(de::Error::custom(
            "expected G2 projective coordinates but z coordinate missing.".to_owned(),
        ))?;
        //check if there are no more elements
        if seq.next_element::<String>()?.is_some() {
            Err(de::Error::invalid_length(4, &self))
        } else if x.len() != 2 {
            Err(de::Error::custom(format!(
                "x coordinates need two field elements for G2, but got {}",
                x.len()
            )))
        } else if y.len() != 2 {
            Err(de::Error::custom(format!(
                "y coordinates need two field elements for G2, but got {}",
                y.len()
            )))
        } else if z.len() != 2 {
            Err(de::Error::custom(format!(
                "z coordinates need two field elements for G2, but got {}",
                z.len()
            )))
        } else {
            Ok(P::g2_from_strings_projective(&x[0], &x[1], &y[0], &y[1], &z[0], &z[1]).unwrap())
        }
    }
}

struct TargetGroupVisitor<P: Pairing + CircomArkworksPairingBridge>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    phantom_data: PhantomData<P>,
}

impl<P: Pairing + CircomArkworksPairingBridge> G2Visitor<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    fn new() -> Self {
        Self {
            phantom_data: PhantomData,
        }
    }
}

/// Bridge trait to serialize and deserialize pairings contained in circom files into and from [`ark_ec::pairing::Pairing`] representation
pub trait CircomArkworksPairingBridge: Pairing
where
    Self::BaseField: CircomArkworksPrimeFieldBridge,
    Self::ScalarField: CircomArkworksPrimeFieldBridge,
{
    /// Size of compressed element of G1 in bytes
    const G1_SERIALIZED_BYTE_SIZE_COMPRESSED: usize;
    /// Size of uncompressed element of G1 in bytes
    const G1_SERIALIZED_BYTE_SIZE_UNCOMPRESSED: usize;
    /// Size of compressed element of G2 in bytes
    const G2_SERIALIZED_BYTE_SIZE_COMPRESSED: usize;
    /// Size of uncompressed element of G2 in bytes
    const G2_SERIALIZED_BYTE_SIZE_UNCOMPRESSED: usize;
    /// Size of compressed element of Gt in bytes
    const GT_SERIALIZED_BYTE_SIZE_COMPRESSED: usize;
    /// Size of uncompressed element of Gt in bytes
    const GT_SERIALIZED_BYTE_SIZE_UNCOMPRESSED: usize;
    /// Returns the name of the curve as defined in circom
    fn get_circom_name() -> String;
    /// Deserializes element of G1 from reader where the element is already in montgomery form (no montgomery reduction performed)
    fn g1_from_reader(reader: impl Read) -> IoResult<Self::G1Affine>;
    /// Deserializes element of G2 from reader where the element is already in montgomery form (no montgomery reduction performed)
    fn g2_from_reader(reader: impl Read) -> IoResult<Self::G2Affine>;
    /// Deserializes element of G1 from strings representing projective coordinates
    fn g1_from_strings_projective(x: &str, y: &str, z: &str) -> IoResult<Self::G1Affine>;
    /// Deserializes element of G2 from strings representing projective coordinates
    fn g2_from_strings_projective(
        x0: &str,
        x1: &str,
        y0: &str,
        y1: &str,
        z0: &str,
        z1: &str,
    ) -> IoResult<Self::G2Affine>;
    /// Deserializes element of G1 using deserializer
    fn deserialize_g1_element<'de, D>(deserializer: D) -> Result<Self::G1Affine, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_seq(G1Visitor::<Self>::new())
    }
    /// Deserializes element of G2 using deserializer
    fn deserialize_g2_element<'de, D>(deserializer: D) -> Result<Self::G2Affine, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_seq(G2Visitor::<Self>::new())
    }
    /// Deserializes element of Gt using deserializer
    fn deserialize_gt_element<'de, D>(deserializer: D) -> Result<Self::TargetField, D::Error>
    where
        D: de::Deserializer<'de>;
    /// Deserializes (single) element of Scalarfield using deserializer
    fn deserialize_fr_element<'de, D>(deserializer: D) -> Result<Self::ScalarField, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserializer.deserialize_str(FrVisitor::<Self>::new())
    }
    /// Serializes element of G1 using serializer
    fn serialize_g1<S: Serializer>(p: &Self::G1Affine, ser: S) -> Result<S::Ok, S::Error> {
        let strings = Self::g1_to_strings_projective(p);
        let mut seq = ser.serialize_seq(Some(strings.len())).unwrap();
        for ele in strings {
            seq.serialize_element(&ele)?;
        }
        seq.end()
    }
    /// Serializes element of G1 into a vec of strings
    fn g1_to_strings_projective(p: &Self::G1Affine) -> Vec<String>;
    /// Serializes element of G2 using serializer
    fn serialize_g2<S: Serializer>(p: &Self::G2Affine, ser: S) -> Result<S::Ok, S::Error>;
    /// Serializes element of Gt using serializer
    fn serialize_gt<S: Serializer>(p: &Self::TargetField, ser: S) -> Result<S::Ok, S::Error>;
    /// Serializes (single) element of Scalarfield using serializer
    fn serialize_fr<S: Serializer>(p: &Self::ScalarField, ser: S) -> Result<S::Ok, S::Error>;
}

/// Bridge trait to deserialize field elements contained in circom files into [`ark_ff::PrimeField`] representation
pub trait CircomArkworksPrimeFieldBridge: PrimeField {
    /// Size of serialized field element in bytes
    const SERIALIZED_BYTE_SIZE: usize;
    /// Deserializes field elements and performs montgomery reduction
    fn from_reader(reader: impl Read) -> IoResult<Self>;
    /// deserializes field elements that are already in montgomery
    /// form. DOES NOT perform montgomery reduction
    fn from_reader_unchecked(reader: impl Read) -> IoResult<Self>;
    /// deserializes field elements that are multiplied by R^2 already (elements in zkey are of this form)
    fn from_reader_unchecked_for_zkey(reader: impl Read) -> IoResult<Self>;
}

impl_bn256!();
impl_bls12_381!();
