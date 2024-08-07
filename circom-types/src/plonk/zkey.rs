//Copyright (c) 2021 Georgios Konstantopoulos
//
//Permission is hereby granted, free of charge, to any
//person obtaining a copy of this software and associated
//documentation files (the "Software"), to deal in the
//Software without restriction, including without
//limitation the rights to use, copy, modify, merge,
//publish, distribute, sublicense, and/or sell copies of
//the Software, and to permit persons to whom the Software
//is furnished to do so, subject to the following
//conditions:
//
//The above copyright notice and this permission notice
//shall be included in all copies or substantial portions
//of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
//ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
//TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
//PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
//SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
//OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
//IN CONNECTION WITH THE SOFTWARE O THE USE OR OTHER
//DEALINGS IN THE SOFTWARE.R

//! This module defines the [`ZKey`] struct that implements deserialization of circom zkey files via [`ZKey::from_reader`].
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;

use ark_serialize::CanonicalDeserialize;
use std::io::{Cursor, Read};

use crate::{
    binfile::{BinFile, ZKeyParserError, ZKeyParserResult},
    traits::{CircomArkworksPairingBridge, CircomArkworksPrimeFieldBridge},
};

macro_rules! u32_to_usize {
    ($x: expr) => {
        usize::try_from($x).expect("u32 fits into usize")
    };
}

/// Represents a zkey in the format defined by circom. Implements [`ZKey::from_reader`] to deserialize a zkey from a reader.
#[derive(Clone)]
pub struct ZKey<P: Pairing> {
    /// The amount of vars in the circuit
    pub n_vars: usize,
    /// The amount of public values in the circuit
    pub n_public: usize,
    /// The domain size (power of two)
    pub domain_size: usize,
    /// The amounts of additions
    pub n_additions: usize,
    /// The amounts of constraints
    pub n_constraints: usize,
    /// The verifying key
    pub verifying_key: VerifyingKey<P>,
    /// The indices of the additions of the witness and their respective blinding factors
    pub additions: Vec<Additions<P>>,
    /// The witness indices of the signals of wire mapping a
    pub map_a: Vec<usize>,
    /// The witness indices of the signals of wire mapping b
    pub map_b: Vec<usize>,
    /// The witness indices of the signals of wire mapping c
    pub map_c: Vec<usize>,
    /// Qm polynomial
    pub qm_poly: Polynomial<P>,
    /// Ql polynomial
    pub ql_poly: Polynomial<P>,
    /// Qr polynomial
    pub qr_poly: Polynomial<P>,
    /// Qo polynomial
    pub qo_poly: Polynomial<P>,
    /// Qc polynomial
    pub qc_poly: Polynomial<P>,
    /// σ1 polynomial
    pub s1_poly: Polynomial<P>,
    /// σ2 polynomial
    pub s2_poly: Polynomial<P>,
    /// σ3 polynomial
    pub s3_poly: Polynomial<P>,
    /// Lagrange polynomials. One [Polynomial] for each public input.
    pub lagrange: Vec<Polynomial<P>>,
    /// The powers of 𝜏
    pub p_tau: Vec<P::G1Affine>,
}

/// A polynomial in coefficient and evaluation form for PLONK's [ZKey].
#[derive(Clone)]
pub struct Polynomial<P: Pairing> {
    /// The polynomial's coefficient form
    pub coeffs: Vec<P::ScalarField>,
    /// The polynomial's evaluation form
    pub evaluations: Vec<P::ScalarField>,
}

#[derive(Clone)]
/// The indices and blinding factors for all additions necessary during a PLONK proof. The id's represent the index
/// in the witness.
pub struct Additions<P: Pairing> {
    /// The index of lhs
    pub signal_id1: u32,
    /// The index of rhs
    pub signal_id2: u32,
    /// The blinding factor of lhs
    pub factor1: P::ScalarField,
    /// The blinding factor of rhs
    pub factor2: P::ScalarField,
}

/// The verifying key for a PLONK proof.
#[derive(Default, Clone, Debug)]
pub struct VerifyingKey<P: Pairing> {
    /// k1
    pub k1: P::ScalarField,
    /// k2
    pub k2: P::ScalarField,
    /// The evaluation of [`ZKey::qm_poly`] with [`ZKey::p_tau`]
    pub qm: P::G1Affine,
    /// The evaluation of [`ZKey::qr_poly`] with [`ZKey::p_tau`]
    pub ql: P::G1Affine,
    /// The evaluation of [`ZKey::ql_poly`] with [`ZKey::p_tau`]
    pub qr: P::G1Affine,
    /// The evaluation of [`ZKey::qo_poly`] with [`ZKey::p_tau`]
    pub qo: P::G1Affine,
    /// The evaluation of [`ZKey::qc_poly`] with [`ZKey::p_tau`]
    pub qc: P::G1Affine,
    /// The evaluation of [`ZKey::s1_poly`] with [`ZKey::p_tau`]
    pub s1: P::G1Affine,
    /// The evaluation of [`ZKey::s2_poly`] with [`ZKey::p_tau`]
    pub s2: P::G1Affine,
    /// The evaluation of [`ZKey::s3_poly`] with [`ZKey::p_tau`]
    pub s3: P::G1Affine,
    /// x_2
    pub x_2: P::G2Affine,
}

impl<P: Pairing + CircomArkworksPairingBridge> ZKey<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    /// Deserializes a [`ZKey`] from a reader.
    pub fn from_reader<R: Read>(mut reader: R) -> ZKeyParserResult<Self> {
        BinFile::<P>::new(&mut reader)?.try_into()
    }

    fn additions_indices<R: Read>(
        n_additions: usize,
        mut reader: R,
    ) -> ZKeyParserResult<Vec<Additions<P>>> {
        let mut additions = Vec::with_capacity(n_additions);
        for _ in 0..n_additions {
            let signal_id1 = u32::deserialize_uncompressed(&mut reader)?;
            let signal_id2 = u32::deserialize_uncompressed(&mut reader)?;
            let factor1 = P::ScalarField::from_reader(&mut reader)?;
            let factor2 = P::ScalarField::from_reader(&mut reader)?;
            additions.push(Additions {
                signal_id1,
                signal_id2,
                factor1,
                factor2,
            })
        }
        Ok(additions)
    }

    fn id_map<R: Read>(n_constraints: usize, mut reader: R) -> ZKeyParserResult<Vec<usize>> {
        let mut map = Vec::with_capacity(n_constraints);
        for _ in 0..n_constraints {
            map.push(u32_to_usize!(u32::deserialize_uncompressed(&mut reader)?));
        }
        Ok(map)
    }

    fn evaluations<R: Read>(domain_size: usize, mut reader: R) -> ZKeyParserResult<Polynomial<P>> {
        let mut coeffs = Vec::with_capacity(domain_size);
        for _ in 0..domain_size {
            coeffs.push(P::ScalarField::deserialize_compressed(&mut reader)?);
        }

        let mut evaluations = Vec::with_capacity(domain_size * 4);
        for _ in 0..domain_size * 4 {
            evaluations.push(P::ScalarField::deserialize_compressed(&mut reader)?);
        }
        Ok(Polynomial {
            coeffs,
            evaluations,
        })
    }

    fn lagrange<R: Read>(
        n_public: usize,
        domain_size: usize,
        mut reader: R,
    ) -> ZKeyParserResult<Vec<Polynomial<P>>> {
        let mut lagrange = Vec::with_capacity(n_public);
        for _ in 0..n_public {
            lagrange.push(Self::evaluations(domain_size, &mut reader)?);
        }
        Ok(lagrange)
    }

    fn taus<R: Read>(domain_size: usize, reader: R) -> ZKeyParserResult<Vec<P::G1Affine>> {
        // //TODO: why domain size + 6?
        Ok(P::g1_vec_from_reader(reader, domain_size + 6)?)
    }
}

impl<P: Pairing + CircomArkworksPairingBridge> TryFrom<BinFile<P>> for ZKey<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    type Error = ZKeyParserError;
    fn try_from(mut binfile: BinFile<P>) -> Result<Self, Self::Error> {
        let header = PlonkHeader::<P>::read(&mut binfile.take_section(2))?;
        let n_vars = header.n_vars;
        let n_additions = header.n_additions;
        let n_constraints = header.n_constraints;
        let n_public = header.n_public;
        let domain_size = header.domain_size;
        //the sigmas are in the same section - so we split it here in separate chunks
        let sigma_section_size = domain_size * header.n8r + domain_size * 4 * header.n8r;

        let add_section = binfile.take_section(3);
        let a_section = binfile.take_section(4);
        let b_section = binfile.take_section(5);
        let c_section = binfile.take_section(6);
        let qm_section = binfile.take_section(7);
        let ql_section = binfile.take_section(8);
        let qr_section = binfile.take_section(9);
        let q0_section = binfile.take_section(10);
        let qc_section = binfile.take_section(11);
        let sigma_sections = binfile.take_section_raw(12);
        let l_section = binfile.take_section(13);
        let t_section = binfile.take_section(14);
        let sigma1_section = Cursor::new(&sigma_sections[..sigma_section_size]);
        let sigma2_section =
            Cursor::new(&sigma_sections[sigma_section_size..sigma_section_size * 2]);
        let sigma3_section = Cursor::new(&sigma_sections[sigma_section_size * 2..]);

        let mut additions = None;
        let mut map_a = None;
        let mut map_b = None;
        let mut map_c = None;
        let mut qm = None;
        let mut ql = None;
        let mut qr = None;
        let mut q0 = None;
        let mut qc = None;
        let mut sigma1 = None;
        let mut sigma2 = None;
        let mut sigma3 = None;
        let mut lagrange = None;
        let mut p_tau = None;
        rayon::scope(|s| {
            s.spawn(|_| additions = Some(Self::additions_indices(n_additions, add_section)));
            s.spawn(|_| map_a = Some(Self::id_map(n_constraints, a_section)));
            s.spawn(|_| map_b = Some(Self::id_map(n_constraints, b_section)));
            s.spawn(|_| map_c = Some(Self::id_map(n_constraints, c_section)));
            s.spawn(|_| qm = Some(Self::evaluations(domain_size, qm_section)));
            s.spawn(|_| ql = Some(Self::evaluations(domain_size, ql_section)));
            s.spawn(|_| qr = Some(Self::evaluations(domain_size, qr_section)));
            s.spawn(|_| q0 = Some(Self::evaluations(domain_size, q0_section)));
            s.spawn(|_| qc = Some(Self::evaluations(domain_size, qc_section)));
            s.spawn(|_| sigma1 = Some(Self::evaluations(domain_size, sigma1_section)));
            s.spawn(|_| sigma2 = Some(Self::evaluations(domain_size, sigma2_section)));
            s.spawn(|_| sigma3 = Some(Self::evaluations(domain_size, sigma3_section)));
            s.spawn(|_| lagrange = Some(Self::lagrange(n_public, domain_size, l_section)));
            s.spawn(|_| p_tau = Some(Self::taus(domain_size, t_section)));
        });
        Ok(Self {
            n_vars,
            n_public,
            domain_size,
            n_additions,
            n_constraints,
            verifying_key: header.verifying_key,
            //we unwrap all elements here, as we know they have to be Some.
            //this thread automatically joins on the rayon scope, therefore we can
            //only be here if the scope finished.
            //Even on the error case, we then have a Some value
            additions: additions.unwrap()?,
            map_a: map_a.unwrap()?,
            map_b: map_b.unwrap()?,
            map_c: map_c.unwrap()?,
            qm_poly: qm.unwrap()?,
            ql_poly: ql.unwrap()?,
            qr_poly: qr.unwrap()?,
            qo_poly: q0.unwrap()?,
            qc_poly: qc.unwrap()?,
            s1_poly: sigma1.unwrap()?,
            s2_poly: sigma2.unwrap()?,
            s3_poly: sigma3.unwrap()?,
            lagrange: lagrange.unwrap()?,
            p_tau: p_tau.unwrap()?,
        })
    }
}

impl<P: Pairing + CircomArkworksPairingBridge> VerifyingKey<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    fn new<R: Read>(mut reader: R) -> ZKeyParserResult<Self> {
        let k1 = <P::ScalarField>::from_reader(&mut reader)?;
        let k2 = <P::ScalarField>::from_reader(&mut reader)?;
        let qm = P::g1_from_reader(&mut reader)?;
        let ql = P::g1_from_reader(&mut reader)?;
        let qr = P::g1_from_reader(&mut reader)?;
        let qo = P::g1_from_reader(&mut reader)?;
        let qc = P::g1_from_reader(&mut reader)?;
        let s1 = P::g1_from_reader(&mut reader)?;
        let s2 = P::g1_from_reader(&mut reader)?;
        let s3 = P::g1_from_reader(&mut reader)?;
        let x2 = P::g2_from_reader(&mut reader)?;

        Ok(Self {
            k1,
            k2,
            qm,
            ql,
            qr,
            qo,
            qc,
            s1,
            s2,
            s3,
            x_2: x2,
        })
    }
}

#[derive(Clone)]
struct PlonkHeader<P: Pairing> {
    n8r: usize,
    n_vars: usize,
    n_public: usize,
    domain_size: usize,
    n_additions: usize,
    n_constraints: usize,
    verifying_key: VerifyingKey<P>,
}

impl<P: Pairing + CircomArkworksPairingBridge> PlonkHeader<P>
where
    P::BaseField: CircomArkworksPrimeFieldBridge,
    P::ScalarField: CircomArkworksPrimeFieldBridge,
{
    fn read<R: Read>(mut reader: &mut R) -> ZKeyParserResult<Self> {
        let _n8q: u32 = u32::deserialize_uncompressed(&mut reader)?;
        //modulos of BaseField
        let q = <P::BaseField as PrimeField>::BigInt::deserialize_uncompressed(&mut reader)?;
        let modulus = <P::BaseField as PrimeField>::MODULUS;
        if q != modulus {
            return Err(ZKeyParserError::InvalidPrimeInHeader);
        }
        let n8r = u32::deserialize_uncompressed(&mut reader)?;
        //modulos of ScalarField
        let r = <P::ScalarField as PrimeField>::BigInt::deserialize_uncompressed(&mut reader)?;
        let modulus = <P::ScalarField as PrimeField>::MODULUS;
        if r != modulus {
            return Err(ZKeyParserError::InvalidPrimeInHeader);
        }
        let n_vars = u32::deserialize_uncompressed(&mut reader)?;
        let n_public = u32::deserialize_uncompressed(&mut reader)?;
        let domain_size = u32::deserialize_uncompressed(&mut reader)?;
        let n_additions = u32::deserialize_uncompressed(&mut reader)?;
        let n_constraints = u32::deserialize_uncompressed(&mut reader)?;
        let verifying_key = VerifyingKey::new(&mut reader)?;
        Ok(Self {
            n8r: u32_to_usize!(n8r),
            n_vars: u32_to_usize!(n_vars),
            n_public: u32_to_usize!(n_public),
            domain_size: u32_to_usize!(domain_size),
            n_additions: u32_to_usize!(n_additions),
            n_constraints: u32_to_usize!(n_constraints),
            verifying_key,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    use ark_bn254::Bn254;
    use std::str::FromStr;

    use super::ZKey;
    use ark_ec::pairing::Pairing;

    macro_rules! bn_fr_from_lines {
        ($lines: expr) => {
            ark_bn254::Fr::from_str($lines.next().unwrap().unwrap().as_str()).unwrap()
        };
    }

    macro_rules! batch_to_fr {
        ($lines:expr, $amount: expr) => {
            (0..$amount)
                .into_iter()
                .map(|_| bn_fr_from_lines!($lines))
                .collect::<Vec<_>>()
        };
    }

    macro_rules! g1_from_lines {
        ($lines: expr) => {
            <ark_bn254::Bn254 as Pairing>::G1Affine::new(
                ark_bn254::Fq::from_str($lines.next().unwrap().unwrap().as_str()).unwrap(),
                ark_bn254::Fq::from_str($lines.next().unwrap().unwrap().as_str()).unwrap(),
            )
        };
    }

    macro_rules! batch_to_g1 {
        ($lines:expr, $amount: expr) => {
            (0..$amount)
                .into_iter()
                .map(|_| g1_from_lines!($lines))
                .collect::<Vec<_>>()
        };
    }
    macro_rules! g2_from_lines {
        ($lines: expr) => {
            <ark_bn254::Bn254 as Pairing>::G2Affine::new(
                ark_bn254::Fq2::new(
                    ark_bn254::Fq::from_str($lines.next().unwrap().unwrap().as_str()).unwrap(),
                    ark_bn254::Fq::from_str($lines.next().unwrap().unwrap().as_str()).unwrap(),
                ),
                ark_bn254::Fq2::new(
                    ark_bn254::Fq::from_str($lines.next().unwrap().unwrap().as_str()).unwrap(),
                    ark_bn254::Fq::from_str($lines.next().unwrap().unwrap().as_str()).unwrap(),
                ),
            )
        };
    }

    #[test]
    fn can_serde_zkey_bn254() {
        let mut reader =
            BufReader::new(File::open("../test_vectors/Plonk/bn254/multiplier2.zkey").unwrap());
        let mut lines =
            BufReader::new(File::open("../test_vectors/Plonk/bn254/multiplier2.zkey.kat").unwrap())
                .lines();
        let zkey = ZKey::<Bn254>::from_reader(&mut reader).unwrap();
        assert_eq!(zkey.n_vars, 4);
        assert_eq!(zkey.n_public, 1);
        assert_eq!(zkey.domain_size, 8);
        assert_eq!(zkey.n_additions, 0);
        assert_eq!(zkey.n_constraints, 2);
        assert_eq!(zkey.verifying_key.k1, bn_fr_from_lines!(lines));
        assert_eq!(zkey.verifying_key.k2, bn_fr_from_lines!(lines));
        assert_eq!(zkey.verifying_key.qm, g1_from_lines!(lines));
        assert_eq!(zkey.verifying_key.ql, g1_from_lines!(lines));
        assert_eq!(zkey.verifying_key.qr, ark_bn254::G1Affine::identity());
        assert_eq!(zkey.verifying_key.qo, g1_from_lines!(lines));
        assert_eq!(zkey.verifying_key.qc, g1_from_lines!(lines));
        assert_eq!(zkey.verifying_key.s1, g1_from_lines!(lines));
        assert_eq!(zkey.verifying_key.s2, g1_from_lines!(lines));
        assert_eq!(zkey.verifying_key.s3, g1_from_lines!(lines));
        assert_eq!(zkey.verifying_key.x_2, g2_from_lines!(lines));
        assert!(zkey.additions.is_empty());
        assert_eq!(zkey.map_a, vec![1, 2]);
        assert_eq!(zkey.map_b, vec![0, 3]);
        assert_eq!(zkey.map_c, vec![0, 1]);
        assert_eq!(zkey.s1_poly.coeffs, batch_to_fr!(lines, zkey.domain_size));
        assert_eq!(
            zkey.s1_poly.evaluations,
            batch_to_fr!(lines, zkey.domain_size * 4)
        );
        assert_eq!(zkey.s2_poly.coeffs, batch_to_fr!(lines, zkey.domain_size));
        assert_eq!(
            zkey.s2_poly.evaluations,
            batch_to_fr!(lines, zkey.domain_size * 4)
        );
        assert_eq!(zkey.s3_poly.coeffs, batch_to_fr!(lines, zkey.domain_size));
        assert_eq!(
            zkey.s3_poly.evaluations,
            batch_to_fr!(lines, zkey.domain_size * 4)
        );
        assert_eq!(zkey.ql_poly.coeffs, batch_to_fr!(lines, zkey.domain_size));
        assert_eq!(zkey.qr_poly.coeffs, batch_to_fr!(lines, zkey.domain_size));
        assert_eq!(zkey.qm_poly.coeffs, batch_to_fr!(lines, zkey.domain_size));
        assert_eq!(zkey.qo_poly.coeffs, batch_to_fr!(lines, zkey.domain_size));
        assert_eq!(zkey.qc_poly.coeffs, batch_to_fr!(lines, zkey.domain_size));
        assert_eq!(
            zkey.ql_poly.evaluations,
            batch_to_fr!(lines, zkey.domain_size * 4)
        );
        assert_eq!(
            zkey.qr_poly.evaluations,
            batch_to_fr!(lines, zkey.domain_size * 4)
        );
        assert_eq!(
            zkey.qm_poly.evaluations,
            batch_to_fr!(lines, zkey.domain_size * 4)
        );
        assert_eq!(
            zkey.qo_poly.evaluations,
            batch_to_fr!(lines, zkey.domain_size * 4)
        );
        assert_eq!(
            zkey.qc_poly.evaluations,
            batch_to_fr!(lines, zkey.domain_size * 4)
        );
        assert_eq!(zkey.lagrange.len(), 1);
        assert_eq!(
            zkey.lagrange[0].coeffs,
            batch_to_fr!(lines, zkey.domain_size)
        );
        assert_eq!(
            zkey.lagrange[0].evaluations,
            batch_to_fr!(lines, zkey.domain_size * 4)
        );
        assert_eq!(zkey.p_tau, batch_to_g1!(lines, 14));
    }
}
