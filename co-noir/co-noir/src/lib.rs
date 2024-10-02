pub mod file_utils;
use acir::{
    acir_field::GenericFieldElement,
    native_types::{WitnessMap, WitnessStack},
};
use ark_ec::pairing::Pairing;
use ark_ff::{PrimeField, Zero};
use clap::{Args, ValueEnum};
use co_ultrahonk::prelude::{SharedBuilderVariable, UltraCircuitVariable};
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use mpc_core::protocols::{
    rep3::{
        self, network::Rep3Network, witness_extension_impl::Rep3VmType, Rep3PrimeFieldShare,
        Rep3Protocol,
    },
    shamir::{self, network::ShamirNetwork, ShamirProtocol},
};
use mpc_net::config::NetworkConfig;
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use std::{array, path::PathBuf};

#[derive(Clone, Debug)]
pub enum PubShared<F: Clone> {
    Public(F),
    Shared(F),
}

impl<F: Clone> PubShared<F> {
    pub fn from_shared(f: F) -> Self {
        Self::Shared(f)
    }

    pub fn set_public(&mut self) {
        if let Self::Shared(ref mut f) = self {
            *self = Self::Public(f.clone());
        }
    }
}

/// An enum representing the MPC protocol to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
#[clap(rename_all = "UPPER")]
pub enum MPCProtocol {
    /// A protocol based on the Replicated Secret Sharing Scheme for 3 parties.
    /// For more information see <https://eprint.iacr.org/2018/403.pdf>.
    REP3,
    /// A protocol based on Shamir Secret Sharing Scheme for n parties.
    /// For more information see <https://iacr.org/archive/crypto2007/46220565/46220565.pdf>.
    SHAMIR,
}

impl std::fmt::Display for MPCProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MPCProtocol::REP3 => write!(f, "REP3"),
            MPCProtocol::SHAMIR => write!(f, "SHAMIR"),
        }
    }
}

/// Cli arguments for `split_witness`
#[derive(Debug, Default, Serialize, Args)]
pub struct SplitWitnessCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the input witness file generated by Noir
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub witness: Option<PathBuf>,
    /// The path to the circuit file, generated by Noir
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The pairing friendly curve to be used
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out_dir: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    #[arg(short, long, default_value_t = 1)]
    pub threshold: usize,
    /// The number of parties
    #[arg(short, long, default_value_t = 3)]
    pub num_parties: usize,
}

/// Config for `split_witness`
#[derive(Debug, Deserialize)]
pub struct SplitWitnessConfig {
    /// The path to the input witness file generated by Circom
    pub witness: PathBuf,
    /// The path to the circuit file, generated by Noir
    pub circuit: PathBuf,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The path to the (existing) output directory
    pub out_dir: PathBuf,
    /// The threshold of tolerated colluding parties
    pub threshold: usize,
    /// The number of parties
    pub num_parties: usize,
}

/// Cli arguments for `split_input`
#[derive(Debug, Default, Clone, Serialize, Args)]
pub struct SplitInputCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the input JSON file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub input: Option<PathBuf>,
    /// The path to the circuit file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<String>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The path to the (existing) output directory
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out_dir: Option<PathBuf>,
}

/// Config for `split_input`
#[derive(Debug, Clone, Deserialize)]
pub struct SplitInputConfig {
    /// The path to the input JSON file
    pub input: PathBuf,
    /// The path to the circuit file
    pub circuit: String,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The path to the (existing) output directory
    pub out_dir: PathBuf,
}

/// Cli arguments for `generate_witness`
#[derive(Debug, Default, Serialize, Args)]
pub struct GenerateWitnessCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the input share file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub input: Option<PathBuf>,
    /// The path to the circuit file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<String>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The output file where the final witness share is written to
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
}

/// Config for `generate_witness`
#[derive(Debug, Deserialize)]
pub struct GenerateWitnessConfig {
    /// The path to the input share file
    pub input: PathBuf,
    /// The path to the circuit file
    pub circuit: String,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The output file where the final witness share is written to
    pub out: PathBuf,
    /// Network config
    pub network: NetworkConfig,
}

/// Cli arguments for `generate_proof`
#[derive(Debug, Serialize, Args)]
pub struct GenerateProofCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the witness share file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub witness: Option<PathBuf>,
    /// The path to the circuit file, generated by Noir
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub circuit: Option<PathBuf>,
    /// The path to the crs file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub crs: Option<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The output file where the final proof is written to. If not passed, this party will not write the proof to a file.
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
    /// The output JSON file where the public inputs are written to. If not passed, this party will not write the public inputs to a file.
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub public_input: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    #[arg(short, long, default_value_t = 1)]
    pub threshold: usize,
}

/// Config for `generate_proof`
#[derive(Debug, Deserialize)]
pub struct GenerateProofConfig {
    /// The path to the witness share file
    pub witness: PathBuf,
    /// The path to the circuit file, generated by Noir
    pub circuit: PathBuf,
    /// The path to the crs file
    pub crs: PathBuf,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The output file where the final proof is written to. If not passed, this party will not write the proof to a file.
    pub out: Option<PathBuf>,
    /// The output JSON file where the public inputs are written to. If not passed, this party will not write the public inputs to a file.
    pub public_input: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    pub threshold: usize,
    /// Network config
    pub network: NetworkConfig,
}

/// Prefix for config env variables
pub const CONFIG_ENV_PREFIX: &str = "CONOIR_";

/// Error type for config parsing and merging
#[derive(thiserror::Error, Debug)]
#[error(transparent)]
pub struct ConfigError(#[from] figment::error::Error);

macro_rules! impl_config {
    ($cli: ty, $config: ty) => {
        impl $config {
            /// Parse config from file, env, cli
            pub fn parse(cli: $cli) -> Result<Self, ConfigError> {
                if let Some(path) = &cli.config {
                    Ok(Figment::new()
                        .merge(Toml::file(path))
                        .merge(Env::prefixed(CONFIG_ENV_PREFIX))
                        .merge(Serialized::defaults(cli))
                        .extract()?)
                } else {
                    Ok(Figment::new()
                        .merge(Env::prefixed(CONFIG_ENV_PREFIX))
                        .merge(Serialized::defaults(cli))
                        .extract()?)
                }
            }
        }
    };
}

impl_config!(SplitInputCli, SplitInputConfig);
impl_config!(SplitWitnessCli, SplitWitnessConfig);
impl_config!(GenerateWitnessCli, GenerateWitnessConfig);
impl_config!(GenerateProofCli, GenerateProofConfig);

#[allow(clippy::type_complexity)]
pub fn share_rep3<P: Pairing, N: Rep3Network, R: Rng + CryptoRng>(
    witness: Vec<PubShared<P::ScalarField>>,
    rng: &mut R,
) -> [Vec<SharedBuilderVariable<Rep3Protocol<P::ScalarField, N>, P>>; 3] {
    let mut res = array::from_fn(|_| Vec::with_capacity(witness.len()));

    for witness in witness {
        match witness {
            PubShared::Public(f) => {
                for r in res.iter_mut() {
                    r.push(SharedBuilderVariable::from_public(f));
                }
            }
            PubShared::Shared(f) => {
                // res.push(SharedBuilderVariable::from_shared(f));
                let shares = rep3::utils::share_field_element(f, rng);
                for (r, share) in res.iter_mut().zip(shares) {
                    r.push(SharedBuilderVariable::from_shared(share));
                }
            }
        }
    }
    res
}

#[allow(clippy::type_complexity)]
pub fn share_shamir<P: Pairing, N: ShamirNetwork, R: Rng + CryptoRng>(
    witness: Vec<PubShared<P::ScalarField>>,
    degree: usize,
    num_parties: usize,
    rng: &mut R,
) -> Vec<Vec<SharedBuilderVariable<ShamirProtocol<P::ScalarField, N>, P>>> {
    let mut res = (0..num_parties)
        .map(|_| Vec::with_capacity(witness.len()))
        .collect::<Vec<_>>();

    for witness in witness {
        match witness {
            PubShared::Public(f) => {
                for r in res.iter_mut() {
                    r.push(SharedBuilderVariable::from_public(f));
                }
            }
            PubShared::Shared(f) => {
                // res.push(SharedBuilderVariable::from_shared(f));
                let shares = shamir::utils::share_field_element(f, degree, num_parties, rng);
                for (r, share) in res.iter_mut().zip(shares) {
                    r.push(SharedBuilderVariable::from_shared(share));
                }
            }
        }
    }
    res
}

pub fn share_input_rep3<P: Pairing, N: Rep3Network, R: Rng + CryptoRng>(
    initial_witness: WitnessMap<GenericFieldElement<P::ScalarField>>,
    rng: &mut R,
) -> [WitnessMap<Rep3PrimeFieldShare<P::ScalarField>>; 3] {
    let mut witnesses = array::from_fn(|_| WitnessMap::default());
    for (witness, v) in initial_witness.into_iter() {
        let v = v.into_repr();
        let shares = rep3::utils::share_field_element(v, rng);
        for (w, share) in witnesses.iter_mut().zip(shares) {
            w.insert(witness, share);
        }
    }

    witnesses
}

pub fn translate_witness_share_rep3<F: PrimeField>(
    witness: WitnessMap<Rep3PrimeFieldShare<F>>,
) -> WitnessMap<Rep3VmType<F>> {
    let mut result = WitnessMap::default();
    for (witness, v) in witness.into_iter() {
        result.insert(witness, Rep3VmType::Shared(v));
    }

    result
}

pub fn convert_witness_to_vec_rep3<P: Pairing, N: Rep3Network>(
    mut witness_stack: WitnessStack<Rep3VmType<P::ScalarField>>,
) -> Vec<SharedBuilderVariable<Rep3Protocol<P::ScalarField, N>, P>> {
    let witness_map = witness_stack
        .pop()
        .expect("Witness should be present")
        .witness;

    let mut wv = Vec::new();
    let mut index = 0;
    for (w, f) in witness_map.into_iter() {
        // ACIR uses a sparse format for WitnessMap where unused witness indices may be left unassigned.
        // To ensure that witnesses sit at the correct indices in the `WitnessVector`, we fill any indices
        // which do not exist within the `WitnessMap` with the dummy value of zero.
        while index < w.0 {
            wv.push(SharedBuilderVariable::from_public(P::ScalarField::zero()));
            index += 1;
        }
        match f {
            Rep3VmType::Public(f) => {
                wv.push(SharedBuilderVariable::from_public(f));
            }
            Rep3VmType::Shared(f) => {
                wv.push(SharedBuilderVariable::from_shared(f));
            }
            Rep3VmType::BitShared => panic!("BitShared not supported"),
        }
        index += 1;
    }
    wv
}
