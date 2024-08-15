#![warn(missing_docs)]
//! This crate provides a binary and associated helper library for running collaborative SNARK proofs.
use std::{io::Read, path::PathBuf, time::Instant};

use ark_ec::pairing::Pairing;
use circom_mpc_compiler::{CompilerBuilder, CompilerConfig};
use circom_mpc_vm::mpc_vm::VMConfig;
use circom_types::{
    groth16::{Groth16Proof, ZKey},
    traits::{CircomArkworksPairingBridge, CircomArkworksPrimeFieldBridge},
};
use clap::Args;
use clap::ValueEnum;
use co_circom_snarks::{SharedInput, SharedWitness};
use co_groth16::CoGroth16;
use color_eyre::eyre::Context;
use figment::{
    providers::{Env, Format, Serialized, Toml},
    Figment,
};
use mpc_core::{
    protocols::rep3::{
        network::{Rep3MpcNet, Rep3Network},
        Rep3Protocol,
    },
    traits::PrimeFieldMpcProtocol,
};
use mpc_net::config::NetworkConfig;
use serde::{Deserialize, Serialize};

/// A module for file utility functions.
pub mod file_utils;

/// An enum representing the ZK proof system to use.
#[derive(Debug, Clone, ValueEnum, Serialize, Deserialize)]
#[clap(rename_all = "lower")]
pub enum ProofSystem {
    /// The Groth16 proof system.
    Groth16,
    /// The Plonk proof system.
    Plonk,
}

impl std::fmt::Display for ProofSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofSystem::Groth16 => write!(f, "Plonk"),
            ProofSystem::Plonk => write!(f, "Groth16"),
        }
    }
}

/// An enum representing the MPC protocol to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MPCCurve {
    /// The BN254 curve (called BN128 in circom).
    BN254,
    /// The BLS12_381 curve.
    BLS12_381,
}

impl ValueEnum for MPCCurve {
    fn value_variants<'a>() -> &'a [Self] {
        &[MPCCurve::BN254, MPCCurve::BLS12_381]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            MPCCurve::BN254 => Some(clap::builder::PossibleValue::new("BN254")),
            MPCCurve::BLS12_381 => Some(clap::builder::PossibleValue::new("BLS12-381")),
        }
    }
}

impl std::fmt::Display for MPCCurve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MPCCurve::BN254 => write!(f, "BN254"),
            MPCCurve::BLS12_381 => write!(f, "BLS12-381"),
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
    /// The path to the input witness file generated by Circom
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub witness: Option<PathBuf>,
    /// The path to the r1cs file, generated by Circom compiler
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub r1cs: Option<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The pairing friendly curve to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub curve: Option<MPCCurve>,
    /// The path to the (existing) output directory
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
    /// The path to the r1cs file, generated by Circom compiler
    pub r1cs: PathBuf,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The pairing friendly curve to be used
    pub curve: MPCCurve,
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
    /// The path to Circom library files
    #[arg(long)]
    pub link_library: Vec<String>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The pairing friendly curve to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub curve: Option<MPCCurve>,
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
    /// The path to Circom library files
    pub link_library: Vec<String>,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The pairing friendly curve to be used
    pub curve: MPCCurve,
    /// The path to the (existing) output directory
    pub out_dir: PathBuf,
    /// MPC compiler config
    #[serde(default)]
    pub compiler: CompilerConfig,
}

/// Cli arguments for `merge_input_shares`
#[derive(Debug, Default, Serialize, Args)]
pub struct MergeInputSharesCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the input JSON file
    #[arg(long)]
    pub inputs: Vec<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The pairing friendly curve to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub curve: Option<MPCCurve>,
    /// The output file where the merged input share is written to
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
}

/// Config for `merge_input_shares`
#[derive(Debug, Deserialize)]
pub struct MergeInputSharesConfig {
    /// The path to the input JSON file
    pub inputs: Vec<PathBuf>,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The pairing friendly curve to be used
    pub curve: MPCCurve,
    /// The output file where the merged input share is written to
    pub out: PathBuf,
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
    /// The path to Circom library files
    #[arg(long)]
    pub link_library: Vec<String>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The pairing friendly curve to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub curve: Option<MPCCurve>,
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
    /// The path to Circom library files
    pub link_library: Vec<String>,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The pairing friendly curve to be used
    pub curve: MPCCurve,
    /// The output file where the final witness share is written to
    pub out: PathBuf,
    /// MPC compiler config
    #[serde(default)]
    pub compiler: CompilerConfig,
    /// MPC VM config
    #[serde(default)]
    pub vm: VMConfig,
    /// Network config
    pub network: NetworkConfig,
}

/// Cli arguments for `transalte_witness`
#[derive(Debug, Serialize, Args)]
pub struct TranslateWitnessCli {
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the witness share file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub witness: Option<PathBuf>,
    /// The MPC protocol that was used for the witness generation
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub src_protocol: Option<MPCProtocol>,
    /// The MPC protocol to be used for the proof generation
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub target_protocol: Option<MPCProtocol>,
    /// The pairing friendly curve to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub curve: Option<MPCCurve>,
    /// The output file where the final witness share is written to
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub out: Option<PathBuf>,
}

/// Config for `transalte_witness`
#[derive(Debug, Deserialize)]
pub struct TranslateWitnessConfig {
    /// The path to the witness share file
    pub witness: PathBuf,
    /// The MPC protocol that was used for the witness generation
    pub src_protocol: MPCProtocol,
    /// The MPC protocol to be used for the proof generation
    pub target_protocol: MPCProtocol,
    /// The pairing friendly curve to be used
    pub curve: MPCCurve,
    /// The output file where the final witness share is written to
    pub out: PathBuf,
    /// Network config
    pub network: NetworkConfig,
}

/// Cli arguments for `generate_proof`
#[derive(Debug, Serialize, Args)]
pub struct GenerateProofCli {
    /// The proof system to be used
    #[arg(value_enum)]
    pub proof_system: ProofSystem,
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the witness share file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub witness: Option<PathBuf>,
    /// The path to the proving key (.zkey) file, generated by snarkjs setup phase
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub zkey: Option<PathBuf>,
    /// The MPC protocol to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub protocol: Option<MPCProtocol>,
    /// The pairing friendly curve to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub curve: Option<MPCCurve>,
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
    /// The proof system to be used
    pub proof_system: ProofSystem,
    /// The path to the witness share file
    pub witness: PathBuf,
    /// The path to the proving key (.zkey) file, generated by snarkjs setup phase
    pub zkey: PathBuf,
    /// The MPC protocol to be used
    pub protocol: MPCProtocol,
    /// The pairing friendly curve to be used
    pub curve: MPCCurve,
    /// The output file where the final proof is written to. If not passed, this party will not write the proof to a file.
    pub out: Option<PathBuf>,
    /// The output JSON file where the public inputs are written to. If not passed, this party will not write the public inputs to a file.
    pub public_input: Option<PathBuf>,
    /// The threshold of tolerated colluding parties
    pub threshold: usize,
    /// Network config
    pub network: NetworkConfig,
}

/// Cli arguments for `verify`
#[derive(Debug, Serialize, Args)]
pub struct VerifyCli {
    /// The proof system to be used
    #[arg(value_enum)]
    pub proof_system: ProofSystem,
    /// The path to the config file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub config: Option<PathBuf>,
    /// The path to the proof file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub proof: Option<PathBuf>,
    /// The pairing friendly curve to be used
    #[arg(long, value_enum)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub curve: Option<MPCCurve>,
    /// The path to the verification key file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub vk: Option<PathBuf>,
    /// The path to the public input JSON file
    #[arg(long)]
    #[serde(skip_serializing_if = "::std::option::Option::is_none")]
    pub public_input: Option<PathBuf>,
}

/// Config for `verify`
#[derive(Debug, Deserialize)]
pub struct VerifyConfig {
    /// The proof system to be used
    pub proof_system: ProofSystem,
    /// The path to the proof file
    pub proof: PathBuf,
    /// The pairing friendly curve to be used
    pub curve: MPCCurve,
    /// The path to the verification key file
    pub vk: PathBuf,
    /// The path to the public input JSON file
    pub public_input: PathBuf,
}

/// Prefix for config env variables
pub const CONFIG_ENV_PREFIX: &str = "COCIRCOM_";

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
impl_config!(MergeInputSharesCli, MergeInputSharesConfig);
impl_config!(GenerateWitnessCli, GenerateWitnessConfig);
impl_config!(TranslateWitnessCli, TranslateWitnessConfig);
impl_config!(GenerateProofCli, GenerateProofConfig);
impl_config!(VerifyCli, VerifyConfig);

/// Try to parse a [SharedWitness] from a [Read]er.
pub fn parse_witness_share<R: Read, P: Pairing, T: PrimeFieldMpcProtocol<P::ScalarField>>(
    reader: R,
) -> color_eyre::Result<SharedWitness<T, P>> {
    bincode::deserialize_from(reader).context("trying to parse witness share file")
}

/// Try to parse a [SharedInput] from a [Read]er.
pub fn parse_shared_input<R: Read, P: Pairing, T: PrimeFieldMpcProtocol<P::ScalarField>>(
    reader: R,
) -> color_eyre::Result<SharedInput<T, P>> {
    bincode::deserialize_from(reader).context("trying to parse input share file")
}

/// Invoke the MPC witness generation process. It will return a [SharedWitness] if successful.
/// It executes several steps:
/// 1. Parse the circuit file.
/// 2. Compile the circuit to MPC VM bytecode.
/// 3. Set up a network connection to the MPC network.
/// 4. Execute the bytecode on the MPC VM to generate the witness.
pub fn generate_witness_rep3<P: Pairing>(
    circuit: String,
    link_library: Vec<String>,
    input_share: SharedInput<Rep3Protocol<P::ScalarField, Rep3MpcNet>, P>,
    config: GenerateWitnessConfig,
) -> color_eyre::Result<SharedWitness<Rep3Protocol<P::ScalarField, Rep3MpcNet>, P>> {
    let circuit_path = PathBuf::from(&circuit);
    file_utils::check_file_exists(&circuit_path)?;

    // parse circuit file & put through our compiler
    let mut builder = CompilerBuilder::<P>::new(config.compiler, circuit);
    for lib in link_library {
        builder = builder.link_library(lib);
    }
    let parsed_circom_circuit = builder
        .build()
        .parse()
        .context("while parsing circuit file")?;

    // connect to network
    let net = Rep3MpcNet::new(config.network).context("while connecting to network")?;
    let id = usize::from(net.get_id());

    // init MPC protocol
    let rep3_vm = parsed_circom_circuit
        .to_rep3_vm_with_network(net, config.vm)
        .context("while constructing MPC VM")?;

    // execute witness generation in MPC
    let start = Instant::now();
    let result_witness_share = rep3_vm
        .run(input_share)
        .context("while running witness generation")?;
    let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
    tracing::info!("Party {}: Witness extension took {} ms", id, duration_ms);
    Ok(result_witness_share.into_shared_witness())
}

/// Invoke the MPC proof generation process. It will return a [`Groth16Proof`] if successful.
/// It executes several steps:
/// 1. Construct a [Rep3Protocol] from the network configuration.
/// 2. Construct a [CoGroth16] prover from the protocol.
/// 3. Execute the proof in MPC
pub fn prove_with_matrices_rep3<P: Pairing + CircomArkworksPairingBridge>(
    witness_share: SharedWitness<Rep3Protocol<P::ScalarField, Rep3MpcNet>, P>,
    config: NetworkConfig,
    zkey: ZKey<P>,
) -> color_eyre::Result<Groth16Proof<P>>
where
    P::ScalarField: CircomArkworksPrimeFieldBridge,
    P::BaseField: CircomArkworksPrimeFieldBridge,
{
    tracing::info!("establishing network....");
    // connect to network
    let net = Rep3MpcNet::new(config)?;
    tracing::info!("done!");
    // init MPC protocol
    tracing::info!("building protocol...");
    let protocol = Rep3Protocol::<P::ScalarField, _>::new(net)?;
    tracing::info!("done!");
    let mut prover = CoGroth16::<Rep3Protocol<P::ScalarField, _>, P>::new(protocol);
    tracing::info!("starting prover...");
    // execute prover in MPC
    prover.prove(&zkey, witness_share)
}
