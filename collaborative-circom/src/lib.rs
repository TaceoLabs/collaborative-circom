#![warn(missing_docs)]
//! This crate provides a binary and associated helper library for running collaborative SNARK proofs.
use std::{io::Read, path::PathBuf};

use ark_ec::pairing::Pairing;
use ark_groth16::Proof;
use circom_mpc_compiler::{CompilerBuilder, CompilerConfig};
use circom_mpc_vm::mpc_vm::VMConfig;
use circom_types::{
    groth16::zkey::ZKey,
    traits::{CircomArkworksPairingBridge, CircomArkworksPrimeFieldBridge},
};
use clap::ValueEnum;
use collaborative_groth16::groth16::{CollaborativeGroth16, SharedInput, SharedWitness};
use color_eyre::eyre::{Context, ContextCompat};
use figment::{
    providers::{Env, Format, Toml},
    Figment,
};
use mpc_core::{
    protocols::rep3::{network::Rep3MpcNet, Rep3Protocol},
    traits::{FFTPostProcessing, PrimeFieldMpcProtocol},
};
use mpc_net::config::NetworkConfig;
use serde::Deserialize;

/// A module for file utility functions.
pub mod file_utils;

/// An enum representing the ZK proof system to use.
#[derive(Clone, ValueEnum)]
pub enum ProofSystem {
    /// The Groth16 proof system.
    Groth16,
    /// The Plonk proof system.
    Plonk,
}

/// An enum representing the MPC protocol to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
pub enum MPCProtocol {
    /// A protocol based on the Replicated Secret Sharing Scheme for 3 parties.
    /// For more information see <https://eprint.iacr.org/2018/403.pdf>.
    REP3,
    /// A protocol based on Shamir Secret Sharing Scheme for n parties.
    /// For more information see <https://iacr.org/archive/crypto2007/46220565/46220565.pdf>.
    SHAMIR,
}

impl ValueEnum for MPCProtocol {
    fn value_variants<'a>() -> &'a [Self] {
        &[MPCProtocol::REP3, MPCProtocol::SHAMIR]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            MPCProtocol::REP3 => Some(clap::builder::PossibleValue::new("REP3")),
            MPCProtocol::SHAMIR => Some(clap::builder::PossibleValue::new("SHAMIR")),
        }
    }
}

impl std::fmt::Display for MPCProtocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MPCProtocol::REP3 => write!(f, "REP3"),
            MPCProtocol::SHAMIR => write!(f, "SHAMIR"),
        }
    }
}

/// Collaborative-circom configuration
#[derive(Debug, Deserialize, Default)]
pub struct Config {
    /// Mpc curve to be sued
    pub curve: Option<MPCCurve>,
    /// Mpc protocl to bu sued
    pub protocol: Option<MPCProtocol>,
    /// Mpc-vm config
    pub vm: Option<VMConfig>,
    /// Mpc-compiler config
    pub compiler: Option<CompilerConfig>,
    /// Network config
    pub network: Option<NetworkConfig>,
}

impl Config {
    /// Create a new config with given path
    pub fn new(path: &str) -> color_eyre::Result<Config> {
        Ok(Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("COCIRCOM_"))
            .extract()?)
    }
}

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
    config: Config,
) -> color_eyre::Result<SharedWitness<Rep3Protocol<P::ScalarField, Rep3MpcNet>, P>> {
    let circuit_path = PathBuf::from(&circuit);
    file_utils::check_file_exists(&circuit_path)?;

    let network_config = config.network.context("expected a network config")?;
    let compiler_config = config.compiler.context("expected a network config")?;
    let vm_config = config.vm.context("expected a vm config")?;

    // parse circuit file & put through our compiler
    let mut builder = CompilerBuilder::<P>::new(compiler_config, circuit);
    for lib in link_library {
        builder = builder.link_library(lib);
    }
    let parsed_circom_circuit = builder
        .build()
        .parse()
        .context("while parsing circuit file")?;

    // connect to network
    let net = Rep3MpcNet::new(network_config).context("while connecting to network")?;

    // init MPC protocol
    let rep3_vm = parsed_circom_circuit
        .to_rep3_vm_with_network(net, vm_config)
        .context("while constructing MPC VM")?;

    // execute witness generation in MPC
    let result_witness_share = rep3_vm
        .run(input_share)
        .context("while running witness generation")?;
    Ok(result_witness_share.into_shared_witness())
}

/// Invoke the MPC proof generation process. It will return a [Proof] if successful.
/// It executes several steps:
/// 1. Construct a [Rep3Protocol] from the network configuration.
/// 2. Construct a [CollaborativeGroth16] prover from the protocol.
/// 3. Execute the proof in MPC
pub fn prove_with_matrices_rep3<P: Pairing + CircomArkworksPairingBridge>(
    witness_share: SharedWitness<Rep3Protocol<P::ScalarField, Rep3MpcNet>, P>,
    config: NetworkConfig,
    zkey: ZKey<P>,
) -> color_eyre::Result<Proof<P>>
where
    P::ScalarField: FFTPostProcessing + CircomArkworksPrimeFieldBridge,
    P::BaseField: CircomArkworksPrimeFieldBridge,
{
    let (pk, matrices) = zkey.split();
    tracing::info!("establishing network....");
    // connect to network
    let net = Rep3MpcNet::new(config)?;
    tracing::info!("done!");
    // init MPC protocol
    tracing::info!("building protocol...");
    let protocol = Rep3Protocol::<P::ScalarField, _>::new(net)?;
    tracing::info!("done!");
    let mut prover = CollaborativeGroth16::<Rep3Protocol<P::ScalarField, _>, P>::new(protocol);
    tracing::info!("starting prover...");
    // execute prover in MPC
    prover.prove_with_matrices(&pk, &matrices, witness_share)
}
