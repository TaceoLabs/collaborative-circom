use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
    process::ExitCode,
};

use ark_bn254::Bn254;
use ark_groth16::{Groth16, Proof};
use circom_types::{
    groth16::{
        proof::JsonProof, verification_key::JsonVerificationKey, witness::Witness, zkey::ZKey,
    },
    r1cs::R1CS,
};
use clap::{Parser, Subcommand};
use collaborative_circom::file_utils;
use collaborative_groth16::groth16::{CollaborativeGroth16, SharedWitness};
use color_eyre::eyre::{eyre, Context};
use mpc_core::protocols::aby3::{network::Aby3MpcNet, Aby3Protocol};
use mpc_net::config::NetworkConfig;

fn install_tracing() {
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    let fmt_layer = fmt::layer().with_target(true).with_line_number(true);
    let filter_layer = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Splits an existing witness file generated by Circom into secret shares for use in MPC
    SplitWitness {
        /// The path to the input witness file generated by Circom
        #[arg(long)]
        witness: PathBuf,
        /// The path to the r1cs file, generated by Circom compiler
        #[arg(long)]
        r1cs: PathBuf,
        /// The MPC protocol to be used
        #[arg(long)]
        protocol: String, // TODO: which datatype? an enum?
        /// The path to the (existing) output directory
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Splits a JSON input file into secret shares for use in MPC
    SplitInput {
        /// The path to the input JSON file
        #[arg(long)]
        input: PathBuf,
        /// The MPC protocol to be used
        #[arg(long)]
        protocol: String, // TODO: which datatype? an enum?
        /// The path to the (existing) output directory
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Evaluates the extended witness generation for the specified circuit and input share in MPC
    GenerateWitness {
        /// The path to the input share file
        #[arg(long)]
        input: PathBuf,
        /// The path to the circuit file
        #[arg(long)]
        circuit: PathBuf,
        /// The MPC protocol to be used
        #[arg(long)]
        protocol: String, // TODO: which datatype? an enum?
        /// The path to MPC network configuration file
        #[arg(long)]
        config: PathBuf,
        /// The output file where the final witness share is written to
        #[arg(long)]
        out: PathBuf,
    },
    /// Evaluates the prover algorithm for the specified circuit and witness share in MPC
    GenerateProof {
        /// The path to the witness share file
        #[arg(long)]
        witness: PathBuf,
        /// The path to the r1cs file, generated by Circom compiler
        #[arg(long)]
        r1cs: PathBuf,
        /// The path to the proving key (.zkey) file, generated by snarkjs setup phase
        #[arg(long)]
        zkey: PathBuf,
        /// The MPC protocol to be used
        #[arg(long)]
        protocol: String, // TODO: which datatype? an enum?
        /// The path to MPC network configuration file
        #[arg(long)]
        config: PathBuf,
        /// The output file where the final proof is written to. If not passed, this party will not write the proof to a file.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Verification of a Circom proof.
    Verify {
        /// The path to the proof file
        #[arg(long)]
        proof: PathBuf,
        /// The path to the verification key file
        #[arg(long)]
        vk: PathBuf,
        /// The path to the public inputs file
        #[arg(long)]
        public_inputs: PathBuf,
    },
}

fn main() -> color_eyre::Result<ExitCode> {
    install_tracing();
    let args = Cli::parse();

    match args.command {
        Commands::SplitWitness {
            witness: witness_path,
            r1cs,
            protocol: _,
            out_dir,
        } => {
            file_utils::check_file_exists(&witness_path)?;
            file_utils::check_file_exists(&r1cs)?;
            file_utils::check_dir_exists(&out_dir)?;

            // TODO: make generic over curve/protocol

            // read the Circom witness file
            let witness_file =
                BufReader::new(File::open(&witness_path).context("while opening witness file")?);
            let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file)
                .context("while parsing witness file")?;

            // read the Circom r1cs file
            let r1cs_file = BufReader::new(File::open(&r1cs).context("while opening r1cs file")?);
            let r1cs = R1CS::<ark_bn254::Bn254>::from_reader(r1cs_file)
                .context("while parsing r1cs file")?;

            let mut rng = rand::thread_rng();

            // create witness shares
            let shares =
                SharedWitness::<Aby3Protocol<ark_bn254::Fr, Aby3MpcNet>, Bn254>::share_aby3(
                    &witness.values[r1cs.num_inputs..],
                    &witness.values[..r1cs.num_inputs],
                    &mut rng,
                );

            // write out the shares to the output directory
            let base_name = witness_path
                .file_name()
                .expect("we have a file name")
                .to_str()
                .ok_or(eyre!("input file name is not valid UTF-8"))?;
            for (i, share) in shares.iter().enumerate() {
                let path = out_dir.join(format!("{}.{}.shared", base_name, i));
                let out_file =
                    BufWriter::new(File::create(&path).context("while creating output file")?);
                bincode::serialize_into(out_file, share)
                    .context("while serializing witness share")?;
                tracing::info!("Wrote witness share {} to file {}", i, path.display());
            }
            tracing::info!("Split witness into shares successfully")
        }
        Commands::SplitInput {
            input,
            protocol: _,
            out_dir,
        } => {
            file_utils::check_file_exists(&input)?;
            file_utils::check_dir_exists(&out_dir)?;

            // read the input file

            // construct relevant protocol

            // create input shares

            // write out the shares to the output directory
        }
        Commands::GenerateWitness {
            input,
            circuit,
            protocol: _,
            config,
            out: _,
        } => {
            file_utils::check_file_exists(&input)?;
            file_utils::check_file_exists(&circuit)?;
            file_utils::check_file_exists(&config)?;

            // parse input shares

            // parse circuit file & put through our compiler

            // parse network configuration
            let config = std::fs::read_to_string(config)?;
            let _config: NetworkConfig = toml::from_str(&config)?;

            // construct relevant protocol

            // connect to network

            // execute witness generation in MPC

            // write result to output file
            // let witness_share: SharedWitness<Aby3Protocol<ark_bn254::Fr, Aby3MpcNet>, Bn254> =
            //     todo!();
            // let out_file = BufWriter::new(std::fs::File::create(out)?);
            // bincode::serialize_into(out_file, &witness_share)?;
            tracing::info!("Witness generation finished successfully")
        }
        Commands::GenerateProof {
            witness,
            r1cs,
            zkey,
            protocol: _,
            config,
            out,
        } => {
            file_utils::check_file_exists(&witness)?;
            file_utils::check_file_exists(&r1cs)?;
            file_utils::check_file_exists(&zkey)?;
            file_utils::check_file_exists(&config)?;

            // parse witness shares
            let witness_file =
                BufReader::new(File::open(witness).context("trying to open witness share file")?);

            // TODO: how to best allow for different MPC protocols here
            let witness_share: SharedWitness<Aby3Protocol<ark_bn254::Fr, Aby3MpcNet>, Bn254> =
                bincode::deserialize_from(witness_file)
                    .context("trying to parse witness share file")?;

            // parse public inputs
            // TODO: decision: ATM 1 is still in the public inputs, should we remove it?
            let public_input = witness_share.public_inputs.clone();

            // parse Circom r1cs file
            let r1cs_file = BufReader::new(File::open(r1cs).context("trying to open R1CS file")?);
            // TODO: allow different curves: move all of this into a generic function and match on curve...
            let r1cs =
                R1CS::<Bn254>::from_reader(r1cs_file).context("trying to parse R1CS file")?;

            // parse Circom zkey file
            let zkey_file = File::open(zkey)?;
            let (pk, _) = ZKey::<Bn254>::from_reader(zkey_file).unwrap().split();

            // parse network configuration
            let config = std::fs::read_to_string(config)?;
            let config: NetworkConfig = toml::from_str(&config)?;

            // connect to network
            let net = Aby3MpcNet::new(config)?;

            // init MPC protocol
            let protocol = Aby3Protocol::<ark_bn254::Fr, _>::new(net)?;
            let mut prover =
                CollaborativeGroth16::<Aby3Protocol<ark_bn254::Fr, _>, Bn254>::new(protocol);

            // execute prover in MPC
            let proof = prover.prove(&pk, &r1cs, &public_input, witness_share)?;

            // write result to output file
            if let Some(out) = out {
                let out_file = BufWriter::new(
                    std::fs::File::create(&out).context("while opening output file")?,
                );

                serde_json::to_writer(out_file, &JsonProof::<Bn254>::from(proof))
                    .context("while serializing proof to JSON file")?;
                tracing::info!("Wrote proof to file {}", out.display());
            }
            tracing::info!("Proof generation finished successfully")
        }
        Commands::Verify {
            proof,
            vk,
            public_inputs,
        } => {
            file_utils::check_file_exists(&proof)?;
            file_utils::check_file_exists(&vk)?;
            file_utils::check_file_exists(&public_inputs)?;

            // parse Circom proof file
            let proof_file =
                BufReader::new(File::open(&proof).context("while opening proof file")?);
            let proof: JsonProof<Bn254> = serde_json::from_reader(proof_file)
                .context("while deserializing proof from file")?;
            let proof = Proof::<Bn254>::from(proof);

            // parse Circom verification key file
            let vk_file =
                BufReader::new(File::open(&vk).context("while opening verification key file")?);
            let vk: JsonVerificationKey<Bn254> = serde_json::from_reader(vk_file)
                .context("while deserializing verification key from file")?;
            let vk: ark_groth16::PreparedVerifyingKey<Bn254> = vk.into();

            // parse public inputs
            let public_inputs_file = BufReader::new(
                File::open(&public_inputs).context("while opening public inputs file")?,
            );
            // TODO: real parsing of public inputs file
            let witness_share: SharedWitness<Aby3Protocol<ark_bn254::Fr, Aby3MpcNet>, Bn254> =
                bincode::deserialize_from(public_inputs_file)
                    .context("trying to parse witness share file")?;
            // skip 1 atm
            let public_inputs = &witness_share.public_inputs[1..];

            // verify proof
            if Groth16::<Bn254>::verify_proof(&vk, &proof, public_inputs)
                .context("while verifying proof")?
            {
                tracing::info!("Proof verified successfully");
            } else {
                tracing::error!("Proof verification failed");
                return Ok(ExitCode::FAILURE);
            }
        }
    }

    Ok(ExitCode::SUCCESS)
}
