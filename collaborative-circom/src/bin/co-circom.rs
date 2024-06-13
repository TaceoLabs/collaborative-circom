use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
    process::ExitCode,
};

use ark_bn254::Bn254;
use ark_groth16::{Groth16, Proof};
use circom_mpc_compiler::CompilerBuilder;
use circom_types::{
    groth16::{
        proof::JsonProof, verification_key::JsonVerificationKey, witness::Witness, zkey::ZKey,
    },
    r1cs::R1CS,
};
use clap::{Parser, Subcommand};
use collaborative_circom::file_utils;
use collaborative_groth16::groth16::{CollaborativeGroth16, SharedInput, SharedWitness};
use color_eyre::eyre::{eyre, Context};
use mpc_core::protocols::rep3::{self, network::Rep3MpcNet, Rep3Protocol};
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
        circuit: String,
        /// The path to Circom library files
        #[arg(long)]
        link_library: Vec<String>,
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
        /// The output JSON file where the public inputs are written to. If not passed, this party will not write the public inputs to a file.
        #[arg(long)]
        public_input: Option<PathBuf>,
    },
    /// Verification of a Circom proof.
    Verify {
        /// The path to the proof file
        #[arg(long)]
        proof: PathBuf,
        /// The path to the verification key file
        #[arg(long)]
        vk: PathBuf,
        /// The path to the public input JSON file
        #[arg(long)]
        public_input: PathBuf,
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
                SharedWitness::<Rep3Protocol<ark_bn254::Fr, Rep3MpcNet>, Bn254>::share_rep3(
                    &witness.values[r1cs.num_inputs..],
                    &witness.values[..r1cs.num_inputs],
                    &mut rng,
                );

            // write out the shares to the output directory
            let base_name = witness_path
                .file_name()
                .expect("we have a file name")
                .to_str()
                .ok_or(eyre!("witness file name is not valid UTF-8"))?;
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
            let input_file =
                BufReader::new(File::open(&input).context("while opening input file")?);

            let input_json: serde_json::Map<String, serde_json::Value> =
                serde_json::from_reader(input_file).context("while parsing input file")?;

            // construct relevant protocol
            // TODO: make generic over curve/protocol

            // create input shares
            let mut shares = [
                SharedInput::<Rep3Protocol<ark_bn254::Fr, Rep3MpcNet>, Bn254>::default(),
                SharedInput::<Rep3Protocol<ark_bn254::Fr, Rep3MpcNet>, Bn254>::default(),
                SharedInput::<Rep3Protocol<ark_bn254::Fr, Rep3MpcNet>, Bn254>::default(),
            ];

            let mut rng = rand::thread_rng();
            for (name, val) in input_json {
                let parsed_vals = if val.is_array() {
                    parse_array(&val)?
                } else {
                    vec![parse_field(&val)?]
                };

                let [share0, share1, share2] =
                    rep3::utils::share_field_elements(&parsed_vals, &mut rng);
                shares[0].shared_inputs.insert(name.clone(), share0);
                shares[1].shared_inputs.insert(name.clone(), share1);
                shares[2].shared_inputs.insert(name.clone(), share2);
            }

            // write out the shares to the output directory
            let base_name = input
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
                tracing::info!("Wrote input share {} to file {}", i, path.display());
            }
            tracing::info!("Split input into shares successfully")
        }
        Commands::GenerateWitness {
            input,
            circuit,
            link_library,
            protocol: _,
            config,
            out,
        } => {
            file_utils::check_file_exists(&input)?;
            let circuit_path = PathBuf::from(&circuit);
            file_utils::check_file_exists(&circuit_path)?;
            file_utils::check_file_exists(&config)?;

            // parse input shares
            let input_share_file =
                BufReader::new(File::open(&input).context("while opening input share file")?);
            let input_share: SharedInput<Rep3Protocol<ark_bn254::Fr, Rep3MpcNet>, Bn254> =
                bincode::deserialize_from(input_share_file)
                    .context("trying to parse input share file")?;

            // parse circuit file & put through our compiler
            let mut builder = CompilerBuilder::<Bn254>::new(circuit);
            for lib in link_library {
                builder = builder.link_library(lib);
            }
            let parsed_circom_circuit = builder
                .build()
                .parse()
                .context("while parsing circuit file")?;

            // parse network configuration
            let config =
                std::fs::read_to_string(config).context("while reading network config file")?;
            let config: NetworkConfig =
                toml::from_str(&config).context("while parsing network config")?;

            // connect to network
            let net = Rep3MpcNet::new(config).context("while connecting to network")?;

            // init MPC protocol
            let rep3_vm = parsed_circom_circuit
                .to_rep3_vm_with_network(net)
                .context("while constructing MPC VM")?;

            // execute witness generation in MPC
            let result_witness_share = rep3_vm
                .run(input_share)
                .context("while running witness generation")?;

            // write result to output file
            let out_file = BufWriter::new(std::fs::File::create(&out)?);
            bincode::serialize_into(out_file, &result_witness_share)?;
            tracing::info!("Witness successfully written to {}", out.display());
        }
        Commands::GenerateProof {
            witness,
            zkey,
            protocol: _,
            config,
            out,
            public_input: public_input_filename,
        } => {
            file_utils::check_file_exists(&witness)?;
            file_utils::check_file_exists(&zkey)?;
            file_utils::check_file_exists(&config)?;

            // parse witness shares
            let witness_file =
                BufReader::new(File::open(witness).context("trying to open witness share file")?);

            // TODO: how to best allow for different MPC protocols here
            let witness_share: SharedWitness<Rep3Protocol<ark_bn254::Fr, Rep3MpcNet>, Bn254> =
                bincode::deserialize_from(witness_file)
                    .context("trying to parse witness share file")?;

            // parse public inputs
            let public_input = witness_share.public_inputs.clone();

            // parse Circom zkey file
            let zkey_file = File::open(zkey)?;
            let (pk, matrices) = ZKey::<Bn254>::from_reader(zkey_file).unwrap().split();

            // parse network configuration
            let config = std::fs::read_to_string(config)?;
            let config: NetworkConfig = toml::from_str(&config)?;

            // connect to network
            let net = Rep3MpcNet::new(config)?;

            // init MPC protocol
            let protocol = Rep3Protocol::<ark_bn254::Fr, _>::new(net)?;
            let mut prover =
                CollaborativeGroth16::<Rep3Protocol<ark_bn254::Fr, _>, Bn254>::new(protocol);

            // execute prover in MPC
            let proof = prover.prove_with_matrices(&pk, &matrices, &public_input, witness_share)?;

            // write result to output file
            if let Some(out) = out {
                let out_file = BufWriter::new(
                    std::fs::File::create(&out).context("while creating output file")?,
                );

                serde_json::to_writer(out_file, &JsonProof::<Bn254>::from(proof))
                    .context("while serializing proof to JSON file")?;
                tracing::info!("Wrote proof to file {}", out.display());
            }
            // write public input to output file
            if let Some(public_input_filename) = public_input_filename {
                let public_input_as_strings = public_input
                    .iter()
                    .skip(1) // we skip the constant 1 at position 0
                    .map(|f| f.to_string())
                    .collect::<Vec<String>>();
                let public_input_file = BufWriter::new(
                    std::fs::File::create(&public_input_filename)
                        .context("while creating public input file")?,
                );
                serde_json::to_writer(public_input_file, &public_input_as_strings)
                    .context("while writing out public inputs to JSON file")?;
                tracing::info!(
                    "Wrote public inputs to file {}",
                    public_input_filename.display()
                );
            }
            tracing::info!("Proof generation finished successfully")
        }
        Commands::Verify {
            proof,
            vk,
            public_input,
        } => {
            file_utils::check_file_exists(&proof)?;
            file_utils::check_file_exists(&vk)?;
            file_utils::check_file_exists(&public_input)?;

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
                File::open(&public_input).context("while opening public inputs file")?,
            );
            let public_inputs_as_strings: Vec<String> = serde_json::from_reader(public_inputs_file)
                .context("while parsing public inputs, expect them to be array of stringified field elements")?;
            // skip 1 atm
            let public_inputs = public_inputs_as_strings
                .into_iter()
                .map(|s| {
                    s.parse::<ark_bn254::Fr>()
                        .map_err(|_| eyre!("could not parse as field element: {}", s))
                })
                .collect::<Result<Vec<ark_bn254::Fr>, _>>()
                .context("while converting public input strings to field elements")?;

            // verify proof
            if Groth16::<Bn254>::verify_proof(&vk, &proof, &public_inputs)
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

fn parse_field(val: &serde_json::Value) -> color_eyre::Result<ark_bn254::Fr> {
    val.as_str()
        .ok_or_else(|| {
            eyre!(
                "expected input to be a field element string, got \"{}\"",
                val
            )
        })?
        .parse::<ark_bn254::Fr>()
        .map_err(|_| eyre!("could not parse field element: \"{}\"", val))
        .context("while parsing field element")
}

fn parse_array(val: &serde_json::Value) -> color_eyre::Result<Vec<ark_bn254::Fr>> {
    let json_arr = val.as_array().expect("is an array");
    let mut field_elements = vec![];
    for ele in json_arr {
        if ele.is_array() {
            field_elements.extend(parse_array(ele)?);
        } else {
            field_elements.push(parse_field(ele)?);
        }
    }
    Ok(field_elements)
}
