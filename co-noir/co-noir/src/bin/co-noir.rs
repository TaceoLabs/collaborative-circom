use acir::native_types::WitnessMap;
use ark_bn254::Bn254;
use ark_ff::Zero;
use clap::{Parser, Subcommand};
use co_acvm::solver::Rep3CoSolver;
use co_noir::{
    convert_witness_to_vec_rep3, file_utils, share_input_rep3, share_rep3, share_shamir,
    translate_witness_share_rep3, GenerateProofCli, GenerateProofConfig, GenerateWitnessCli,
    GenerateWitnessConfig, MPCProtocol, PubShared, SplitInputCli, SplitInputConfig,
    SplitWitnessCli, SplitWitnessConfig, TranslateWitnessCli, TranslateWitnessConfig, VerifyCli,
    VerifyConfig,
};
use co_ultrahonk::prelude::{
    CoUltraHonk, HonkProof, ProvingKey, Rep3CoBuilder, ShamirCoBuilder, SharedBuilderVariable,
    UltraHonk, Utils, VerifyingKey,
};
use color_eyre::eyre::{eyre, Context, ContextCompat};
use mpc_core::protocols::{
    rep3::{
        fieldshare::Rep3PrimeFieldShareVec,
        network::{Rep3MpcNet, Rep3Network},
        Rep3PrimeFieldShare, Rep3Protocol,
    },
    shamir::{
        fieldshare::ShamirPrimeFieldShareVec,
        network::{ShamirMpcNet, ShamirNetwork},
        ShamirProtocol,
    },
};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};
use tracing::instrument;

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
    /// Splits an existing witness file generated by noir into secret shares for use in MPC
    SplitWitness(SplitWitnessCli),
    /// Splits a input toml file into secret shares for use in MPC
    SplitInput(SplitInputCli),
    /// Evaluates the extended witness generation for the specified circuit and input share in MPC
    GenerateWitness(GenerateWitnessCli),
    /// Translates the witness generated with one MPC protocol to a witness for a different one
    TranslateWitness(TranslateWitnessCli),
    /// Evaluates the prover algorithm for the specified circuit and witness share in MPC
    GenerateProof(GenerateProofCli),
    /// Verification of a Noir proof.
    Verify(VerifyCli),
}

fn main() -> color_eyre::Result<ExitCode> {
    install_tracing();
    let args = Cli::parse();

    match args.command {
        Commands::SplitWitness(cli) => {
            let config = SplitWitnessConfig::parse(cli).context("while parsing config")?;
            run_split_witness(config)
        }
        Commands::SplitInput(cli) => {
            let config = SplitInputConfig::parse(cli).context("while parsing config")?;
            run_split_input(config)
        }
        Commands::GenerateWitness(cli) => {
            let config = GenerateWitnessConfig::parse(cli).context("while parsing config")?;
            run_generate_witness(config)
        }
        Commands::TranslateWitness(cli) => {
            let config = TranslateWitnessConfig::parse(cli).context("while parsing config")?;
            run_translate_witness(config)
        }
        Commands::GenerateProof(cli) => {
            let config = GenerateProofConfig::parse(cli).context("while parsing config")?;
            run_generate_proof(config)
        }
        Commands::Verify(cli) => {
            let config = VerifyConfig::parse(cli).context("while parsing config")?;
            run_verify(config)
        }
    }
}

#[instrument(skip(config))]
fn run_split_witness(config: SplitWitnessConfig) -> color_eyre::Result<ExitCode> {
    let witness_path = config.witness;
    let circuit_path = config.circuit;
    let protocol = config.protocol;
    let out_dir = config.out_dir;
    let t = config.threshold;
    let n = config.num_parties;

    file_utils::check_file_exists(&witness_path)?;
    file_utils::check_file_exists(&circuit_path)?;
    file_utils::check_dir_exists(&out_dir)?;

    // parse constraint system
    let constraint_system = Utils::get_constraint_system_from_file(&circuit_path, true)
        .context("while parsing program artifact")?;
    let pub_inputs = constraint_system.public_inputs;

    // parse witness
    let witness = Utils::get_witness_from_file(&witness_path).context("while parsing witness")?;

    // create witness map storing pub/private information
    let mut witness = witness
        .into_iter()
        .map(PubShared::from_shared)
        .collect::<Vec<_>>();
    for index in pub_inputs {
        let index = index as usize;
        if index >= witness.len() {
            return Err(eyre!("Public input index out of bounds"));
        }
        PubShared::set_public(&mut witness[index]);
    }

    let mut rng = rand::thread_rng();

    match protocol {
        MPCProtocol::REP3 => {
            if t != 1 {
                return Err(eyre!("REP3 only allows the threshold to be 1"));
            }
            if n != 3 {
                return Err(eyre!("REP3 only allows the number of parties to be 3"));
            }
            // create witness shares
            let start = Instant::now();
            let shares = share_rep3::<Bn254, Rep3MpcNet, _>(witness, &mut rng);
            let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
            tracing::info!("Sharing took {} ms", duration_ms);

            // write out the shares to the output directory
            let base_name = witness_path
                .file_name()
                .context("we have a file name")?
                .to_str()
                .context("witness file name is not valid UTF-8")?;
            for (i, share) in shares.iter().enumerate() {
                let path = out_dir.join(format!("{}.{}.shared", base_name, i));
                let out_file =
                    BufWriter::new(File::create(&path).context("while creating output file")?);
                bincode::serialize_into(out_file, share)
                    .context("while serializing witness share")?;
                tracing::info!("Wrote witness share {} to file {}", i, path.display());
            }
        }
        MPCProtocol::SHAMIR => {
            // create witness shares
            let start = Instant::now();
            let shares = share_shamir::<Bn254, ShamirMpcNet, _>(witness, t, n, &mut rng);
            let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
            tracing::info!("Sharing took {} ms", duration_ms);

            // write out the shares to the output directory
            let base_name = witness_path
                .file_name()
                .context("we have a file name")?
                .to_str()
                .context("witness file name is not valid UTF-8")?;
            for (i, share) in shares.iter().enumerate() {
                let path = out_dir.join(format!("{}.{}.shared", base_name, i));
                let out_file =
                    BufWriter::new(File::create(&path).context("while creating output file")?);
                bincode::serialize_into(out_file, share)
                    .context("while serializing witness share")?;
                tracing::info!("Wrote witness share {} to file {}", i, path.display());
            }
        }
    }
    tracing::info!("Split witness into shares successfully");
    Ok(ExitCode::SUCCESS)
}

#[instrument(skip(config))]
fn run_split_input(config: SplitInputConfig) -> color_eyre::Result<ExitCode> {
    let input = config.input;
    let circuit = config.circuit;
    let protocol = config.protocol;
    let out_dir = config.out_dir;

    if protocol != MPCProtocol::REP3 {
        return Err(eyre!(
            "Only REP3 protocol is supported for splitting inputs"
        ));
    }
    file_utils::check_file_exists(&input)?;
    let circuit_path = PathBuf::from(&circuit);
    file_utils::check_file_exists(&circuit_path)?;
    file_utils::check_dir_exists(&out_dir)?;

    // parse constraint system
    let compiled_program = Utils::get_program_artifact_from_file(&circuit_path)
        .context("while parsing program artifact")?;

    // read the input file
    let inputs =
        Rep3CoSolver::<_, Rep3MpcNet>::read_abi_bn254_fieldelement(&input, &compiled_program.abi)?;

    // create input shares
    let mut rng = rand::thread_rng();
    let start = Instant::now();
    let shares = share_input_rep3::<Bn254, Rep3MpcNet, _>(inputs, &mut rng);
    let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
    tracing::info!("Sharing took {} ms", duration_ms);

    // write out the shares to the output directory
    let base_name = input
        .file_name()
        .context("we have a file name")?
        .to_str()
        .context("input file name is not valid UTF-8")?;
    for (i, share) in shares.iter().enumerate() {
        let path = out_dir.join(format!("{}.{}.shared", base_name, i));
        let out_file = BufWriter::new(File::create(&path).context("while creating output file")?);
        bincode::serialize_into(out_file, share).context("while serializing witness share")?;
        tracing::info!("Wrote input share {} to file {}", i, path.display());
    }

    tracing::info!("Split input into shares successfully");
    Ok(ExitCode::SUCCESS)
}

#[instrument(skip(config))]
fn run_generate_witness(config: GenerateWitnessConfig) -> color_eyre::Result<ExitCode> {
    let input = config.input;
    let circuit = config.circuit;
    let protocol = config.protocol;
    let out = config.out;

    if protocol != MPCProtocol::REP3 {
        return Err(eyre!(
            "Only REP3 protocol is supported for merging input shares"
        ));
    }
    file_utils::check_file_exists(&input)?;
    let circuit_path = PathBuf::from(&circuit);
    file_utils::check_file_exists(&circuit_path)?;

    // parse constraint system
    let compiled_program = Utils::get_program_artifact_from_file(&circuit_path)
        .context("while parsing program artifact")?;

    // parse input shares
    let input_share_file =
        BufReader::new(File::open(&input).context("while opening input share file")?);
    let input_share: WitnessMap<Rep3PrimeFieldShare<ark_bn254::Fr>> =
        bincode::deserialize_from(input_share_file).context("while deserializing input share")?;
    let input_share = translate_witness_share_rep3(input_share);

    // connect to network
    let net = Rep3MpcNet::new(config.network).context("while connecting to network")?;
    let id = usize::from(net.get_id());

    // init MPC protocol
    let rep3_vm = Rep3CoSolver::from_network_with_witness(net, compiled_program, input_share)
        .context("while creating VM")?;

    // execute witness generation in MPC
    let start = Instant::now();
    let result_witness_share = rep3_vm
        .solve()
        .context("while running witness generation")?;
    let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
    tracing::info!("Party {}: Witness extension took {} ms", id, duration_ms);

    let result_witness_share =
        convert_witness_to_vec_rep3::<Bn254, Rep3MpcNet>(result_witness_share);

    // write result to output file
    let out_file = BufWriter::new(std::fs::File::create(&out)?);
    bincode::serialize_into(out_file, &result_witness_share)
        .context("while serializing witness share")?;
    tracing::info!("Witness successfully written to {}", out.display());
    Ok(ExitCode::SUCCESS)
}

#[instrument(skip(config))]
fn run_translate_witness(config: TranslateWitnessConfig) -> color_eyre::Result<ExitCode> {
    let witness = config.witness;
    let src_protocol = config.src_protocol;
    let target_protocol = config.target_protocol;
    let out = config.out;

    if src_protocol != MPCProtocol::REP3 || target_protocol != MPCProtocol::SHAMIR {
        return Err(eyre!("Only REP3 to SHAMIR translation is supported"));
    }
    file_utils::check_file_exists(&witness)?;

    // parse witness shares
    let witness_file =
        BufReader::new(File::open(witness).context("trying to open witness share file")?);
    let witness_share: Vec<SharedBuilderVariable<Rep3Protocol<ark_bn254::Fr, Rep3MpcNet>, Bn254>> =
        bincode::deserialize_from(witness_file).context("while deserializing witness share")?;

    // extract shares only
    let mut shares = vec![];
    for share in witness_share.iter() {
        if let SharedBuilderVariable::Shared(value) = share {
            shares.push(value.to_owned());
        }
    }

    // connect to network
    let net = Rep3MpcNet::new(config.network)?;
    let id = usize::from(net.get_id());

    // init MPC protocol
    let protocol = Rep3Protocol::<ark_bn254::Fr, Rep3MpcNet>::new(net)?;
    let mut protocol = protocol.get_shamir_protocol()?;

    // Translate witness to shamir shares
    let start = Instant::now();
    let result_shares: ShamirPrimeFieldShareVec<ark_bn254::Fr> =
        protocol.translate_primefield_repshare_vec(Rep3PrimeFieldShareVec::from(shares))?;
    let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
    tracing::info!("Party {}: Translating witness took {} ms", id, duration_ms);

    let mut result: Vec<SharedBuilderVariable<ShamirProtocol<ark_bn254::Fr, ShamirMpcNet>, Bn254>> =
        Vec::with_capacity(witness_share.len());
    let mut iter = result_shares.into_iter();
    for val in witness_share.into_iter() {
        match val {
            SharedBuilderVariable::Public(value) => {
                result.push(SharedBuilderVariable::Public(value))
            }
            SharedBuilderVariable::Shared(_) => {
                let share = iter.next().expect("enough shares");
                result.push(SharedBuilderVariable::Shared(share))
            }
        }
    }

    // write result to output file
    let out_file = BufWriter::new(std::fs::File::create(&out)?);
    bincode::serialize_into(out_file, &result)?;
    tracing::info!("Witness successfully written to {}", out.display());
    Ok(ExitCode::SUCCESS)
}

#[instrument(skip(config))]
fn run_generate_proof(config: GenerateProofConfig) -> color_eyre::Result<ExitCode> {
    let witness = config.witness;
    let circuit_path = config.circuit;
    let crs_path = config.crs;
    let protocol = config.protocol;
    let out = config.out;
    let public_input_filename = config.public_input;
    let t = config.threshold;

    file_utils::check_file_exists(&witness)?;
    file_utils::check_file_exists(&circuit_path)?;
    file_utils::check_file_exists(&crs_path)?;

    // parse witness shares
    let witness_file =
        BufReader::new(File::open(witness).context("trying to open witness share file")?);

    // parse constraint system
    let constraint_system = Utils::get_constraint_system_from_file(&circuit_path, true)
        .context("while parsing program artifact")?;

    let (proof, public_input) = match protocol {
        MPCProtocol::REP3 => {
            if t != 1 {
                return Err(eyre!("REP3 only allows the threshold to be 1"));
            }
            let witness_share = bincode::deserialize_from(witness_file)
                .context("while deserializing witness share")?;
            // connect to network
            let net = Rep3MpcNet::new(config.network)?;
            let id = usize::from(net.get_id());

            // init MPC protocol
            let protocol = Rep3Protocol::new(net)?;

            // Create the circuit
            tracing::info!("Party {}: starting to generate proving key..", id);
            let start = Instant::now();
            let builder = Rep3CoBuilder::<Bn254, _>::create_circuit(
                constraint_system,
                0,
                witness_share,
                true,
                false,
            );

            // parse the crs
            let prover_crs = ProvingKey::get_prover_crs(
                &builder,
                crs_path.to_str().context("while opening crs file")?,
            )
            .expect("failed to get prover crs");

            // Get the proving key and prover
            let proving_key = ProvingKey::create(&protocol, builder, prover_crs);
            let public_input = proving_key.get_public_inputs();
            let prover = CoUltraHonk::new(protocol);
            let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
            tracing::info!(
                "Party {}: Proving key generation took {} ms",
                id,
                duration_ms
            );

            // execute prover in MPC
            tracing::info!("Party {}: starting proof generation..", id);
            let start = Instant::now();
            let proof = prover.prove(proving_key)?;
            let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
            tracing::info!("Party {}: Proof generation took {} ms", id, duration_ms);

            (proof, public_input)
        }
        MPCProtocol::SHAMIR => {
            let witness_share = bincode::deserialize_from(witness_file)
                .context("while deserializing witness share")?;
            // connect to network
            let net = ShamirMpcNet::new(config.network)?;
            let id = net.get_id();

            // init MPC protocol
            let protocol = ShamirProtocol::new(t, net)?;

            // Create the circuit
            tracing::info!("Party {}: starting to generate proving key..", id);
            let start = Instant::now();
            let builder = ShamirCoBuilder::<Bn254, _>::create_circuit(
                constraint_system,
                0,
                witness_share,
                true,
                false,
            );

            // parse the crs
            let prover_crs = ProvingKey::get_prover_crs(
                &builder,
                crs_path.to_str().context("while opening crs file")?,
            )
            .expect("failed to get prover crs");

            // Get the proving key and prover
            let proving_key = ProvingKey::create(&protocol, builder, prover_crs);
            let public_input = proving_key.get_public_inputs();
            let prover = CoUltraHonk::new(protocol);
            let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
            tracing::info!(
                "Party {}: Proving key generation took {} ms",
                id,
                duration_ms
            );

            // execute prover in MPC
            tracing::info!("Party {}: starting proof generation..", id);
            let start = Instant::now();
            let proof = prover.prove(proving_key)?;
            let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
            tracing::info!("Party {}: Proof generation took {} ms", id, duration_ms);

            (proof, public_input)
        }
    };

    // write result to output file
    if let Some(out) = out {
        let mut out_file =
            BufWriter::new(std::fs::File::create(&out).context("while creating output file")?);

        let proof_u8 = proof.to_buffer();
        out_file
            .write(proof_u8.as_slice())
            .context("while writing proof to file")?;
        tracing::info!("Wrote proof to file {}", out.display());
    }

    // write public input to output file
    if let Some(public_input_filename) = public_input_filename {
        let public_input_as_strings = public_input
            .iter()
            .map(|f| {
                if f.is_zero() {
                    "0".to_string()
                } else {
                    f.to_string()
                }
            })
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

    tracing::info!("Proof generation finished successfully");
    Ok(ExitCode::SUCCESS)
}

#[instrument(skip(config))]
fn run_verify(config: VerifyConfig) -> color_eyre::Result<ExitCode> {
    let proof = config.proof;
    let vk = config.vk;
    let crs_path = config.crs;

    file_utils::check_file_exists(&proof)?;
    file_utils::check_file_exists(&vk)?;
    file_utils::check_file_exists(&crs_path)?;

    // parse proof file
    let proof_u8 = std::fs::read(&proof).context("while reading proof file")?;
    let proof = HonkProof::from_buffer(&proof_u8).context("while deserializing proof")?;

    // parse the crs
    let crs = VerifyingKey::get_verifier_crs(crs_path.to_str().context("while opening crs file")?)
        .expect("failed to get verifier crs");

    // parse verification key file
    // let vk_file = BufReader::new(File::open(&vk).context("while opening verification key file")?);

    // The actual verifier
    let start = Instant::now();
    let res = UltraHonk::verify(proof, verifying_key).context("while verifying proof")?;
    let duration_ms = start.elapsed().as_micros() as f64 / 1000.;
    tracing::info!("Proof verification took {} ms", duration_ms);

    if res {
        tracing::info!("Proof verified successfully");
        Ok(ExitCode::SUCCESS)
    } else {
        tracing::error!("Proof verification failed");
        Ok(ExitCode::FAILURE)
    }
}
