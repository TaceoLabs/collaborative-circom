use acir::native_types::{WitnessMap, WitnessStack};
use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_ff::Zero;
use co_acvm::{solver::Rep3CoSolver, Rep3AcvmType};
use co_ultrahonk::prelude::{
    CoUltraHonk, HonkProof, ProvingKey, Rep3CoBuilder, Rep3UltraHonkDriver, SharedBuilderVariable,
    UltraCircuitBuilder, UltraCircuitVariable, UltraHonk, Utils, VerifyingKey,
};
use mpc_core::protocols::rep3::network::{IoContext, Rep3Network};
use std::thread;
use tests::rep3_network::Rep3TestNetwork;
use tokio::runtime;

use crate::proof_tests::{CRS_PATH_G1, CRS_PATH_G2};

fn witness_map_to_witness_vector<P: Pairing, N: Rep3Network>(
    witness_map: WitnessMap<Rep3AcvmType<P::ScalarField>>,
) -> Vec<SharedBuilderVariable<Rep3UltraHonkDriver<N>, P>> {
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
            Rep3AcvmType::Public(f) => {
                wv.push(SharedBuilderVariable::from_public(f));
            }
            Rep3AcvmType::Shared(f) => {
                wv.push(SharedBuilderVariable::from_shared(f));
            }
        }
        index += 1;
    }
    wv
}

fn convert_witness_rep3<P: Pairing, N: Rep3Network>(
    mut witness_stack: WitnessStack<Rep3AcvmType<P::ScalarField>>,
) -> Vec<SharedBuilderVariable<Rep3UltraHonkDriver<N>, P>> {
    let witness_map = witness_stack
        .pop()
        .expect("Witness should be present")
        .witness;
    witness_map_to_witness_vector(witness_map)
}

fn proof_test(name: &str) {
    let circuit_file = format!("../test_vectors/noir/{}/kat/{}.json", name, name);
    let witness_file = format!("../test_vectors/noir/{}/kat/{}.gz", name, name);
    let proof_file = format!("../test_vectors/noir/{}/kat/{}.proof", name, name);

    let program_artifact = Utils::get_program_artifact_from_file(&circuit_file)
        .expect("failed to parse program artifact");
    let witness = Utils::get_witness_from_file(&witness_file).expect("failed to parse witness");

    // Will be trivially shared anyways
    let witness = witness
        .into_iter()
        .map(SharedBuilderVariable::from_public)
        .collect::<Vec<_>>();

    let test_network = Rep3TestNetwork::default();
    let mut threads = Vec::with_capacity(3);
    for net in test_network.get_party_networks() {
        let artifact = program_artifact.clone();
        let witness = witness.clone();
        threads.push(thread::spawn(move || {
            let constraint_system = Utils::get_constraint_system_from_artifact(&artifact, true);

            let builder = Rep3CoBuilder::<Bn254, _>::create_circuit(
                constraint_system,
                0,
                witness,
                true,
                false,
            );

            let crs = ProvingKey::get_prover_crs(&builder, CRS_PATH_G1)
                .expect("failed to get prover crs");

            let id = net.id;

            let runtime = runtime::Builder::new_current_thread().build().unwrap();
            let mut io_context0 = runtime.block_on(IoContext::init(net)).unwrap();
            let io_context1 = runtime.block_on(io_context0.fork()).unwrap();
            let driver = Rep3UltraHonkDriver::new(io_context0, io_context1);
            let proving_key = ProvingKey::create(id, builder, crs);

            let prover = CoUltraHonk::new(driver);
            runtime.block_on(prover.prove(proving_key)).unwrap()
        }));
    }

    let mut proofs = threads
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect::<Vec<_>>();
    let proof = proofs.pop().unwrap();
    for p in proofs {
        assert_eq!(proof, p);
    }

    let proof_u8 = proof.to_buffer();

    let read_proof_u8 = std::fs::read(&proof_file).unwrap();
    assert_eq!(proof_u8, read_proof_u8);

    let read_proof = HonkProof::from_buffer(&read_proof_u8).unwrap();
    assert_eq!(proof, read_proof);

    // Get vk
    let constraint_system = Utils::get_constraint_system_from_artifact(&program_artifact, true);
    let builder =
        UltraCircuitBuilder::<Bn254>::create_circuit(constraint_system, 0, vec![], true, false);
    let crs = VerifyingKey::get_crs(&builder, CRS_PATH_G1, CRS_PATH_G2).unwrap();
    let verifying_key = VerifyingKey::create(builder, crs).unwrap();

    let is_valid = UltraHonk::verify(proof, verifying_key).unwrap();
    assert!(is_valid);
}

fn witness_and_proof_test(name: &str) {
    let circuit_file = format!("../test_vectors/noir/{}/kat/{}.json", name, name);
    let prover_toml = format!("../test_vectors/noir/{}/Prover.toml", name);
    let proof_file = format!("../test_vectors/noir/{}/kat/{}.proof", name, name);

    let program_artifact = Utils::get_program_artifact_from_file(&circuit_file)
        .expect("failed to parse program artifact");

    let test_network1 = Rep3TestNetwork::default();
    let test_network2 = Rep3TestNetwork::default();
    let mut threads = Vec::with_capacity(3);
    for (net1, net2) in test_network1
        .get_party_networks()
        .into_iter()
        .zip(test_network2.get_party_networks())
    {
        let artifact = program_artifact.clone();
        let prover_toml = prover_toml.clone();
        threads.push(thread::spawn(move || {
            let constraint_system = Utils::get_constraint_system_from_artifact(&artifact, true);
            let solver = Rep3CoSolver::from_network(net1, artifact, prover_toml).unwrap();
            let witness = solver.solve().unwrap();
            let witness = convert_witness_rep3(witness);

            let builder = Rep3CoBuilder::<Bn254, _>::create_circuit(
                constraint_system,
                0,
                witness,
                true,
                false,
            );

            let prover_crs = ProvingKey::get_prover_crs(&builder, CRS_PATH_G1)
                .expect("failed to get prover crs");

            let id = net2.id;

            let runtime = runtime::Builder::new_current_thread().build().unwrap();
            let mut io_context0 = runtime.block_on(IoContext::init(net2)).unwrap();
            let io_context1 = runtime.block_on(io_context0.fork()).unwrap();
            let driver = Rep3UltraHonkDriver::new(io_context0, io_context1);
            let proving_key = ProvingKey::create(id, builder, prover_crs);

            let prover = CoUltraHonk::new(driver);
            runtime.block_on(prover.prove(proving_key)).unwrap()
        }));
    }

    let mut proofs = threads
        .into_iter()
        .map(|t| t.join().unwrap())
        .collect::<Vec<_>>();
    let proof = proofs.pop().unwrap();
    for p in proofs {
        assert_eq!(proof, p);
    }

    let proof_u8 = proof.to_buffer();

    let read_proof_u8 = std::fs::read(&proof_file).unwrap();
    assert_eq!(proof_u8, read_proof_u8);

    let read_proof = HonkProof::from_buffer(&read_proof_u8).unwrap();
    assert_eq!(proof, read_proof);

    // Get vk
    let constraint_system = Utils::get_constraint_system_from_artifact(&program_artifact, true);
    let builder =
        UltraCircuitBuilder::<Bn254>::create_circuit(constraint_system, 0, vec![], true, false);
    let crs = VerifyingKey::get_crs(&builder, CRS_PATH_G1, CRS_PATH_G2).unwrap();
    let verifying_key = VerifyingKey::create(builder, crs).unwrap();

    let is_valid = UltraHonk::verify(proof, verifying_key).unwrap();
    assert!(is_valid);
}

#[test]
fn poseidon_witness_and_proof_test() {
    witness_and_proof_test("poseidon");
}

#[test]
fn poseidon_proof_test() {
    proof_test("poseidon");
}
