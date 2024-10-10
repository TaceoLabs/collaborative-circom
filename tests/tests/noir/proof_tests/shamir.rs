use crate::proof_tests::{CRS_PATH_G1, CRS_PATH_G2};
use ark_bn254::Bn254;
use co_ultrahonk::{
    prelude::{
        CoUltraHonk, HonkProof, ProvingKey, ShamirCoBuilder, ShamirUltraHonkDriver,
        SharedBuilderVariable, UltraCircuitBuilder, UltraCircuitVariable, UltraHonk, Utils,
        VerifyingKey,
    },
    MAX_PARTIAL_RELATION_LENGTH, OINK_CRAND_PAIRS_CONST, OINK_CRAND_PAIRS_FACTOR_N,
    OINK_CRAND_PAIRS_FACTOR_N_MINUS_ONE, SUMCHECK_ROUND_CRAND_PAIRS_FACTOR,
};
use mpc_core::protocols::shamir::{ShamirPreprocessing, ShamirProtocol};
use std::thread;
use tests::shamir_network::ShamirTestNetwork;
use tokio::runtime;

fn proof_test(name: &str, num_parties: usize, threshold: usize) {
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

    let test_network = ShamirTestNetwork::new(num_parties);
    let mut threads = Vec::with_capacity(num_parties);
    for net in test_network.get_party_networks() {
        let artifact = program_artifact.clone();
        let witness = witness.clone();
        threads.push(thread::spawn(move || {
            let constraint_system = Utils::get_constraint_system_from_artifact(&artifact, true);

            let builder = ShamirCoBuilder::<Bn254, _>::create_circuit(
                constraint_system,
                0,
                witness,
                true,
                false,
            );

            let prover_crs = ProvingKey::get_prover_crs(&builder, CRS_PATH_G1)
                .expect("failed to get prover crs");

            let id = net.id;

            let proving_key = ProvingKey::create(id, builder, prover_crs);

            let runtime = runtime::Builder::new_current_thread().build().unwrap();
            let n = proving_key.circuit_size as usize;
            let num_pairs_oink_prove = OINK_CRAND_PAIRS_FACTOR_N * n
                + OINK_CRAND_PAIRS_FACTOR_N_MINUS_ONE * (n - 1)
                + OINK_CRAND_PAIRS_CONST;
            // log2(n) * ((n >>= 1) / 2) == n - 1
            let num_pairs_sumcheck_prove =
                SUMCHECK_ROUND_CRAND_PAIRS_FACTOR * MAX_PARTIAL_RELATION_LENGTH * (n - 1);
            let num_pairs = num_pairs_oink_prove + num_pairs_sumcheck_prove;
            let preprocessing = runtime
                .block_on(ShamirPreprocessing::new(threshold, net, num_pairs))
                .unwrap();
            let mut io_context0 = ShamirProtocol::from(preprocessing);
            let io_context1 = runtime.block_on(io_context0.fork_with_pairs(0)).unwrap();
            let driver = ShamirUltraHonkDriver::new(io_context0, io_context1);

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
fn poseidon_proof_test() {
    proof_test("poseidon", 3, 1);
}
