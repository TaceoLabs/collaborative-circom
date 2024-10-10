use crate::proof_tests::{CRS_PATH_G1, CRS_PATH_G2};
use acir::native_types::{WitnessMap, WitnessStack};
use ark_bn254::Bn254;
use ark_ec::pairing::Pairing;
use ark_ff::Zero;
use co_acvm::solver::PlainCoSolver;
use co_ultrahonk::prelude::{
    CoUltraHonk, HonkProof, PlainCoBuilder, PlainUltraHonkDriver, ProvingKey,
    SharedBuilderVariable, UltraCircuitVariable, UltraHonk, Utils,
};
use tokio::runtime;

fn witness_map_to_witness_vector<P: Pairing>(
    witness_map: WitnessMap<P::ScalarField>,
) -> Vec<SharedBuilderVariable<PlainUltraHonkDriver, P>> {
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
        wv.push(SharedBuilderVariable::from_public(f));
        index += 1;
    }
    wv
}

fn convert_witness_plain<P: Pairing>(
    mut witness_stack: WitnessStack<P::ScalarField>,
) -> Vec<SharedBuilderVariable<PlainUltraHonkDriver, P>> {
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

    let constraint_system = Utils::get_constraint_system_from_file(&circuit_file, true)
        .expect("failed to parse program artifact");
    let witness = Utils::get_witness_from_file(&witness_file).expect("failed to parse witness");

    let witness = SharedBuilderVariable::promote_public_witness_vector(witness);

    let builder =
        PlainCoBuilder::<Bn254>::create_circuit(constraint_system, 0, witness, true, false);

    let driver = PlainUltraHonkDriver;

    let crs = ProvingKey::get_crs(&builder, CRS_PATH_G1, CRS_PATH_G2).expect("failed to get crs");
    let (proving_key, verifying_key) = ProvingKey::create_keys(0, builder, crs).unwrap();

    let runtime = runtime::Builder::new_current_thread().build().unwrap();

    let prover = CoUltraHonk::new(driver);
    let proof = runtime.block_on(prover.prove(proving_key)).unwrap();
    let proof_u8 = proof.to_buffer();

    let read_proof_u8 = std::fs::read(&proof_file).unwrap();
    assert_eq!(proof_u8, read_proof_u8);

    let read_proof = HonkProof::from_buffer(&read_proof_u8).unwrap();
    assert_eq!(proof, read_proof);

    let is_valid = UltraHonk::verify(proof, verifying_key).unwrap();
    assert!(is_valid);
}

fn witness_and_proof_test(name: &str) {
    let circuit_file = format!("../test_vectors/noir/{}/kat/{}.json", name, name);
    let prover_toml = format!("../test_vectors/noir/{}/Prover.toml", name);
    let proof_file = format!("../test_vectors/noir/{}/kat/{}.proof", name, name);

    let program_artifact = Utils::get_program_artifact_from_file(&circuit_file)
        .expect("failed to parse program artifact");
    let constraint_system = Utils::get_constraint_system_from_artifact(&program_artifact, true);

    let solver = PlainCoSolver::init_plain_driver(program_artifact, prover_toml).unwrap();
    let witness = solver.solve().unwrap();
    let witness = convert_witness_plain(witness);

    let builder =
        PlainCoBuilder::<Bn254>::create_circuit(constraint_system, 0, witness, true, false);

    let driver = PlainUltraHonkDriver;

    let crs = ProvingKey::get_crs(&builder, CRS_PATH_G1, CRS_PATH_G2).expect("failed to get crs");
    let (proving_key, verifying_key) = ProvingKey::create_keys(0, builder, crs).unwrap();

    let runtime = runtime::Builder::new_current_thread().build().unwrap();

    let prover = CoUltraHonk::new(driver);
    let proof = runtime.block_on(prover.prove(proving_key)).unwrap();
    let proof_u8 = proof.to_buffer();

    let read_proof_u8 = std::fs::read(&proof_file).unwrap();
    assert_eq!(proof_u8, read_proof_u8);

    let read_proof = HonkProof::from_buffer(&read_proof_u8).unwrap();
    assert_eq!(proof, read_proof);

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
