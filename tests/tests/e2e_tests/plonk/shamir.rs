use ark_bn254::Bn254;
use circom_types::{
    groth16::witness::Witness,
    plonk::{JsonVerificationKey, PlonkProof, ZKey},
    r1cs::R1CS,
};
use collaborative_groth16::{circuit::Circuit, groth16::SharedWitness};
use collaborative_plonk::{plonk::Plonk, CollaborativePlonk};
use itertools::izip;
use mpc_core::protocols::shamir::ShamirProtocol;
use rand::thread_rng;
use std::{fs::File, thread};
use tests::shamir_network::{PartyTestNetwork, ShamirTestNetwork};

fn e2e_poseidon_bn254_inner(num_parties: usize, threshold: usize) {
    let zkey_file = File::open("../test_vectors/Plonk/bn254/poseidon/poseidon.zkey").unwrap();
    let r1cs_file = File::open("../test_vectors/Plonk/bn254/poseidon/poseidon.r1cs").unwrap();
    let witness_file = File::open("../test_vectors/Plonk/bn254/poseidon/witness.wtns").unwrap();
    let vk: JsonVerificationKey<Bn254> = serde_json::from_reader(
        File::open("../test_vectors/Plonk/bn254/multiplierAdd2/verification_key.json").unwrap(),
    )
    .unwrap();

    let witness = Witness::<ark_bn254::Fr>::from_reader(witness_file).unwrap();
    let pk1 = ZKey::<Bn254>::from_reader(zkey_file).unwrap();
    let pk = vec![pk1.clone(); num_parties];
    let r1cs1 = R1CS::<Bn254>::from_reader(r1cs_file).unwrap();
    let circuit = Circuit::new(r1cs1.clone(), witness);
    let (public_inputs1, witness) = circuit.get_wire_mapping();
    let mut rng = thread_rng();
    let witness_share =
        SharedWitness::share_shamir(&witness, &public_inputs1, threshold, num_parties, &mut rng);

    let test_network = ShamirTestNetwork::new(num_parties);
    let mut threads = vec![];

    for (net, x, pk) in izip!(test_network.get_party_networks(), witness_share, pk,) {
        threads.push(thread::spawn(move || {
            let shamir =
                ShamirProtocol::<ark_bn254::Fr, PartyTestNetwork>::new(threshold, net).unwrap();
            let prover =
                CollaborativePlonk::<ShamirProtocol<ark_bn254::Fr, PartyTestNetwork>, Bn254>::new(
                    shamir,
                );
            prover.prove(pk, x).unwrap()
        }));
    }
    let mut results = Vec::with_capacity(num_parties);
    for r in threads {
        results.push(r.join().unwrap());
    }
    let result1 = results.pop().unwrap();
    for r in results {
        assert_eq!(result1, r);
    }

    let ser_proof = serde_json::to_string(&result1).unwrap();
    let der_proof = serde_json::from_str::<PlonkProof<Bn254>>(&ser_proof).unwrap();
    assert_eq!(der_proof, result1);
    let verified =
        Plonk::<Bn254>::verify(&vk, &der_proof, &public_inputs1[1..]).expect("can verify");
    assert!(verified);
}

#[test]
fn e2e_poseidon_bn254() {
    e2e_poseidon_bn254_inner(3, 1);
    e2e_poseidon_bn254_inner(10, 4);
}
