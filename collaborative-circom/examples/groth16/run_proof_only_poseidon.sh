# split input into shares
cargo run --release --bin co-circom -- split-witness --config ../configs/config.toml --witness test_vectors/poseidon/witness.wtns --r1cs test_vectors/poseidon/poseidon.r1cs --protocol REP3 --curve BN254 --out-dir test_vectors/poseidon
# run proving in MPC
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/poseidon/witness.wtns.0.shared --zkey test_vectors/poseidon/poseidon.zkey --protocol REP3 --curve BN254 --config ../configs/party1.toml --out proof.0.json --public-input public_input.json &
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/poseidon/witness.wtns.1.shared --zkey test_vectors/poseidon/poseidon.zkey --protocol REP3 --curve BN254 --config ../configs/party2.toml --out proof.1.json &
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/poseidon/witness.wtns.2.shared --zkey test_vectors/poseidon/poseidon.zkey --protocol REP3 --curve BN254 --config ../configs/party3.toml --out proof.2.json
# verify proof
cargo run --release --bin co-circom -- verify groth16 --config ../configs/config.toml --proof proof.0.json --vk test_vectors/poseidon/verification_key.json --public-input public_input.json --curve BN254
