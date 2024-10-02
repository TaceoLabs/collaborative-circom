# split input into shares
cargo run --release --bin co-noir -- split-witness --witness test_vectors/poseidon/poseidon.gz --circuit test_vectors/poseidon/poseidon.json --protocol REP3 --out-dir test_vectors/poseidon
# run proving in MPC
cargo run --release --bin co-noir -- generate-proof --witness test_vectors/poseidon/poseidon.gz.0.shared --circuit test_vectors/poseidon/poseidon.json --crs test_vectors/bn254_g1.dat --protocol REP3 --config configs/party1.toml --out proof.0.proof --public-input public_input.json &
cargo run --release --bin co-noir -- generate-proof --witness test_vectors/poseidon/poseidon.gz.1.shared --circuit test_vectors/poseidon/poseidon.json --crs test_vectors/bn254_g1.dat --protocol REP3 --config configs/party2.toml --out proof.1.proof &
cargo run --release --bin co-noir -- generate-proof --witness test_vectors/poseidon/poseidon.gz.2.shared --circuit test_vectors/poseidon/poseidon.json --crs test_vectors/bn254_g1.dat --protocol REP3 --config configs/party3.toml --out proof.2.proof
# verify proof
# cargo run --release --bin co-circom -- verify groth16 --proof proof.0.json --vk test_vectors/poseidon/verification_key.json --public-input public_input.json --curve BN254
