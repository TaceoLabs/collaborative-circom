# split input into shares
cargo run --release --bin co-circom -- split-input --config ../configs/config.toml  --circuit test_vectors/sum_arrays/circuit.circom --input test_vectors/sum_arrays/input.json --protocol REP3 --curve BN254 --out-dir test_vectors/sum_arrays
# run witness extension in MPC
cargo run --release --bin co-circom -- generate-witness --input test_vectors/sum_arrays/input.json.0.shared --circuit test_vectors/sum_arrays/circuit.circom --link-library test_vectors/sum_arrays/lib --protocol REP3 --curve BN254 --config ../configs/party1.toml --out test_vectors/sum_arrays/witness.wtns.0.shared &
cargo run --release --bin co-circom -- generate-witness --input test_vectors/sum_arrays/input.json.1.shared --circuit test_vectors/sum_arrays/circuit.circom --link-library test_vectors/sum_arrays/lib --protocol REP3 --curve BN254 --config ../configs/party2.toml --out test_vectors/sum_arrays/witness.wtns.1.shared &
cargo run --release --bin co-circom -- generate-witness --input test_vectors/sum_arrays/input.json.2.shared --circuit test_vectors/sum_arrays/circuit.circom --link-library test_vectors/sum_arrays/lib --protocol REP3 --curve BN254 --config ../configs/party3.toml --out test_vectors/sum_arrays/witness.wtns.2.shared
# run proving in MPC
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/sum_arrays/witness.wtns.0.shared --zkey test_vectors/sum_arrays/sum_arrays.zkey --protocol REP3 --curve BN254 --config ../configs/party1.toml --out proof.0.json --public-input public_input.json &
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/sum_arrays/witness.wtns.1.shared --zkey test_vectors/sum_arrays/sum_arrays.zkey --protocol REP3 --curve BN254 --config ../configs/party2.toml --out proof.1.json &
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/sum_arrays/witness.wtns.2.shared --zkey test_vectors/sum_arrays/sum_arrays.zkey --protocol REP3 --curve BN254 --config ../configs/party3.toml --out proof.2.json
# verify proof
cargo run --release --bin co-circom -- verify groth16 --config ../configs/config.toml --proof proof.0.json --vk test_vectors/sum_arrays/verification_key.json --public-input public_input.json --curve BN254
