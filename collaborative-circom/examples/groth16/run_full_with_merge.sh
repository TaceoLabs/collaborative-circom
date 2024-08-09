EXAMPLE_NAME=multiplier2

# split inputs into shares
cargo run --release --bin co-circom -- split-input --config ../configs/config.toml --circuit test_vectors/$EXAMPLE_NAME/circuit.circom --link-library test_vectors/$EXAMPLE_NAME/lib --input test_vectors/$EXAMPLE_NAME/input0.json --protocol REP3 --curve BN254 --out-dir test_vectors/$EXAMPLE_NAME
cargo run --release --bin co-circom -- split-input --config ../configs/config.toml --circuit test_vectors/$EXAMPLE_NAME/circuit.circom --link-library test_vectors/$EXAMPLE_NAME/lib --input test_vectors/$EXAMPLE_NAME/input1.json --protocol REP3 --curve BN254 --out-dir test_vectors/$EXAMPLE_NAME
# merge inputs into single input file
cargo run --release --bin co-circom -- merge-input-shares --config ../configs/config.toml --inputs test_vectors/$EXAMPLE_NAME/input0.json.0.shared --inputs test_vectors/$EXAMPLE_NAME/input1.json.0.shared --protocol REP3 --curve BN254 --out test_vectors/$EXAMPLE_NAME/input.json.0.shared
cargo run --release --bin co-circom -- merge-input-shares --config ../configs/config.toml --inputs test_vectors/$EXAMPLE_NAME/input0.json.1.shared --inputs test_vectors/$EXAMPLE_NAME/input1.json.1.shared --protocol REP3 --curve BN254 --out test_vectors/$EXAMPLE_NAME/input.json.1.shared
cargo run --release --bin co-circom -- merge-input-shares --config ../configs/config.toml --inputs test_vectors/$EXAMPLE_NAME/input0.json.2.shared --inputs test_vectors/$EXAMPLE_NAME/input1.json.2.shared --protocol REP3 --curve BN254 --out test_vectors/$EXAMPLE_NAME/input.json.2.shared
# run witness extension in MPC
cargo run --release --bin co-circom -- generate-witness --input test_vectors/$EXAMPLE_NAME/input.json.0.shared --circuit test_vectors/$EXAMPLE_NAME/circuit.circom --link-library test_vectors/$EXAMPLE_NAME/lib --protocol REP3 --curve BN254 --config ../configs/party1.toml --out test_vectors/$EXAMPLE_NAME/witness.wtns.0.shared &
cargo run --release --bin co-circom -- generate-witness --input test_vectors/$EXAMPLE_NAME/input.json.1.shared --circuit test_vectors/$EXAMPLE_NAME/circuit.circom --link-library test_vectors/$EXAMPLE_NAME/lib --protocol REP3 --curve BN254 --config ../configs/party2.toml --out test_vectors/$EXAMPLE_NAME/witness.wtns.1.shared &
cargo run --release --bin co-circom -- generate-witness --input test_vectors/$EXAMPLE_NAME/input.json.2.shared --circuit test_vectors/$EXAMPLE_NAME/circuit.circom --link-library test_vectors/$EXAMPLE_NAME/lib --protocol REP3 --curve BN254 --config ../configs/party3.toml --out test_vectors/$EXAMPLE_NAME/witness.wtns.2.shared
# run proving in MPC
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/$EXAMPLE_NAME/witness.wtns.0.shared --zkey test_vectors/$EXAMPLE_NAME/$EXAMPLE_NAME.zkey --protocol REP3 --curve BN254 --config ../configs/party1.toml --out proof.0.json --public-input public_input.json &
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/$EXAMPLE_NAME/witness.wtns.1.shared --zkey test_vectors/$EXAMPLE_NAME/$EXAMPLE_NAME.zkey --protocol REP3 --curve BN254 --config ../configs/party2.toml --out proof.1.json &
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/$EXAMPLE_NAME/witness.wtns.2.shared --zkey test_vectors/$EXAMPLE_NAME/$EXAMPLE_NAME.zkey --protocol REP3 --curve BN254 --config ../configs/party3.toml --out proof.2.json
# verify proof
cargo run --release --bin co-circom -- verify groth16 --config ../configs/config.toml --proof proof.0.json --vk test_vectors/$EXAMPLE_NAME/verification_key.json --public-input public_input.json --curve BN254
