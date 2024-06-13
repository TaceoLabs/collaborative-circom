EXAMPLE_NAME=poseidon

# split input into shares
cargo run --release --bin co-circom -- split-input --input test_vectors/$EXAMPLE_NAME/input.json --protocol bla --out-dir test_vectors/$EXAMPLE_NAME
# run witness extension in MPC
cargo run --release --bin co-circom -- generate-witness --input test_vectors/$EXAMPLE_NAME/input.json.0.shared --circuit test_vectors/$EXAMPLE_NAME/circuit.circom --link-library test_vectors/$EXAMPLE_NAME/lib --protocol bla --config configs/party1.toml --out test_vectors/$EXAMPLE_NAME/witness.wtns.0.shared &
cargo run --release --bin co-circom -- generate-witness --input test_vectors/$EXAMPLE_NAME/input.json.1.shared --circuit test_vectors/$EXAMPLE_NAME/circuit.circom --link-library test_vectors/$EXAMPLE_NAME/lib --protocol bla --config configs/party2.toml --out test_vectors/$EXAMPLE_NAME/witness.wtns.1.shared &
cargo run --release --bin co-circom -- generate-witness --input test_vectors/$EXAMPLE_NAME/input.json.2.shared --circuit test_vectors/$EXAMPLE_NAME/circuit.circom --link-library test_vectors/$EXAMPLE_NAME/lib --protocol bla --config configs/party3.toml --out test_vectors/$EXAMPLE_NAME/witness.wtns.2.shared
# run proving in MPC
cargo run --release --bin co-circom -- generate-proof --witness test_vectors/$EXAMPLE_NAME/witness.wtns.0.shared --zkey test_vectors/$EXAMPLE_NAME/$EXAMPLE_NAME.zkey --protocol bla --config configs/party1.toml --out proof.0.json --public-input public_input.json &
cargo run --release --bin co-circom -- generate-proof --witness test_vectors/$EXAMPLE_NAME/witness.wtns.1.shared --zkey test_vectors/$EXAMPLE_NAME/$EXAMPLE_NAME.zkey --protocol bla --config configs/party2.toml --out proof.1.json &
cargo run --release --bin co-circom -- generate-proof --witness test_vectors/$EXAMPLE_NAME/witness.wtns.2.shared --zkey test_vectors/$EXAMPLE_NAME/$EXAMPLE_NAME.zkey --protocol bla --config configs/party3.toml --out proof.2.json
# # verify proof
cargo run --release --bin co-circom -- verify --proof proof.0.json --vk test_vectors/$EXAMPLE_NAME/verification_key.json --public-input public_input.json 
