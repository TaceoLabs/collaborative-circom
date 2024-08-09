# split input into shares
cargo run --release --bin co-circom -- split-input --config ../configs/config.toml --circuit test_vectors/kyc/circuit.circom --link-library test_vectors/kyc/lib --input test_vectors/kyc/input.json --protocol REP3 --curve BLS12-381 --out-dir test_vectors/kyc
# run witness extension in MPC
cargo run --release --bin co-circom -- generate-witness --input test_vectors/kyc/input.json.0.shared --circuit test_vectors/kyc/circuit.circom --link-library test_vectors/kyc/lib --protocol REP3 --curve BLS12-381 --config ../configs/party1.toml --out test_vectors/kyc/witness.wtns.0.shared &
cargo run --release --bin co-circom -- generate-witness --input test_vectors/kyc/input.json.1.shared --circuit test_vectors/kyc/circuit.circom --link-library test_vectors/kyc/lib --protocol REP3 --curve BLS12-381 --config ../configs/party2.toml --out test_vectors/kyc/witness.wtns.1.shared &
cargo run --release --bin co-circom -- generate-witness --input test_vectors/kyc/input.json.2.shared --circuit test_vectors/kyc/circuit.circom --link-library test_vectors/kyc/lib --protocol REP3 --curve BLS12-381 --config ../configs/party3.toml --out test_vectors/kyc/witness.wtns.2.shared
# run translation from REP3 to Shamir
cargo run --release --bin co-circom -- translate-witness --witness test_vectors/kyc/witness.wtns.0.shared --src-protocol REP3 --target-protocol SHAMIR --curve BLS12-381 --config ../configs/party1.toml --out test_vectors/kyc/shamir_witness.wtns.0.shared &
cargo run --release --bin co-circom -- translate-witness --witness test_vectors/kyc/witness.wtns.1.shared --src-protocol REP3 --target-protocol SHAMIR --curve BLS12-381 --config ../configs/party2.toml --out test_vectors/kyc/shamir_witness.wtns.1.shared &
cargo run --release --bin co-circom -- translate-witness --witness test_vectors/kyc/witness.wtns.2.shared --src-protocol REP3 --target-protocol SHAMIR --curve BLS12-381 --config ../configs/party3.toml --out test_vectors/kyc/shamir_witness.wtns.2.shared
# run proving in MPC
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/kyc/shamir_witness.wtns.0.shared --zkey test_vectors/kyc/bls12/kyc.zkey --protocol SHAMIR --curve BLS12-381 --config ../configs/party1.toml --out proof.0.json --public-input public_input.json &
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/kyc/shamir_witness.wtns.1.shared --zkey test_vectors/kyc/bls12/kyc.zkey --protocol SHAMIR --curve BLS12-381 --config ../configs/party2.toml --out proof.1.json &
cargo run --release --bin co-circom -- generate-proof groth16 --witness test_vectors/kyc/shamir_witness.wtns.2.shared --zkey test_vectors/kyc/bls12/kyc.zkey --protocol SHAMIR --curve BLS12-381 --config ../configs/party3.toml --out proof.2.json
# verify proof
cargo run --release --bin co-circom -- verify groth16 --config ../configs/config.toml --proof proof.0.json --vk test_vectors/kyc/bls12/verification_key.json --public-input public_input.json --curve BLS12-381
