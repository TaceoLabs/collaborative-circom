# UltraHonk

This crate uses our rewrite of Atec's UltraHonk prover and verifier in Rust (see `collaborative-circom/co-noir/ultrahonk`). It is compatible with Barretenberg v0.55.1. To get Barretenberg with this version, use the following commands:

```bash
git clone https://github.com/AztecProtocol/aztec-packages.git
cd aztec-packages
git checkout tags/aztec-package-v0.55.1
```

To compile Barretenberg, one can use:

```bash
cd barretenberg/cpp
bash ./scripts/docker_interactive.sh ubuntu
mkdir build
cd build
cmake --preset clang16 -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build .
```

The prover in this repository, i.e., ``UltraHonk::prove`` in `src/prover.rs`, is compatible with `UltraProver_<UltraFlavor>` in Barretenberg. Similar, the ``Ultrahnok::verify`` verifier in `src/verifier.rs` is compatible with `UltraVerifier_<UltraFlavor>` in Barretenberg.

Currently, the circuit builder related code in `src/parse/` is only compatible with basic field arithmetic gates from Noir, stay tuned for more features.

## Usage

First, one needs to create the circuit file from a Noir source code. Your Noir project should have the following files/folders:

- `src/main.nr`: Contains the code which should be executed in MPC and/or proven.
- `Nargo.toml`: Similar to Cargo.toml, just for Noir projects.
- `Prover.toml`: The inputs for the main function in `src/main.nr` used in proof generation.

To create the circuit file used in Co-Noir, one needs to install Nargo following the instructions in [https://noir-lang.org/docs/getting_started/installation/](https://noir-lang.org/docs/getting_started/installation/). Our prover is compatible with Nargo version 0.33.0.
Then you can just execute the following command:

```bash
nargo compile
```

The resulting circuit file (*.json) is then located in the `target` folder.

Alternatively, if you want to create the extended witness from the input in `Prover.toml`, use

```bash
nargo execute witness
```

This command then stores the resulting circuit file (*.json), alongside the extended witness (`witness.gz`) in the `target` folder.

### Co-Noir Commands

Currently the Co-Noir binary allows the following commands, which we illustrate on the Noir example of proving a Poseidon Hash computation. See the Nargo source code in `examples/test_vectors/poseidon/src/main.nr`.

The following commands are all executed from the `examples` folder, just like the .sh examples.

#### SplitInput

If you want to calculate the extended witness and the prove from the input file `test_vectors/poseidon/Prover.toml` (i.e., if you want to hash the input ["0", "1", "2", "3", "4", "5", "6", "7"] in MPC and prove the computation), you first have to secret share this input file. This can be done using the SplitInput command:

```bash
cargo run --release --bin co-noir -- split-input --circuit test_vectors/poseidon/poseidon.json --input test_vectors/poseidon/Prover.toml --protocol REP3 --out-dir test_vectors/poseidon
```

This command uses the *REP3* MPC protocol and produces the shares for 3 MPC in the `test_vectors/poseidon` folder. As input it takes the circuit file `poseidon.json` and the `Prover.toml` file which contains the preimage of the hash which we want to share.

#### GenerateWitness

After the SplitInput command, the computing parties engage in executing the circuit (in our case Poseidon) on the shared input in MPC. Each party thus executes the following command:

```bash
cargo run --release --bin co-noir -- generate-witness --input test_vectors/poseidon/Prover.toml.shared --circuit test_vectors/poseidon/poseidon.json --protocol REP3 --config configs/party.toml --out test_vectors/poseidon/poseidon.gz.shared
```

Again, `poseidon.json` is the circuit file from Noir, while `Prover.toml.shared` is one output of SplitWitness and `party.toml` is a network configuration. As MPC protocol we currently only support *REP3* for the Witness extension.

#### SplitWitness

Instead of performing the witness extension in MPC, one can also secret share the .gz witness file computed by Noir. The command is the following:

```bash
cargo run --release --bin co-noir -- split-witness --witness test_vectors/poseidon/poseidon.gz --circuit test_vectors/poseidon/poseidon.json --protocol REP3 --out-dir test_vectors/poseidon
```

Here, `poseidon.json` is the circuit file from Noir, `poseidon.gz` the extended witness from noir, and the output shares are stored in `test_vectors/poseidon`. The output shares are therebey indistinguishable from the output shares produced by the GenerateWitness command.
For SplitWitness, both REP3 and Shamir are supported.

#### TranslateWitness

This command can be used to translate extended witnesses (outputs of GenerateWitness or SplitWitness) from REP3 to 3-party Shamir secret sharing:

```bash
cargo run --release --bin co-noir -- translate-witness --witness test_vectors/poseidon/poseidon.gz.shared --src-protocol REP3 --target-protocol SHAMIR --config configs/party.toml --out test_vectors/poseidon/shamir_poseidon.gz.shared
```

Here, `poseidon.gz.shared` is the REP3 input share, `shamir_poseidon.gz.shared` the Shamir output share, and `party.toml` is a network configuration.

#### GenerateProof

To create a proof in MPC, one needs the extended witness (from GenerateWitness, SplitWitness, or TranslateWitness):

```bash
cargo run --release --bin co-noir -- generate-proof --witness test_vectors/poseidon/poseidon.gz.shared --circuit test_vectors/poseidon/poseidon.json --crs test_vectors/bn254_g1.dat --protocol REP3 --config configs/party.toml --out proof.proof --public-input public_input.json
```

Here, `poseidon.gz.shared` is the share of the witness, `poseidon.json` is the circuit file from Noir, `bn254_g1.dat` is the file storing the prover CRS and `party.toml` is the network configuration. As output, one creates the UltraHonk proof `proof.proof` and the output of the circuit `public_input.json`.

The corresponding Barretenberg command (from `barretenberg/cpp/build/bin`) is:

```bash
./bb prove_ultra_honk -b poseidon.json -w poseidon.gz -o proof.proof
```

where poseidon.gz is the witness file created by Noir (which is equivalent to a non-secret-shared variant of `poseidon.gz.shared`). The generated proof key is the same, regardless of using Co-Noir or Barretenberg.
Note: Barretenberg does not require the file for storing the CRS, since Barretenberg automatically downloads it if it is not present.

#### CreateVK

To verify the created proof, we first need to create a verification key. This can be done with:

```bash
cargo run --release --bin co-noir -- create-vk --circuit test_vectors/poseidon/poseidon.json --crs test_vectors/bn254_g1.dat --vk test_vectors/poseidon/verification_key
```

Here, `poseidon.json` is the circuit file from Noir, `bn254_g1.dat` is the file storing the prover CRS, and the output is written to `verification_key`.

The corresponding Barretenberg command (from `barretenberg/cpp/build/bin`) is:

```bash
./bb write_vk_ultra_honk -b poseidon.json -o verification_key
```

Here, `poseidon.json` is the circuit file from Noir. The output verification key is the same, regardless of using Co-Noir or Barretenberg.
Note: Barretenberg does not require the file for storing the CRS, since Barretenberg automatically downloads it if it is not present.

#### Verify

To verify the proof, just use:

```bash
cargo run --release --bin co-noir -- verify --proof proof.proof --vk test_vectors/poseidon/verification_key --crs test_vectors/bn254_g2.dat
```

Here, `proof.proof` is the proof we want to verify, `verification_key` is the output of CreateVK, and `bn254_g2.dat` is the verifier CRS.

The corresponding Barretenberg command (from `barretenberg/cpp/build/bin`) is:

```bash
./bb verify_ultra_honk -k verification_key -p proof.proof
```

Note: Barretenberg does not require the file for storing the CRS, since Barretenberg automatically downloads it if it is not present.
