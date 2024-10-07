# UltraHonk

This crate is a rewrite of Atec's UltraHonk prover and verifier in Rust. It is compatible with Barretenberg v0.55.1. To get Barretenberg with this version, use the following commands:

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

For examples and corresponding Barretenberg operations we refer to `collaborative-circom/co-noir/co-noir/Readme.md`.
