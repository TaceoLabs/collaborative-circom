# coCircom

[![X (formerly Twitter) Follow](https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white)](https://twitter.com/TACEO_IO)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/collaborativeSNARK)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/gWZW2TANpk)

[![docs.io](https://img.shields.io/badge/coCircom-docs-green)](https://docs.taceo.io/)

<!--[![crates.io](https://img.shields.io/badge/crates.io-v0.1.0-blue)](https://crates.io/)-->

**coCircom** is a tool for building **coSNARKs**, a new technology that enables
multiple distrusting parties to collaboratively compute a **zero-knowledge
proof** (ZKP). It leverages the existing domain-specific language
[circom](https://github.com/iden3/circom) to define arithmetic circuits. With
coCircom, all existing circom circuits can be promoted to coSNARKs without any
modification to the original circuit.

Additionally, coCircom is fully compatible with the **Groth16** and **PLONK**
backends of [snarkjs](https://github.com/iden3/snarkjs). Proofs built with
coCircom can be verified using snarkjs, and vice versa.

The project is built with pure Rust and consists of multiple libraries:

- **collaborative-groth16**: A library for creating
  [Groth16](https://eprint.iacr.org/2016/260.pdf) coSNARKs, verifiable by
  snarkjs.
- **collaborative-plonk**: A library for creating
  [PLONK](https://eprint.iacr.org/2019/953.pdf) coSNARKs, verifiable by snarkjs.
- **circom-mpc-compiler**: A compiler that lowers circom files to byte code
  consumable by our MPC-VM.
- **circom-mpc-vm**: An MPC-VM that executes the byte code produced by the
  compiler in a distributed manner (building the extended witness).
- **circom-types**: A library for serialization and deserialization of snarkjs
  artifacts, such as ZKeys and R1CS files.

The following libraries are agnostic to coCircom and will be used in the future
for other coSNARKs:

- **mpc-core**: Implementation of MPC protocols.
- **mpc-net**: Network library for MPC protocols.

The binary `co-circom` is a CLI tool that uses the libraries to build a coSNARK
(source found in the **collaborative-circom** folder).

## Installation

### Prerequisites

1. Install Rust. You can find the instructions
   [here](https://www.rust-lang.org/tools/install).
2. Install the circom ecosystem. You can find the instructions
   [here](https://docs.circom.io/getting-started/installation/).

### Download Binary from Release

1. You can find the latest release
   [here](https://github.com/TaceoLabs/collaborative-circom/releases/latest).
2. Download the binary for your operating system.

3. Extract the binary from the archive

```bash
tar xf co-circom-YOUR_ARCHITECTURE.tar.gz
```

4. Make the binary executable (if necessary):

```bash
chmod +x co-circom
```

### Install from Source

1. Clone the repository:

```bash
git clone https://github.com/TaceoLabs/collaborative-circom.github
```

2. Build the project:

```bash
cd collaborative-circom && cargo build --release
```

3. You can find the binary in the `target/release` directory.

## Documentation

You can find the documentation of coCircom [here](https://docs.taceo.io/).

## CLI Usage

This section covers the necessary steps to build a Groth16 coSNARK using the previously
installed coCircom binary. For demonstration purposes, we will use a simple
circuit called `adder.circom`.

```c++
pragma circom 2.0.0;

template Adder() {
    signal input a;
    signal input b;
    signal output c;

    c <== a + b;
}
component main = Adder();
```

With this circuit, we can prove that we know two numbers that sum up to `c`.
While this is a basic example, it is sufficient for demonstration purposes. We
will use a replicated secret sharing scheme with 3 parties for all steps
(denoted as REP3 in the commands).

### Step 1: Generate the R1CS File

First, we need to generate the R1CS file from the circom file. We use circom for
this step:

```bash
circom adder.circom --r1cs
```

### Step 2: Perform Groth16 Setup

Next, we need to perform the Groth16 setup using circom and snarkjs. Refer to
the
[circom documentation](https://docs.circom.io/getting-started/proving-circuits/)
for detailed instructions up to the "Generating a Proof" section.

For the following steps, we call the ZKey file `adder.zkey`.

### Step 3: Prepare the Input

This step involves handling inputs from either a single party or multiple
distrusting parties.

#### Input from a Single Party

If the input comes from a single party, we simply split (secret-share) the
input. Assume we have a file named `input.json` with the following content:

```json
{
  "a": "3",
  "b": "4"
}
```

To split the input, use the following command:

```bash
mkdir out && ./co-circom split-input --circuit adder.circom --input input.json --protocol REP3 --curve BN254 --out-dir out/
```

This command will generate secret-shared inputs in the `out` directory, creating
separate files for each party. These files will be named `input.json.0.shared`,
`input.json.1.shared`, and `input.json.2.shared`, corresponding to the shares
for each respective party.

**Note**: In practice, it is crucial that each party has exclusive access to
their respective file. Sharing these files across parties compromises the
security of the shared witness.

#### Input from Multiple Parties

When the input comes from multiple parties, each party first secret-shares their
respective inputs locally. For example, consider two input files: `input0.json`:

```json
{
  "a": "3"
}
```

`input1.json`:

```json
{
  "b": "4"
}
```

In practice, these files would typically reside on different machines. Each
party secret-shares their input individually:

```bash
mkdir out && ./co-circom split-input --circuit adder.circom --input input0.json --protocol REP3 --curve BN254 --out-dir out/
```

The parties then send their shares to the computing nodes, which can merge the
shares. All computing nodes execute the following command (provided here for the
first party):

```bash
./co-circom merge-input-shares --inputs out/input0.json.0.shared --inputs out/input1.json.0.shared --protocol REP3 --curve BN254 --out out/input.json.0.shared
```

### Step 4: Extended Witness Generation

To generate the witness, we execute the circuit with the secret-shared input
obtained from the previous step. Additionally, computing nodes require
networking configuration files and TLS key material. Examples of these
configurations can be found in the
[configs](/collaborative-circom/examples/configs) and key materials in the
[keys](/collaborative-circom/examples/data) directory. Refer to our
[documentation](https://docs.taceo.io/network-config.html) for detailed
configuration instructions.

All parties execute the following command (provided here for the first party):

```bash
./co-circom generate-witness --input out/input.json.0.shared --circuit adder.circom --protocol REP3 --curve BN254 --config configs/party1.toml --out out/witness.wtns.0.shared
```

**Note**: You need to execute three nodes in parallel. This command will block
until all nodes have finished, so you will likely need three separate terminals
;)

### Step 5: Generate the Proof

Next, we generate the proof. Each computing node executes the following command:

```bash
./co-circom generate-proof --witness out/witness.wtns.0.shared --zkey adder.zkey --protocol REP3 --curve BN254 --config configs/party1.toml --out proof.0.json --public-input public_input.0.json
```

Remember to execute this command on all three nodes.

### Step 6: Verify the Proof

You can verify the proof using either coCircom or snarkjs. Here's the command
for using coCircom:

```bash
./co-circom verify --proof proof.0.json --vk verification_key.json --public-input public_input.0.json --curve BN254
```

**Note**: The `verification_key.json` was generated in Step 2.

For more examples, please refer to the
[examples folder](/collaborative-circom/examples/). You'll find bash scripts
there that demonstrate all the necessary steps.

## License

This project is licensed under either the [MIT License](LICENSE-MIT) or the
[Apache](LICENSE-APACHE), at your choice.

`SPDX-License-Identifier: Apache-2.0 OR MIT`

Select sub-libraries within this project have different licenses, reflecting
their dependencies on
[circom](https://github.com/iden3/circom?tab=GPL-3.0-1-ov-file).

- **collaborative-circom**: Licensed under [GPL-3.0](LICENSE.GPL)
  `SPDX-License-Identifier: GPL-3.0-only`.
- **circom-mpc-compiler**: Licensed under [GPL-3.0](LICENSE.GPL)
  `SPDX-License-Identifier: GPL-3.0-only`.

## Disclaimer

This software is **experimental** and **un-audited**, provided on an "as is" and
"as available" basis. We do **not provide any warranties**, express or implied,
including but not limited to warranties of merchantability or fitness for a
particular purpose. We will **not be liable for any losses, damages, or issues**
arising from the use of this software, whether direct or indirect.

Users are encouraged to exercise caution and conduct their own independent
assessments and testing. **By using this software, you acknowledge and accept
the risks associated with its experimental** nature and **agree that the
developers and contributors are not responsible for any consequences** resulting
from its use.
