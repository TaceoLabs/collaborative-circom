# Quick Start

Collaborative Circom is an implementation of [collaborative SNARKs](./collsnarks.md), with a focus on the [Circom](https://circom.io) framework.
In contrast to traditional SNARKs, which are run by a single prover, collaborative SNARKs are executed using a [multiparty computation protocol](./mpc.md).

If you just want to get your hands dirty as fast as possible, here is a run-down on how to collaboratively prove the `Multiplier2` example from the [Circom documentation](https://docs.circom.io/getting-started/installation/) using Groth16.

First of all, here is the relevant Circom file:

```c++
pragma circom 2.0.0;

/*This circuit template checks that c is the multiplication of a and b.*/

template Multiplier2 () {

   // Declaration of signals.
   signal input a;
   signal input b;
   signal output c;

   // Constraints.
   c <== a * b;
}
component main{public [b]} = Multiplier2();
```

This circuit proves that we know two numbers that factor the output number c. We also reveal one of the numbers we used to factor c. This is not really impressive, but we stick to the classics for explanations! Copy the code and put it in a file named `multiplier2.circom`.

## Compile the Circuit

In the first step, we compile an `.r1cs` file using Circom and create a verification/proving key using SnarkJS. To compile the `.r1cs` file open your terminal (after installing Circom) and type:

```bash
circom multiplier2.circom --r1cs
```

You will find a file called `multiplier2.r1cs` in your working folder. To create the keys you can either follow the Circom documentation, or download the two keys from our [GitHub](https://github.com/TaceoLabs/collaborative-circom/tree/main/collaborative-circom/examples/test_vectors/multiplier2), where we created the keys already (you will need `multiplier2.zkey` and `verification_key.json`).

## Split the Input

Ok, after we finished the setup, we need to prepare the inputs for the witness extension. If you have read the Circom documentation (or used Circom in the past), you will remember a step between compiling the circuits and the actual proving. That is, the witness extension (or "computing the witness" as Circom calls it).

We prepare an input file and call it `input.json`:

```json
{"a": "3", "b": "11"}
```

> Remember that `b` is a public input, as defined by our circuit.

As we want to execute an MPC protocol, we have to split the input for the parties. At the moment we support 3 parties for the witness extension. To do that, execute the following command:

```bash
$ mkdir out

$ co-circom split-input --circuit multiplier2.circom --input input.json --protocol REP3 --curve BN254 --out-dir out/
INFO co_circom: 275: Wrote input share 0 to file out/input.json.0.shared
INFO co_circom: 275: Wrote input share 1 to file out/input.json.1.shared
INFO co_circom: 275: Wrote input share 2 to file out/input.json.2.shared
INFO co_circom: 277: Split input into shares successfully

$ ls out/
input.json.0.shared
input.json.1.shared
input.json.2.shared
```

This command secret shares the private inputs (everything that is not explicitly public) and creates a `.json` file for each of the three parties, containing the shared and the public values.

## Witness Extension

Now we have to compute the extended witness. In a real-world setting you would have to send the input files from the previous step to the parties.

To achieve that we need another config file for every party, namely the network config (you can read an in-depth explanation about the config at [here](./network-config.md)). You can copy-paste the config from here and call it `party0.toml` for party0 and so on:

```toml
my_id = 0
bind_addr = "0.0.0.0:10000"
key_path = "data/key0.der"
[[parties]]
id = 0
dns_name = "localhost:10000"
cert_path = "data/cert0.der"
[[parties]]
id = 1
dns_name = "localhost:10001"
cert_path = "data/cert1.der"
[[parties]]
id = 2
dns_name = "localhost:10002"
cert_path = "data/cert2.der"
```

You can download the TLS certificates from our [GitHub](https://github.com/TaceoLabs/collaborative-circom/tree/a37d24abcc1069e6e30b0e6638870bcd8834451a/collaborative-circom/examples/data) and put them under `data/`.

We move the `.toml` files to `configs/` and execute the following command (for every party).

```bash
$ co-circom generate-witness --input out/input.json.0.shared --circuit multiplier2.circom --protocol REP3 --curve BN254 --config configs/party0.toml --out out/witness.wtns.0.shared

INFO co_circom: 365: Witness successfully written to out/witness.wtns.0.shared
```

> For brevity we only showed the command for a the 0-th party. You have to call it for all three parties in parallel.

After all parties finished successfully, you will have three witness files in your `out/` folder. Each one of them contains a share of the extended witness.

## Prove the Circuit

We need another MPC step to finally get our co-SNARK Groht16 proof. We can reuse TLS certificates and the network config from the previous step. Also, we finally need the proving key from the very first step! In your terminal execute the following command:

```bash
$ co-circom generate-proof groth16 --witness out/witness.wtns.0.shared --zkey multiplier2.zkey --protocol REP3 --curve BN254 --config configs/party0.toml --out proof.0.json --public-input public_input.json

INFO co_circom: 418: Wrote proof to file proof.0.json
INFO co_circom: 438: Proof generation finished successfully
```

> Again, for brevity, we only gave the command for party 0. You know the drill, all at the same time.

The three proofs produced by the separate parties are equivalent and valid Groth16 proofs - Congratulations, you did it 🎉

You will find another file, namely `public_input.json`. This file contains all public information necessary to verify the proof, which, in our case, means:

```json
["33","11"]
```

This is the factored number and the public input `b`.

## Verify the Proof

To verify we can either use snarkjs or the `co-circom` binary.

```bash
$ co-circom verify groth16 --proof proof.0.json --vk verification_key.json --public-input public_input.json --curve BN254
co_circom: 483: Proof verified successfully

$ snarkjs groth16 verify verification_key.json public_input.json proof.0.json
[INFO]  snarkJS: OK!
```

For a full `shell` script executing all of the commands at once, have a look at our [GitHub](collaborative-circom/examples). In this folder you find this exact example, and some more.

**And now you can dive into the rest of the book** 🦀

<!-- knowledge of a pre-image to a Poseidon[^1] hash.

This circuit computes the Poseidon hash of one field element. You will need the circom standard library (or at least the relevant Poseidon files) located at `libs/` (download it [here](https://github.com/iden3/circomlib/tree/master/circuits)).
[^1]: Poseidon: [https://eprint.iacr.org/2019/458.pdf](https://eprint.iacr.org/2019/458.pdf) -->
