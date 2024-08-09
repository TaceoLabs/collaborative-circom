# verify

The aim of the `verify` command is to verify a Groth16 Circom proof using the provided verification key and public inputs.

## Example

```bash
co-circom verify --proof proof.json --vk test_vectors/multiplier2/verification_key.json --public-input public_input.json --curve BN254
```

The above command verifies the proof in `proof.json` using the verification key `test_vectors/multiplier2/verification_key.json` and public input `public_input.json`.

## Reference

```txt
Verification of a Circom proof

Usage: co-circom verify [OPTIONS] --config <CONFIG> --proof <PROOF> --vk <VK> --public-input <PUBLIC_INPUT> <PROOFSYSTEM>

Arguments:
  <PROOFSYSTEM>  [possible values: groth16, plonk]

Options:
      --config <CONFIG>              The path to the config file
      --proof <PROOF>                The path to the proof file
      --curve <CURVE>                The pairing friendly curve to be used [possible values: BN254, BLS12-381]
      --vk <VK>                      The path to the verification key file
      --public-input <PUBLIC_INPUT>  The path to the public input JSON file
  -h, --help                         Print help (see more with '--help')
```
