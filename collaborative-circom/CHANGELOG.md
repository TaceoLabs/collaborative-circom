# Changelog

## [0.3.0](https://github.com/TaceoLabs/collaborative-circom/compare/collaborative-circom-v0.2.1...collaborative-circom-v0.3.0) (2024-08-09)


### ⚠ BREAKING CHANGES

* share_rep3 and share_shamir interfaces changed
* new config implementation, config option to allow leaking of secret values in logs ([#132](https://github.com/TaceoLabs/collaborative-circom/issues/132))
* Adds a method to the ArkworksPairingBridge trait
* the function signature of the two run methods of the witness extension now changed. To retrieve the shared witness now another call `into_shared_witness()` is necessary.
* Add the possibility to specify another curve in the co-circom binary
* Add possibility to use Shamir secret sharing for proofing and

### Features

* add deserialization of plonk circom types ([d1f0d4d](https://github.com/TaceoLabs/collaborative-circom/commit/d1f0d4dd5ac63e85523c139e573161bd2ff0061a))
* Add possibility to use Shamir secret sharing for proofing and ([6205475](https://github.com/TaceoLabs/collaborative-circom/commit/6205475b78d4654c61f5058befe5d5990da19432))
* Add the possibility to specify another curve in the co-circom binary ([fdd6bf2](https://github.com/TaceoLabs/collaborative-circom/commit/fdd6bf2f5274da790fd7cbe09ee48563b404d153))
* can now retrieve certain outputs after witness extension by name ([d9e3399](https://github.com/TaceoLabs/collaborative-circom/commit/d9e33996d10cea5f8197d507a13ed9087f0f4c20))
* now co-circom supports hex values ([d004d10](https://github.com/TaceoLabs/collaborative-circom/commit/d004d10b8a9b5c39e77abd37c8b862107aaa14c1))
* plonk support ([9b65797](https://github.com/TaceoLabs/collaborative-circom/commit/9b6579724f6f5ba4fc6af8a98d386b96818dc08b))


### Code Refactoring

* new config implementation, config option to allow leaking of secret values in logs ([#132](https://github.com/TaceoLabs/collaborative-circom/issues/132)) ([964b04f](https://github.com/TaceoLabs/collaborative-circom/commit/964b04f47e8d491ae140cb7c10c596e1c40b6b5c))
* share_rep3 and share_shamir interfaces changed ([5e7420f](https://github.com/TaceoLabs/collaborative-circom/commit/5e7420f95a46466304c2ab80de2069c2feb3432d))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * circom-types bumped from 0.3.0 to 0.4.0
    * circom-mpc-compiler bumped from 0.3.0 to 0.4.0
    * circom-mpc-vm bumped from 0.2.0 to 0.3.0
    * collaborative-groth16 bumped from 0.2.1 to 0.3.0
    * mpc-core bumped from 0.2.1 to 0.3.0

## [0.2.1](https://github.com/TaceoLabs/collaborative-circom/compare/collaborative-circom-v0.2.0...collaborative-circom-v0.2.1) (2024-07-10)


### Bug Fixes

* better handling of ipv4 and ipv6 in networking ([#119](https://github.com/TaceoLabs/collaborative-circom/issues/119)) ([090227d](https://github.com/TaceoLabs/collaborative-circom/commit/090227d372215e9459c06777064b04ec4865bdb6))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * circom-types bumped from 0.2.0 to 0.3.0
    * circom-mpc-compiler bumped from 0.2.0 to 0.3.0
    * collaborative-groth16 bumped from 0.2.0 to 0.2.1
    * mpc-core bumped from 0.2.0 to 0.2.1

## [0.2.0](https://github.com/TaceoLabs/collaborative-circom/compare/collaborative-circom-v0.1.2...collaborative-circom-v0.2.0) (2024-07-09)


### Features

* added a lot tracings ([c4f24d1](https://github.com/TaceoLabs/collaborative-circom/commit/c4f24d15f0c7af0560fbffe4a4aaedda2fa515e8))
* added kyc example ([8b45982](https://github.com/TaceoLabs/collaborative-circom/commit/8b4598239fec55f4a4f6d87dfe12ea4aca19fddb))


### Bug Fixes

* now writes 0 instead of empty string when public inputs is zero ([eca6676](https://github.com/TaceoLabs/collaborative-circom/commit/eca667608774484733925632358dedd6608d318b))
* pushed input.json for kyc ([be46bc2](https://github.com/TaceoLabs/collaborative-circom/commit/be46bc28c3ff28a135754a72664ba5732b413345))
* pushed ver_key.json for kyc ([96d745d](https://github.com/TaceoLabs/collaborative-circom/commit/96d745df27894b81b391c36d957e0b2ab66b16d1))


### Dependencies

* The following workspace dependencies were updated
  * dependencies
    * circom-types bumped from 0.1.0 to 0.2.0
    * circom-mpc-compiler bumped from 0.1.0 to 0.2.0
    * collaborative-groth16 bumped from 0.1.0 to 0.2.0
    * mpc-core bumped from 0.1.0 to 0.2.0

## [0.1.2](https://github.com/TaceoLabs/collaborative-circom/compare/collaborative-circom-v0.1.1...collaborative-circom-v0.1.2) (2024-06-14)


### Bug Fixes

* delete not needed impl ([f4dc747](https://github.com/TaceoLabs/collaborative-circom/commit/f4dc74770cd92531851b777dfaa1385b033b657c))

## [0.1.1](https://github.com/TaceoLabs/collaborative-circom/compare/collaborative-circom-v0.1.0...collaborative-circom-v0.1.1) (2024-06-14)


### Bug Fixes

* minor spelling ([e7f7178](https://github.com/TaceoLabs/collaborative-circom/commit/e7f7178b7e3c2605e0295a591eaa8b6e46499db1))

## [0.1.0](https://github.com/TaceoLabs/collaborative-circom/compare/collaborative-circom-v0.0.1...collaborative-circom-v0.1.0) (2024-06-14)


### Features

* First impl for Circom Parser to MPC.  ([#26](https://github.com/TaceoLabs/collaborative-circom/issues/26)) ([779682d](https://github.com/TaceoLabs/collaborative-circom/commit/779682d7d824d782109db8b4584604b23637dad7))
* first version of command line interface ([#36](https://github.com/TaceoLabs/collaborative-circom/issues/36)) ([6abe716](https://github.com/TaceoLabs/collaborative-circom/commit/6abe716268f1e165cdae07a10f4d2dafd010cc04))
* first version of mpc vm ([#42](https://github.com/TaceoLabs/collaborative-circom/issues/42)) ([6dcd5f4](https://github.com/TaceoLabs/collaborative-circom/commit/6dcd5f4ce7c8431b94dd7262a4219a3a63efd702))
* fixed poseidonex_test ([f119394](https://github.com/TaceoLabs/collaborative-circom/commit/f1193948e1edbed19be7d9684b6f96a0e83d3045))
* generate public input json file during proving ([2d20229](https://github.com/TaceoLabs/collaborative-circom/commit/2d20229e4e614354071cdb20a6207725b597fa28)), closes [#70](https://github.com/TaceoLabs/collaborative-circom/issues/70)
* integrate witness extension via MPC VM into CLI binary ([f526081](https://github.com/TaceoLabs/collaborative-circom/commit/f526081a01e3faa6b48fb463f3690f968218a1a4))
* make protocol argument in CLI an enum ([1c025a1](https://github.com/TaceoLabs/collaborative-circom/commit/1c025a1ef612603b31062c6a56b6dd15a0917c9e))
* negative numbers work, also added an example with negative numbers ([#80](https://github.com/TaceoLabs/collaborative-circom/issues/80)) ([1a54649](https://github.com/TaceoLabs/collaborative-circom/commit/1a54649ee859f2492a225ae3647f795852d7e368))
* network configuration structure + parsing ([7f1cb06](https://github.com/TaceoLabs/collaborative-circom/commit/7f1cb0645fe7d6319367f8846e5e5d05b7ff2ae9))
* public inputs support ([#76](https://github.com/TaceoLabs/collaborative-circom/issues/76)) ([07cf260](https://github.com/TaceoLabs/collaborative-circom/commit/07cf26007285822ba42e8dce2439f676a2cf08ef))
* skeleton of cli interface for coll-circom ([807a822](https://github.com/TaceoLabs/collaborative-circom/commit/807a822ac4b33c16540a32f4a34f7703b0fc134e))
* support merging shared inputs from multiple parties + examples ([#75](https://github.com/TaceoLabs/collaborative-circom/issues/75)) ([1168488](https://github.com/TaceoLabs/collaborative-circom/commit/11684884b3e1d2be6309fd98e1603626d5e58c93))
* use existing information in zkey to not require regeneration of matrices ([c7b75c3](https://github.com/TaceoLabs/collaborative-circom/commit/c7b75c34e69479bea06583e9fc17f3b1dc8f3d9a))


### Bug Fixes

* fix wrong argument name in example runner script ([244a72e](https://github.com/TaceoLabs/collaborative-circom/commit/244a72eeda2d6d8d31f8e9bc493565e076a12fc2))
