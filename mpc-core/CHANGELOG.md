# Changelog



## [0.3.0](https://github.com/TaceoLabs/collaborative-circom/compare/mpc-core-v0.2.1...mpc-core-v0.3.0) (2024-08-09)


### ⚠ BREAKING CHANGES

* Added functionality for MPC traits
* Add possibility to use Shamir secret sharing for proofing and

### Features

* Add possibility to use Shamir secret sharing for proofing and ([6205475](https://github.com/TaceoLabs/collaborative-circom/commit/6205475b78d4654c61f5058befe5d5990da19432))
* Added functionality for MPC traits ([0897066](https://github.com/TaceoLabs/collaborative-circom/commit/089706629ab863814276309b330744f909c976cd))

## [0.2.0](https://github.com/TaceoLabs/collaborative-circom/compare/mpc-core-v0.1.0...mpc-core-v0.2.0) (2024-07-09)


### ⚠ BREAKING CHANGES

* clean up visibility of interfaces and add docs

### Features

* expose promot_from_trivial for Rep3 ([b329a42](https://github.com/TaceoLabs/collaborative-circom/commit/b329a427031d4e787addbf37902c710ba0132ccf))


### Bug Fixes

* implement optional post-processing permutation for FFTs and correct root of unity calculation to match circom output ([5ab3292](https://github.com/TaceoLabs/collaborative-circom/commit/5ab329294959c85ea6e0823cbe651ba6efa747f8))


### Code Refactoring

* clean up visibility of interfaces and add docs ([8a4f085](https://github.com/TaceoLabs/collaborative-circom/commit/8a4f08582a950d11f88e1de8fb6c4e28279b2891))

## [0.1.0](https://github.com/TaceoLabs/collaborative-circom/compare/mpc-core-v0.0.1...mpc-core-v0.1.0) (2024-06-14)


### ⚠ BREAKING CHANGES

* remove slice and slicemut types from traits

### Features

* Add a bridge to translate Rep3 shares to Shamir shares ([7855e8d](https://github.com/TaceoLabs/collaborative-circom/commit/7855e8dc65c40b2f9b4da6e8e35aa5269fc1ab11))
* Add arithmetic&lt;-&gt;binary conversions for field elements in ABY3 ([#39](https://github.com/TaceoLabs/collaborative-circom/issues/39)) ([190622d](https://github.com/TaceoLabs/collaborative-circom/commit/190622def82fac9e17a9d5ce75cffc3fbad9bbaa))
* add bool_or opcode ([7018352](https://github.com/TaceoLabs/collaborative-circom/commit/7018352f1a8eba34e3abcdea9cf00ac6ff77d846))
* Add semi-honest GSZ (i.e., Shamir) ([#21](https://github.com/TaceoLabs/collaborative-circom/issues/21)) ([746edb4](https://github.com/TaceoLabs/collaborative-circom/commit/746edb45d14da4d8a54f7503848a4e62e2bfc899))
* added collaborative groth16 prover ([#18](https://github.com/TaceoLabs/collaborative-circom/issues/18)) ([6e5bb98](https://github.com/TaceoLabs/collaborative-circom/commit/6e5bb98afa5be816188bc019036ba4786f448749))
* added pow/mod opcodes for plain VM ([#50](https://github.com/TaceoLabs/collaborative-circom/issues/50)) ([eb6e55c](https://github.com/TaceoLabs/collaborative-circom/commit/eb6e55c5fdf5e650ba7cdab52acab0b4af392615))
* added skeleton for mpc/collab-groth16 ([#12](https://github.com/TaceoLabs/collaborative-circom/issues/12)) ([9c03331](https://github.com/TaceoLabs/collaborative-circom/commit/9c03331171429f061ead8cddda292cd97d498f1a))
* bool_or aby3 ([710b2af](https://github.com/TaceoLabs/collaborative-circom/commit/710b2affc684595654b2dcad788ac3dcd3f2730f))
* change plain semantics of comparisons for easier translation to mpc ([660656b](https://github.com/TaceoLabs/collaborative-circom/commit/660656bf093cfa8b79de3d60db7af4b8f1422311))
* comparison impls ([d470c82](https://github.com/TaceoLabs/collaborative-circom/commit/d470c82be3570e4b2103fe0f34d686a42561db41))
* Creating Proof from assignment in MPC ([#17](https://github.com/TaceoLabs/collaborative-circom/issues/17)) ([cc4f6f5](https://github.com/TaceoLabs/collaborative-circom/commit/cc4f6f5de873fea80bd51a724dbd001d6351f68d))
* First impl for Circom Parser to MPC.  ([#26](https://github.com/TaceoLabs/collaborative-circom/issues/26)) ([779682d](https://github.com/TaceoLabs/collaborative-circom/commit/779682d7d824d782109db8b4584604b23637dad7))
* first version of command line interface ([#36](https://github.com/TaceoLabs/collaborative-circom/issues/36)) ([6abe716](https://github.com/TaceoLabs/collaborative-circom/commit/6abe716268f1e165cdae07a10f4d2dafd010cc04))
* first version of mpc vm ([#42](https://github.com/TaceoLabs/collaborative-circom/issues/42)) ([6dcd5f4](https://github.com/TaceoLabs/collaborative-circom/commit/6dcd5f4ce7c8431b94dd7262a4219a3a63efd702))
* fixed poseidonex_test ([f119394](https://github.com/TaceoLabs/collaborative-circom/commit/f1193948e1edbed19be7d9684b6f96a0e83d3045))
* implement a basic variant of shift_right for ABY3 ([e17fc61](https://github.com/TaceoLabs/collaborative-circom/commit/e17fc614b358eeb884e56121154c98968c0e0ce4))
* implement eq/neq in mpc vm ([1e32551](https://github.com/TaceoLabs/collaborative-circom/commit/1e3255108578635ac869a564a6fcf5fab854fb03))
* implement mod and pow for public values in MPC VM, as well as pow of secret with public values ([75c70dd](https://github.com/TaceoLabs/collaborative-circom/commit/75c70dd096ab87ccadd59cc279b25dd1dc0d191c))
* implement more shared bit operations ([9e326de](https://github.com/TaceoLabs/collaborative-circom/commit/9e326de158b5219e7e93311d2357e74b597272c7))
* implement shift right for public shift values ([7db3730](https://github.com/TaceoLabs/collaborative-circom/commit/7db3730d02624ec2f28dfd9d93f6bac174b88ff6))
* implemented plain/aby3 cmux and bool not for shared if handling ([e5701aa](https://github.com/TaceoLabs/collaborative-circom/commit/e5701aa8d967ab9d111556c8dfba3eeacfda4782))
* integrate witness extension via MPC VM into CLI binary ([f526081](https://github.com/TaceoLabs/collaborative-circom/commit/f526081a01e3faa6b48fb463f3690f968218a1a4))
* make all comparisons based on LT ([6e8cbbf](https://github.com/TaceoLabs/collaborative-circom/commit/6e8cbbf5bf5472ef1eb7ffe3dfce89ed790f508b))
* mpc accelerator first draft ([#79](https://github.com/TaceoLabs/collaborative-circom/issues/79)) ([5f2709b](https://github.com/TaceoLabs/collaborative-circom/commit/5f2709b2e56277328180f9990f1f21c77cdac06e))
* naively implemented plain val() for bool expr ([3e6cabb](https://github.com/TaceoLabs/collaborative-circom/commit/3e6cabb0b3fa9b903f7d747393347f531d6516bf))
* public inputs support ([#76](https://github.com/TaceoLabs/collaborative-circom/issues/76)) ([07cf260](https://github.com/TaceoLabs/collaborative-circom/commit/07cf26007285822ba42e8dce2439f676a2cf08ef))
* shared control flow test working for single return values ([6f4aabb](https://github.com/TaceoLabs/collaborative-circom/commit/6f4aabb3a842d47e148343a6b5e0c5b6d27b9b31))
* shared_control_flow arrays working except loops ([15cdecf](https://github.com/TaceoLabs/collaborative-circom/commit/15cdecf5d4dc6d0400367856a48f2571925745c3))
* some more aby3 impls for bitops ([ff0ed61](https://github.com/TaceoLabs/collaborative-circom/commit/ff0ed612a0616fe864f85242956ec1ede0f76236))
* traits and implementations of aby3 shares, fft's, and msm ([#16](https://github.com/TaceoLabs/collaborative-circom/issues/16)) ([a6bf90e](https://github.com/TaceoLabs/collaborative-circom/commit/a6bf90e6d326df3e9caa2dbbabf7bd60acb50fbd))
* use CanonicalSerialize from ark for ser/de of arkworks structures, with additional serde compat layer on top level ([e3e7af3](https://github.com/TaceoLabs/collaborative-circom/commit/e3e7af340d1fbfc148fbe6614b004a8c70aba1f0))
* VM if logic first draft ([cb9e525](https://github.com/TaceoLabs/collaborative-circom/commit/cb9e525e8ff4d96fb18a73a59589c09fcb756dff))
* warning on opening shared values ([a276601](https://github.com/TaceoLabs/collaborative-circom/commit/a276601150bfc81c97c7e5c714be23cbabdbc1ba))


### Bug Fixes

* ab3 is_shared function + fixed a typo in cmux ([c6e4576](https://github.com/TaceoLabs/collaborative-circom/commit/c6e4576ac22de7569a6433e2dc862783c3bb02e2))
* broken impl of shifts ([0413ca0](https://github.com/TaceoLabs/collaborative-circom/commit/0413ca0af39f6786c6caca402311b56e5c5bccc6))
* correct handling of is_zero in binary MPC protocol ([432326e](https://github.com/TaceoLabs/collaborative-circom/commit/432326e9f2c24bca7a3a2f795711d677d1d37503))
* fix a2b for zero case ([53592b1](https://github.com/TaceoLabs/collaborative-circom/commit/53592b12756efa9c63c6e5185cafc13c2bd4154d))
* fixed array as paramters and return val for functions (escalarmulw4table_test) ([8f38648](https://github.com/TaceoLabs/collaborative-circom/commit/8f386487a40de20951d2124ed10d2ee76876e9bd))
* fixed iszero for aby3 ([244072a](https://github.com/TaceoLabs/collaborative-circom/commit/244072a1c5f98501dc8ba8003684db792fda92db))
* mixup for shift left and right ([a2757fc](https://github.com/TaceoLabs/collaborative-circom/commit/a2757fc63222b4a37d991533a0980e9be55023f3))
* only send what is needed in the AND gates of is_zero ([a84d36d](https://github.com/TaceoLabs/collaborative-circom/commit/a84d36d2c964f876527e9f9755a063bcef021eb0))


### Code Refactoring

* remove slice and slicemut types from traits ([d60cd06](https://github.com/TaceoLabs/collaborative-circom/commit/d60cd0645a397898cd997a516dc513c7f29ecb55))
