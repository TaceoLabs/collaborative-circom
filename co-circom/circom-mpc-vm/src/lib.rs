#![warn(missing_docs)]
//! Defines an [MPC-VM](mpc_vm::WitnessExtension) (Multiparty Computation Virtual Machine) to perform circom's [witness extension step](https://docs.circom.io/getting-started/computing-the-witness/) for a Groth16 proof.
//!
//! This VM is specifically tailored for the witness extension and is not a generic MPC-VM capable of running arbitrary code. The VM is stack-based (two different stacks for field elements and indices)
//! and, mirroring circom's design, has two "RAM-like" structures in the form of two consecutive chunks of memory (`Vec`), representing the [Signals](https://docs.circom.io/circom-language/signals/)
//! and [Vars](https://docs.circom.io/circom-language/variables-and-mutability/).
//!
//! When running the MPC-VM, the output will be a [`SharedWitness`](co_circom_snarks::SharedWitness), constructed from the `Vec` of Signals mentioned above.
//!
//! Currently, we only support a [semi-honest 3-party replicated secret-sharing](https://eprint.iacr.org/2018/403.pdf) protocol, allowing for easy switching between
//! Arithmetic (A) and Binary (B) Shares, which is necessary for circom's witness extension. The current implementation of the MPC-VM is somewhat naive with respect to
//! run-time optimization. We eagerly communicate after every non-linear operation and perform many unnecessary conversions between A and B shares.
//!
//! Major changes and optimizations are expected in the near future.

mod accelerator;
/// This module contains the MPC-VM witness extension trait
pub mod mpc;
/// This module contains the MPC-VM
pub mod mpc_vm;
/// Defines the bytecode for the MPC-VM
pub mod op_codes;
mod stack;
/// Defines the types for the MPC-VM, including [template declaration](types::TemplateDecl) and [function declarations](types::FunDecl).
pub mod types;

pub use mpc::rep3::Rep3VmType;
