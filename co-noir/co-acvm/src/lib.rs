//![warn(missing_docs)]

mod mpc;
pub mod solver;

pub use mpc::plain::PlainAcvmSolver;
pub use mpc::rep3::{Rep3AcvmSolver, Rep3AcvmType};
