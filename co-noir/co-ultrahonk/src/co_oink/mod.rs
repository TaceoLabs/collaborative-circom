pub(crate) mod prover;
pub(crate) mod types;

// execute_log_derivative_inverse_round: n
pub const CRAND_PAIRS_FACTOR_N: usize = 1;
// execute_grand_product_computation_round:
// 	compute_grand_product:
// 		4 * batched_grand_product_num_denom: (n - 1)
// 		2 * array_prod_mul: (n - 1) * 4 + 2
// 		misc: (n - 1) * 4
pub const CRAND_PAIRS_FACTOR_N_MINUS_ONE: usize = 4 + 8 + 4;
pub const CRAND_PAIRS_CONST: usize = 4;
