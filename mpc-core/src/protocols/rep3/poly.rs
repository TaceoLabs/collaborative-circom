//! Polynomial Evaluation
//!
//! This module contains functions to evaluate shared polynomials

use rayon::prelude::*;
use std::cmp::max;

use ark_ff::PrimeField;

use super::{arithmetic, Rep3PrimeFieldShare};

type FieldShare<F> = Rep3PrimeFieldShare<F>;

//Same as ark_poly
const MIN_ELEMENTS_PER_THREAD: usize = 16;

// This is copied from
// https://docs.rs/ark-poly/latest/src/ark_poly/polynomial/univariate/dense.rs.html#56
//
// The DensePolynomial implementation expects a Field, therefore we cannot use it.
// Therefore we copy it and call the respective rep3 operations

// Horner's method for polynomial evaluation
// Important note: We do not use the rep3 add/muls as we just want to
// do both a/b in parallel. Usually, we split a/b and do it once on a and once on b. So
// here we abuse the additions and multiplications to not have to split the Shares and
// do it in one run
fn horner_evaluate<F: PrimeField>(poly_coeffs: &[FieldShare<F>], point: F) -> FieldShare<F> {
    poly_coeffs
        .iter()
        .rfold(FieldShare::zero_share(), move |result, coeff| {
            arithmetic::add(arithmetic::mul_public(result, point), *coeff)
        })
}

// This is copied from
// https://docs.rs/ark-poly/latest/src/ark_poly/polynomial/univariate/dense.rs.html#56
/// Evaluate the shared polynomial at a public point.
pub fn eval_poly<F: PrimeField>(coeffs: &[FieldShare<F>], point: F) -> FieldShare<F> {
    if point.is_zero() {
        return coeffs[0];
    }

    // Horners method - parallel method
    // compute the number of threads we will be using.
    // TODO investigate how this behaves if we are in a rayon scope. Does this return all
    // free threads or also the busy ones? Because then the chunks size is wrong...
    let num_cpus_available = rayon::current_num_threads();
    let num_coeffs = coeffs.len();
    let num_elem_per_thread = max(num_coeffs / num_cpus_available, MIN_ELEMENTS_PER_THREAD);

    // run Horners method on each thread as follows:
    // 1) Split up the coefficients across each thread evenly.
    // 2) Do polynomial evaluation via horner's method for the thread's coefficeints
    // 3) Scale the result point^{thread coefficient start index}
    // Then obtain the final polynomial evaluation by summing each threads result.
    let result = coeffs
        .par_chunks(num_elem_per_thread)
        .enumerate()
        .map(|(i, chunk)| {
            let mut thread_result = horner_evaluate(chunk, point);
            let power = point.pow([(i * num_elem_per_thread) as u64]);
            thread_result.a *= power;
            thread_result.b *= power;
            thread_result
        })
        .reduce(FieldShare::zero_share, |acc, e| arithmetic::add(acc, e));
    result
}
