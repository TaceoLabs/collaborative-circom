use ark_ec::pairing::Pairing;
use ark_ff::Field;
use std::array;
use ultrahonk::prelude::{Barycentric, Univariate};

use crate::mpc::NoirUltraHonkProver;

pub(crate) struct SharedUnivariate<T: NoirUltraHonkProver<P>, P: Pairing, const SIZE: usize> {
    pub(crate) evaluations: [T::ArithmeticShare; SIZE],
}

impl<T: NoirUltraHonkProver<P>, P: Pairing, const SIZE: usize> SharedUnivariate<T, P, SIZE> {
    pub(crate) fn from_vec(vec: &[T::ArithmeticShare]) -> Self {
        assert_eq!(vec.len(), SIZE);
        let mut res = Self::default();
        res.evaluations.clone_from_slice(vec);
        res
    }

    pub(crate) fn scale_inplace(&mut self, driver: &mut T, rhs: P::ScalarField) {
        for i in 0..SIZE {
            self.evaluations[i] = driver.mul_with_public(rhs, self.evaluations[i]);
        }
    }

    pub(crate) fn neg(&self, driver: &mut T) -> Self {
        let mut result = Self::default();
        for i in 0..SIZE {
            result.evaluations[i] = driver.neg(self.evaluations[i]);
        }
        result
    }

    pub(crate) fn scale(&self, driver: &mut T, rhs: P::ScalarField) -> Self {
        let mut result = Self::default();
        for i in 0..SIZE {
            result.evaluations[i] = driver.mul_with_public(rhs, self.evaluations[i]);
        }
        result
    }

    pub(crate) fn add_scalar(&self, driver: &mut T, rhs: P::ScalarField) -> Self {
        let mut result = Self::default();
        for i in 0..SIZE {
            result.evaluations[i] = driver.add_with_public(rhs, self.evaluations[i]);
        }
        result
    }

    pub(crate) fn sub_scalar(&self, driver: &mut T, rhs: P::ScalarField) -> Self {
        let neg = -rhs;
        let mut result = Self::default();
        for i in 0..SIZE {
            result.evaluations[i] = driver.add_with_public(neg, self.evaluations[i]);
        }
        result
    }

    pub(crate) fn add(&self, driver: &mut T, rhs: &Self) -> Self {
        let mut result = Self::default();
        for i in 0..SIZE {
            result.evaluations[i] = driver.add(self.evaluations[i], rhs.evaluations[i]);
        }
        result
    }

    pub(crate) fn sub(&self, driver: &mut T, rhs: &Self) -> Self {
        let mut result = Self::default();
        for i in 0..SIZE {
            result.evaluations[i] = driver.sub(self.evaluations[i], rhs.evaluations[i]);
        }
        result
    }

    pub(crate) fn add_public(
        &self,
        driver: &mut T,
        rhs: &Univariate<P::ScalarField, SIZE>,
    ) -> Self {
        let mut result = Self::default();
        for i in 0..SIZE {
            result.evaluations[i] = driver.add_with_public(rhs.evaluations[i], self.evaluations[i]);
        }
        result
    }

    pub(crate) fn sub_public(
        &self,
        driver: &mut T,
        rhs: &Univariate<P::ScalarField, SIZE>,
    ) -> Self {
        let mut result = Self::default();
        for i in 0..SIZE {
            result.evaluations[i] =
                driver.add_with_public(-rhs.evaluations[i], self.evaluations[i]);
        }
        result
    }

    pub(crate) fn add_assign(&mut self, driver: &mut T, rhs: &Self) {
        for i in 0..SIZE {
            self.evaluations[i] = driver.add(self.evaluations[i], rhs.evaluations[i]);
        }
    }

    pub(crate) fn mul_public(
        &self,
        driver: &mut T,
        rhs: &Univariate<P::ScalarField, SIZE>,
    ) -> Self {
        let mut result = Self::default();
        for i in 0..SIZE {
            result.evaluations[i] = driver.mul_with_public(rhs.evaluations[i], self.evaluations[i]);
        }
        result
    }

    pub(crate) fn vec_to_univariates(vec: &[T::ArithmeticShare]) -> Vec<Self> {
        assert_eq!(vec.len() % SIZE, 0);
        vec.chunks(SIZE)
            .map(|chunk| {
                let mut evaluations = array::from_fn(|_| T::ArithmeticShare::default());
                evaluations.clone_from_slice(chunk);
                Self { evaluations }
            })
            .collect()
    }

    pub(crate) fn univariates_to_vec(univariates: &[Self]) -> Vec<T::ArithmeticShare> {
        univariates
            .iter()
            .flat_map(|univariate| univariate.evaluations.to_owned())
            .collect()
    }

    pub(crate) fn extend_and_batch_univariates<const SIZE2: usize>(
        &self,
        driver: &mut T,
        result: &mut SharedUnivariate<T, P, SIZE2>,
        extended_random_poly: &Univariate<P::ScalarField, SIZE2>,
        partial_evaluation_result: &P::ScalarField,
        linear_independent: bool,
    ) {
        let mut extended = SharedUnivariate::<T, P, SIZE2>::default();
        extended.extend_from(driver, &self.evaluations);

        if linear_independent {
            let tmp = extended_random_poly.to_owned() * partial_evaluation_result;
            let tmp = extended.mul_public(driver, &tmp);
            result.add_assign(driver, &tmp);
        } else {
            result.add_assign(driver, &extended);
        }
    }

    /**
     * @brief Given a univariate f represented by {f(domain_start), ..., f(domain_end - 1)}, compute the
     * evaluations {f(domain_end),..., f(extended_domain_end -1)} and return the Univariate represented by
     * {f(domain_start),..., f(extended_domain_end -1)}
     *
     * @details Write v_i = f(x_i) on a the domain {x_{domain_start}, ..., x_{domain_end-1}}. To efficiently
     * compute the needed values of f, we use the barycentric formula
     *      - f(x) = B(x) Σ_{i=domain_start}^{domain_end-1} v_i / (d_i*(x-x_i))
     * where
     *      - B(x) = Π_{i=domain_start}^{domain_end-1} (x-x_i)
     *      - d_i  = Π_{j ∈ {domain_start, ..., domain_end-1}, j≠i} (x_i-x_j) for i ∈ {domain_start, ...,
     * domain_end-1}
     *
     * When the domain size is two, extending f = v0(1-X) + v1X to a new value involves just one addition
     * and a subtraction: setting Δ = v1-v0, the values of f(X) are f(0)=v0, f(1)= v0 + Δ, v2 = f(1) + Δ, v3
     * = f(2) + Δ...
     *
     */
    pub fn extend_from(&mut self, driver: &mut T, poly: &[T::ArithmeticShare]) {
        let length = poly.len();
        let extended_length = SIZE;

        // TACEO TODO add assign could be used here often

        assert!(length <= extended_length);
        self.evaluations[..length].clone_from_slice(poly);

        if length == 2 {
            let delta = driver.sub(poly[1], poly[0]);
            for i in length..extended_length {
                self.evaluations[i] = driver.add(self.evaluations[i - 1], delta);
            }
        } else if length == 3 {
            // Based off https://hackmd.io/@aztec-network/SyR45cmOq?type=view
            // The technique used here is the same as the length == 3 case below.
            let inverse_two = P::ScalarField::from(2u64).inverse().unwrap();
            let tmp = driver.add(poly[0], poly[2]);
            let tmp = driver.mul_with_public(inverse_two, tmp);
            let a = driver.sub(tmp, poly[1]);

            let tmp = driver.sub(poly[1], poly[0]);
            let b = driver.sub(tmp, a);

            let a2 = driver.add(a, a);
            let mut a_mul = a2.to_owned();
            for _ in 0..length - 2 {
                a_mul = driver.add(a_mul, a2);
            }

            let tmp = driver.add(a_mul, a);
            let mut extra = driver.add(tmp, b);
            for i in length..extended_length {
                self.evaluations[i] = driver.add(self.evaluations[i - 1], extra);
                extra = driver.add(extra, a2);
            }
        } else if length == 4 {
            // To compute a barycentric extension, we can compute the coefficients of the univariate.
            // We have the evaluation of the polynomial at the domain (which is assumed to be 0, 1, 2, 3).
            // Therefore, we have the 4 linear equations from plugging into f(x) = ax^3 + bx^2 + cx + d:
            //          a*0 + b*0 + c*0 + d = f(0)
            //          a*1 + b*1 + c*1 + d = f(1)
            //          a*2^3 + b*2^2 + c*2 + d = f(2)
            //          a*3^3 + b*3^2 + c*3 + d = f(3)
            // These equations can be rewritten as a matrix equation M * [a, b, c, d] = [f(0), f(1), f(2),
            // f(3)], where M is:
            //          0,  0,  0,  1
            //          1,  1,  1,  1
            //          2^3, 2^2, 2,  1
            //          3^3, 3^2, 3,  1
            // We can invert this matrix in order to compute a, b, c, d:
            //      -1/6,	1/2,	-1/2,	1/6
            //      1,	    -5/2,	2,	    -1/2
            //      -11/6,	3,	    -3/2,	1/3
            //      1,	    0,	    0,	    0
            // To compute these values, we can multiply everything by 6 and multiply by inverse_six at the
            // end for each coefficient The resulting computation here does 18 field adds, 6 subtracts, 3
            // muls to compute a, b, c, and d.
            let inverse_six = P::ScalarField::from(6u64).inverse().unwrap();

            let zero_times_2 = driver.add(poly[0], poly[0]);
            let zero_times_3 = driver.add(zero_times_2, poly[0]);
            let zero_times_6 = driver.add(zero_times_3, zero_times_3);
            let zero_times_12 = driver.add(zero_times_6, zero_times_6);

            let one_times_2 = driver.add(poly[1], poly[1]);
            let one_times_3 = driver.add(one_times_2, poly[1]);
            let one_times_6 = driver.add(one_times_3, one_times_3);

            let two_times_2 = driver.add(poly[2], poly[2]);
            let two_times_3 = driver.add(two_times_2, poly[2]);

            let three_times_2 = driver.add(poly[3], poly[3]);
            let three_times_3 = driver.add(three_times_2, poly[3]);

            let one_minus_two_times_3 = driver.sub(one_times_3, two_times_3);
            let one_minus_two_times_6 = driver.add(one_minus_two_times_3, one_minus_two_times_3);
            let one_minus_two_times_12 = driver.add(one_minus_two_times_6, one_minus_two_times_6);

            let tmp = driver.add(one_minus_two_times_3, poly[3]);
            let tmp = driver.sub(tmp, poly[0]);
            let a = driver.mul_with_public(inverse_six, tmp); // compute a in 1 muls and 4 adds

            let tmp = driver.sub(zero_times_6, one_minus_two_times_12);
            let tmp = driver.sub(tmp, one_times_3);
            let tmp = driver.sub(tmp, three_times_3);
            let b = driver.mul_with_public(inverse_six, tmp);

            let tmp = driver.sub(poly[0], zero_times_12);
            let tmp = driver.add(tmp, one_minus_two_times_12);
            let tmp = driver.add(tmp, one_times_6);
            let tmp = driver.add(tmp, two_times_3);
            let tmp = driver.add(tmp, three_times_2);
            let c = driver.mul_with_public(inverse_six, tmp);

            // Then, outside of the a, b, c, d computation, we need to do some extra precomputation
            // This work is 3 field muls, 8 adds
            let a_plus_b = driver.add(a, b);
            let a_plus_b_times_2 = driver.add(a_plus_b, a_plus_b);
            let start_idx_sqr = (length - 1) * (length - 1);
            let idx_sqr_three = start_idx_sqr + start_idx_sqr + start_idx_sqr;
            let mut idx_sqr_three_times_a =
                driver.mul_with_public(P::ScalarField::from(idx_sqr_three as u64), a);
            let mut x_a_term =
                driver.mul_with_public(P::ScalarField::from(6 * (length - 1) as u64), a);
            let two_a = driver.add(a, a);
            let three_a = driver.add(two_a, a);
            let six_a = driver.add(three_a, three_a);

            let three_a_plus_two_b = driver.add(a_plus_b_times_2, a);

            let tmp =
                driver.mul_with_public(P::ScalarField::from(length as u64 - 1), three_a_plus_two_b);
            let tmp = driver.add(tmp, a_plus_b);
            let mut linear_term = driver.add(tmp, c);

            // For each new evaluation, we do only 6 field additions and 0 muls.
            for i in length..extended_length {
                let tmp = driver.add(idx_sqr_three_times_a, linear_term);
                self.evaluations[i] = driver.add(self.evaluations[i - 1], tmp);

                idx_sqr_three_times_a = driver.add(idx_sqr_three_times_a, x_a_term);
                idx_sqr_three_times_a = driver.add(idx_sqr_three_times_a, three_a);

                x_a_term = driver.add(x_a_term, six_a);

                linear_term = driver.add(linear_term, three_a_plus_two_b);
            }
        } else {
            let big_domain = Barycentric::construct_big_domain(length, extended_length);
            let lagrange_denominators =
                Barycentric::construct_lagrange_denominators(length, &big_domain);
            let dominator_inverses = Barycentric::construct_denominator_inverses(
                extended_length,
                &big_domain,
                &lagrange_denominators,
            );
            let full_numerator_values =
                Barycentric::construct_full_numerator_values(length, extended_length, &big_domain);

            for k in length..extended_length {
                self.evaluations[k] = Default::default();

                // compute each term v_j / (d_j*(x-x_j)) of the sum
                for (j, term) in poly.iter().cloned().enumerate() {
                    let term = driver.mul_with_public(dominator_inverses[length * k + j], term);
                    self.evaluations[k] = driver.add(self.evaluations[k], term);
                }
                // scale the sum by the value of of B(x)
                self.evaluations[k] =
                    driver.mul_with_public(full_numerator_values[k], self.evaluations[k]);
            }
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing, const SIZE: usize> Default
    for SharedUnivariate<T, P, SIZE>
{
    fn default() -> Self {
        Self {
            evaluations: array::from_fn(|_| T::ArithmeticShare::default()),
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing, const SIZE: usize> Clone
    for SharedUnivariate<T, P, SIZE>
{
    fn clone(&self) -> Self {
        Self {
            evaluations: self.evaluations,
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing, const SIZE: usize> std::fmt::Debug
    for SharedUnivariate<T, P, SIZE>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.evaluations.iter()).finish()
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing, const SIZE: usize> AsRef<[T::ArithmeticShare]>
    for SharedUnivariate<T, P, SIZE>
{
    fn as_ref(&self) -> &[T::ArithmeticShare] {
        self.evaluations.as_ref()
    }
}
