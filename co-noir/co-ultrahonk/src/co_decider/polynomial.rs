use ark_ec::pairing::Pairing;
use ark_ff::{Field, Zero};
use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};
use ultrahonk::prelude::Polynomial;

use crate::mpc::NoirUltraHonkProver;

pub(crate) struct SharedPolynomial<T: NoirUltraHonkProver<P>, P: Pairing> {
    pub(crate) coefficients: Vec<T::ArithmeticShare>,
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> SharedPolynomial<T, P> {
    pub fn new(coefficients: Vec<T::ArithmeticShare>) -> Self {
        Self { coefficients }
    }

    pub(crate) fn promote_poly(driver: &T, poly: Polynomial<P::ScalarField>) -> Self {
        let coefficients = T::promote_to_trivial_shares(driver.get_party_id(), poly.as_ref());
        Self { coefficients }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &T::ArithmeticShare> {
        self.coefficients.iter()
    }

    pub(crate) fn add_assign_slice(&mut self, driver: &mut T, other: &[T::ArithmeticShare]) {
        // Barrettenberg uses multithreading here
        for (des, src) in self.coefficients.iter_mut().zip(other.iter()) {
            *des = driver.add(*des, *src);
        }
    }

    pub(crate) fn add_scaled_slice(
        &mut self,
        driver: &mut T,
        src: &[T::ArithmeticShare],
        scalar: &P::ScalarField,
    ) {
        // Barrettenberg uses multithreading here
        for (des, src) in self.coefficients.iter_mut().zip(src.iter()) {
            let tmp = driver.mul_with_public(*scalar, *src);
            *des = driver.add(*des, tmp);
        }
    }

    pub(crate) fn add_scaled(
        &mut self,
        driver: &mut T,
        src: &SharedPolynomial<T, P>,
        scalar: &P::ScalarField,
    ) {
        self.add_scaled_slice(driver, &src.coefficients, scalar);
    }

    pub(crate) fn add_scaled_slice_public(
        &mut self,
        driver: &mut T,
        src: &[P::ScalarField],
        scalar: &P::ScalarField,
    ) {
        // Barrettenberg uses multithreading here
        for (des, src) in self.coefficients.iter_mut().zip(src.iter()) {
            let tmp = *scalar * src;
            *des = driver.add_with_public(tmp, *des);
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.coefficients.len()
    }

    // Can only shift by 1
    pub(crate) fn shifted(&self) -> &[T::ArithmeticShare] {
        assert!(!self.coefficients.is_empty());
        &self.coefficients[1..]
    }

    /**
     * @brief Divides p(X) by (X-r) in-place.
     */
    pub(crate) fn factor_roots(&mut self, driver: &mut T, root: &P::ScalarField) {
        if root.is_zero() {
            // if one of the roots is 0 after having divided by all other roots,
            // then p(X) = a₁⋅X + ⋯ + aₙ₋₁⋅Xⁿ⁻¹
            // so we shift the array of coefficients to the left
            // and the result is p(X) = a₁ + ⋯ + aₙ₋₁⋅Xⁿ⁻² and we subtract 1 from the size.
            self.coefficients.remove(0);
        } else {
            // assume
            //  • r != 0
            //  • (X−r) | p(X)
            //  • q(X) = ∑ᵢⁿ⁻² bᵢ⋅Xⁱ
            //  • p(X) = ∑ᵢⁿ⁻¹ aᵢ⋅Xⁱ = (X-r)⋅q(X)
            //
            // p(X)         0           1           2       ...     n-2             n-1
            //              a₀          a₁          a₂              aₙ₋₂            aₙ₋₁
            //
            // q(X)         0           1           2       ...     n-2             n-1
            //              b₀          b₁          b₂              bₙ₋₂            0
            //
            // (X-r)⋅q(X)   0           1           2       ...     n-2             n-1
            //              -r⋅b₀       b₀-r⋅b₁     b₁-r⋅b₂         bₙ₋₃−r⋅bₙ₋₂      bₙ₋₂
            //
            // b₀   = a₀⋅(−r)⁻¹
            // b₁   = (a₁ - b₀)⋅(−r)⁻¹
            // b₂   = (a₂ - b₁)⋅(−r)⁻¹
            //      ⋮
            // bᵢ   = (aᵢ − bᵢ₋₁)⋅(−r)⁻¹
            //      ⋮
            // bₙ₋₂ = (aₙ₋₂ − bₙ₋₃)⋅(−r)⁻¹
            // bₙ₋₁ = 0

            // For the simple case of one root we compute (−r)⁻¹ and
            let root_inverse = (-*root).inverse().expect("Root is not zero here");
            // set b₋₁ = 0
            let mut temp = Default::default();
            // We start multiplying lower coefficient by the inverse and subtracting those from highter coefficients
            // Since (x - r) should divide the polynomial cleanly, we can guide division with lower coefficients
            for coeff in self.coefficients.iter_mut() {
                // at the start of the loop, temp = bᵢ₋₁
                // and we can compute bᵢ   = (aᵢ − bᵢ₋₁)⋅(−r)⁻¹
                temp = driver.sub(*coeff, temp);
                temp = driver.mul_with_public(root_inverse, temp);
                *coeff = temp.to_owned();
            }
        }
        self.coefficients.pop();
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> Clone for SharedPolynomial<T, P> {
    fn clone(&self) -> Self {
        Self {
            coefficients: self.coefficients.clone(),
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> Default for SharedPolynomial<T, P> {
    fn default() -> Self {
        Self {
            coefficients: Default::default(),
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> Debug for SharedPolynomial<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedPolynomial")
            .field("coefficients", &self.coefficients)
            .finish()
    }
}
impl<T: NoirUltraHonkProver<P>, P: Pairing> AsRef<[T::ArithmeticShare]> for SharedPolynomial<T, P> {
    fn as_ref(&self) -> &[T::ArithmeticShare] {
        &self.coefficients
    }
}
impl<T: NoirUltraHonkProver<P>, P: Pairing> Index<usize> for SharedPolynomial<T, P> {
    type Output = T::ArithmeticShare;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coefficients[index]
    }
}
impl<T: NoirUltraHonkProver<P>, P: Pairing> IndexMut<usize> for SharedPolynomial<T, P> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coefficients[index]
    }
}
