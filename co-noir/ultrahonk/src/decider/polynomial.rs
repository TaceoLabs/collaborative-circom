use ark_ff::{PrimeField, Zero};
use std::ops::{AddAssign, Index, IndexMut};

#[derive(Clone, Debug, Default)]
pub struct Polynomial<F> {
    pub(crate) coefficients: Vec<F>,
}

pub struct ShiftedPoly<'a, F> {
    pub(crate) coefficients: &'a [F],
    zero: F, // TODO is there are better solution
}

impl<'a, F: PrimeField> ShiftedPoly<'a, F> {
    pub(crate) fn to_vec(&self) -> Vec<F> {
        let mut res = Vec::with_capacity(self.coefficients.len() + 1);
        for c in self.coefficients.iter() {
            res.push(*c);
        }
        res.push(self.zero);
        res
    }

    pub(crate) fn as_ref(&self) -> &[F] {
        self.coefficients
    }
}

impl<'a, F: PrimeField> Index<usize> for ShiftedPoly<'a, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        if index == self.coefficients.len() {
            &self.zero
        } else {
            &self.coefficients[index]
        }
    }
}

impl<F: Clone> Polynomial<F> {
    pub(crate) fn new(coefficients: Vec<F>) -> Self {
        Self { coefficients }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &F> {
        self.coefficients.iter()
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.coefficients.iter_mut()
    }

    pub(crate) fn len(&self) -> usize {
        self.coefficients.len()
    }

    pub(crate) fn as_ref(&self) -> &[F] {
        &self.coefficients
    }

    pub(crate) fn as_mut(&mut self) -> &mut [F] {
        &mut self.coefficients
    }

    pub(crate) fn resize(&mut self, size: usize, value: F) {
        self.coefficients.resize(size, value);
    }

    pub(crate) fn into_vec(self) -> Vec<F> {
        self.coefficients
    }
}

impl<F: Zero + Clone> Polynomial<F> {
    pub(crate) fn new_zero(size: usize) -> Self {
        Self {
            coefficients: vec![F::zero(); size],
        }
    }

    pub(crate) fn degree(&self) -> usize {
        let mut len = self.coefficients.len() - 1;
        for c in self.coefficients.iter().rev() {
            if c.is_zero() {
                len -= 1;
            } else {
                break;
            }
        }
        len
    }

    // Can only shift by 1
    pub(crate) fn shifted(&self) -> ShiftedPoly<F> {
        assert!(!self.coefficients.is_empty());
        assert!(self.coefficients[0].is_zero());
        ShiftedPoly {
            coefficients: &self.coefficients[1..],
            zero: F::zero(),
        }
    }
}

impl<F: PrimeField> Polynomial<F> {
    /**
     * @brief Divides p(X) by (X-r) in-place.
     */
    pub(crate) fn factor_roots(&mut self, root: &F) {
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
            let mut temp = F::zero();
            // We start multiplying lower coefficient by the inverse and subtracting those from highter coefficients
            // Since (x - r) should divide the polynomial cleanly, we can guide division with lower coefficients
            for coeff in self.coefficients.iter_mut() {
                // at the start of the loop, temp = bᵢ₋₁
                // and we can compute bᵢ   = (aᵢ − bᵢ₋₁)⋅(−r)⁻¹
                temp = *coeff - temp;
                temp *= root_inverse;
                *coeff = temp;
            }
        }
        self.coefficients.pop();
    }

    pub(crate) fn add_scaled_slice(&mut self, src: &[F], scalar: &F) {
        // Barrettenberg uses multithreading here
        for (des, &src) in self.coefficients.iter_mut().zip(src.iter()) {
            *des += *scalar * src;
        }
    }

    pub(crate) fn add_scaled(&mut self, src: &Polynomial<F>, scalar: &F) {
        self.add_scaled_slice(&src.coefficients, scalar);
    }
}

impl<F> Index<usize> for Polynomial<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coefficients[index]
    }
}

impl<F> IndexMut<usize> for Polynomial<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coefficients[index]
    }
}

impl<F: PrimeField> AddAssign<&[F]> for Polynomial<F> {
    fn add_assign(&mut self, rhs: &[F]) {
        if rhs.len() > self.coefficients.len() {
            panic!("Polynomial too large, this should not have happened");
            // self.coefficients.resize(rhs.len(), F::zero());
        }
        for (l, r) in self.coefficients.iter_mut().zip(rhs.iter()) {
            *l += *r;
        }
    }
}
