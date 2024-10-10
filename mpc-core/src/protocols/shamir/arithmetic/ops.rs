use ark_ff::PrimeField;

use super::types::ShamirPrimeFieldShare;

impl<F: PrimeField> std::ops::Add for ShamirPrimeFieldShare<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { a: self.a + rhs.a }
    }
}

impl<F: PrimeField> std::ops::AddAssign for ShamirPrimeFieldShare<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.a += rhs.a;
    }
}

impl<F: PrimeField> std::ops::AddAssign<&ShamirPrimeFieldShare<F>> for ShamirPrimeFieldShare<F> {
    fn add_assign(&mut self, rhs: &Self) {
        self.a += rhs.a;
    }
}

impl<F: PrimeField> std::ops::Add<&ShamirPrimeFieldShare<F>> for ShamirPrimeFieldShare<F> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        Self { a: self.a + rhs.a }
    }
}

impl<F: PrimeField> std::ops::Add<&ShamirPrimeFieldShare<F>> for &'_ ShamirPrimeFieldShare<F> {
    type Output = ShamirPrimeFieldShare<F>;

    fn add(self, rhs: &ShamirPrimeFieldShare<F>) -> Self::Output {
        ShamirPrimeFieldShare::<F> { a: self.a + rhs.a }
    }
}

impl<F: PrimeField> std::ops::Add<&F> for &'_ ShamirPrimeFieldShare<F> {
    type Output = ShamirPrimeFieldShare<F>;

    fn add(self, rhs: &F) -> Self::Output {
        Self::Output { a: self.a + rhs }
    }
}

impl<F: PrimeField> std::ops::Add<F> for ShamirPrimeFieldShare<F> {
    type Output = ShamirPrimeFieldShare<F>;

    fn add(self, rhs: F) -> Self::Output {
        Self::Output { a: self.a + rhs }
    }
}

impl<F: PrimeField> std::ops::AddAssign<F> for ShamirPrimeFieldShare<F> {
    fn add_assign(&mut self, rhs: F) {
        self.a += rhs;
    }
}

impl<F: PrimeField> std::ops::Sub for ShamirPrimeFieldShare<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self { a: self.a - rhs.a }
    }
}

impl<F: PrimeField> std::ops::SubAssign for ShamirPrimeFieldShare<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.a -= rhs.a;
    }
}

impl<F: PrimeField> std::ops::Sub<&ShamirPrimeFieldShare<F>> for ShamirPrimeFieldShare<F> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        Self { a: self.a - rhs.a }
    }
}

impl<F: PrimeField> std::ops::Sub<&ShamirPrimeFieldShare<F>> for &'_ ShamirPrimeFieldShare<F> {
    type Output = ShamirPrimeFieldShare<F>;

    fn sub(self, rhs: &ShamirPrimeFieldShare<F>) -> Self::Output {
        ShamirPrimeFieldShare::<F> { a: self.a - rhs.a }
    }
}

impl<F: PrimeField> std::ops::Mul for ShamirPrimeFieldShare<F> {
    type Output = Self;

    // Result has higher degree than the inputs
    fn mul(self, rhs: Self) -> Self::Output {
        ShamirPrimeFieldShare::<F> { a: self.a * rhs.a }
    }
}

impl<F: PrimeField> std::ops::Mul<&ShamirPrimeFieldShare<F>> for ShamirPrimeFieldShare<F> {
    type Output = ShamirPrimeFieldShare<F>;

    // Result has higher degree than the inputs
    fn mul(self, rhs: &Self) -> Self::Output {
        ShamirPrimeFieldShare::<F> { a: self.a * rhs.a }
    }
}

impl<F: PrimeField> std::ops::Mul<&ShamirPrimeFieldShare<F>> for &'_ ShamirPrimeFieldShare<F> {
    type Output = ShamirPrimeFieldShare<F>;

    // Result has higher degree than the inputs
    fn mul(self, rhs: &ShamirPrimeFieldShare<F>) -> Self::Output {
        ShamirPrimeFieldShare::<F> { a: self.a * rhs.a }
    }
}

impl<F: PrimeField> std::ops::Mul<&F> for &'_ ShamirPrimeFieldShare<F> {
    type Output = ShamirPrimeFieldShare<F>;

    fn mul(self, rhs: &F) -> Self::Output {
        Self::Output { a: self.a * rhs }
    }
}

impl<F: PrimeField> std::ops::Mul<F> for ShamirPrimeFieldShare<F> {
    type Output = ShamirPrimeFieldShare<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Self::Output { a: self.a * rhs }
    }
}

impl<F: PrimeField> std::ops::MulAssign<F> for ShamirPrimeFieldShare<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.a *= rhs;
    }
}

impl<F: PrimeField> std::ops::Neg for ShamirPrimeFieldShare<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { a: -self.a }
    }
}
impl<F: PrimeField> std::ops::Neg for &ShamirPrimeFieldShare<F> {
    type Output = ShamirPrimeFieldShare<F>;

    fn neg(self) -> Self::Output {
        ShamirPrimeFieldShare::<F> { a: -self.a }
    }
}

impl<F: PrimeField> ark_ff::Zero for ShamirPrimeFieldShare<F> {
    fn zero() -> Self {
        Self { a: F::zero() }
    }

    fn is_zero(&self) -> bool {
        panic!("is_zero is not a meaningful operation for Rep3PrimeFieldShare, use interative zero check instead");
    }
}
