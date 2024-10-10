use ark_ff::PrimeField;

use super::types::Rep3PrimeFieldShare;

impl<F: PrimeField> std::ops::Add for Rep3PrimeFieldShare<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Rep3PrimeFieldShare::<F> {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}
impl<F: PrimeField> std::ops::Add<&Rep3PrimeFieldShare<F>> for &'_ Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn add(self, rhs: &Rep3PrimeFieldShare<F>) -> Self::Output {
        Rep3PrimeFieldShare::<F> {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl<F: PrimeField> std::ops::Add<F> for &Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn add(self, rhs: F) -> Self::Output {
        Self::Output {
            a: self.a + rhs,
            b: self.b + rhs,
        }
    }
}

impl<F: PrimeField> std::ops::AddAssign<Rep3PrimeFieldShare<F>> for Rep3PrimeFieldShare<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl<F: PrimeField> std::ops::AddAssign<&Rep3PrimeFieldShare<F>> for Rep3PrimeFieldShare<F> {
    fn add_assign(&mut self, rhs: &Rep3PrimeFieldShare<F>) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl<F: PrimeField> std::ops::Sub for Rep3PrimeFieldShare<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Rep3PrimeFieldShare::<F> {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<F: PrimeField> std::ops::Sub<&Rep3PrimeFieldShare<F>> for &'_ Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn sub(self, rhs: &Rep3PrimeFieldShare<F>) -> Self::Output {
        Rep3PrimeFieldShare::<F> {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<F: PrimeField> std::ops::Sub<F> for &Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn sub(self, rhs: F) -> Self::Output {
        Self::Output {
            a: self.a - rhs,
            b: self.b - rhs,
        }
    }
}

impl<F: PrimeField> std::ops::SubAssign<Rep3PrimeFieldShare<F>> for Rep3PrimeFieldShare<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.a -= rhs.a;
        self.b -= rhs.b;
    }
}

impl<F: PrimeField> std::ops::Mul for Rep3PrimeFieldShare<F> {
    type Output = F;

    // Local part of mul only
    fn mul(self, rhs: Rep3PrimeFieldShare<F>) -> Self::Output {
        self.a * rhs.a + self.a * rhs.b + self.b * rhs.a
    }
}

impl<F: PrimeField> std::ops::Mul<F> for Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Self::Output {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}

impl<F: PrimeField> std::ops::Mul<F> for &Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Self::Output {
            a: self.a * rhs,
            b: self.b * rhs,
        }
    }
}

impl<F: PrimeField> std::ops::Mul<&Rep3PrimeFieldShare<F>> for &'_ Rep3PrimeFieldShare<F> {
    type Output = F;

    // Local part of mul only
    fn mul(self, rhs: &Rep3PrimeFieldShare<F>) -> Self::Output {
        self.a * rhs.a + self.a * rhs.b + self.b * rhs.a
    }
}

impl<F: PrimeField> std::ops::MulAssign<F> for Rep3PrimeFieldShare<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.a *= rhs;
        self.b *= rhs;
    }
}

impl<F: PrimeField> std::ops::Neg for Rep3PrimeFieldShare<F> {
    type Output = Rep3PrimeFieldShare<F>;

    fn neg(self) -> Self::Output {
        Rep3PrimeFieldShare::<F> {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl<F: PrimeField> ark_ff::Zero for Rep3PrimeFieldShare<F> {
    fn zero() -> Self {
        Self {
            a: F::zero(),
            b: F::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        panic!("is_zero is not a meaningful operation for Rep3PrimeFieldShare, use interative zero check instead");
    }
}
