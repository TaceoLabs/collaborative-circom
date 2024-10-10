use ark_ec::CurveGroup;

use crate::protocols::shamir::ShamirPrimeFieldShare;

use super::ShamirPointShare;

impl<C: CurveGroup> std::ops::Add for ShamirPointShare<C> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { a: self.a + rhs.a }
    }
}

impl<C: CurveGroup> std::ops::Add<&ShamirPointShare<C>> for ShamirPointShare<C> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        Self { a: self.a + rhs.a }
    }
}

impl<C: CurveGroup> std::ops::Add<&ShamirPointShare<C>> for &'_ ShamirPointShare<C> {
    type Output = ShamirPointShare<C>;

    fn add(self, rhs: &ShamirPointShare<C>) -> Self::Output {
        ShamirPointShare::<C> { a: self.a + rhs.a }
    }
}

impl<C: CurveGroup> std::ops::AddAssign<&ShamirPointShare<C>> for ShamirPointShare<C> {
    fn add_assign(&mut self, rhs: &Self) {
        self.a += rhs.a;
    }
}

impl<C: CurveGroup> std::ops::Sub for ShamirPointShare<C> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self { a: self.a - rhs.a }
    }
}

impl<C: CurveGroup> std::ops::Sub<&ShamirPointShare<C>> for ShamirPointShare<C> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        Self { a: self.a - rhs.a }
    }
}
impl<C: CurveGroup> std::ops::Sub<&ShamirPointShare<C>> for &'_ ShamirPointShare<C> {
    type Output = ShamirPointShare<C>;

    fn sub(self, rhs: &ShamirPointShare<C>) -> Self::Output {
        ShamirPointShare::<C> { a: self.a - rhs.a }
    }
}

impl<C: CurveGroup> std::ops::SubAssign<&ShamirPointShare<C>> for ShamirPointShare<C> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.a -= rhs.a;
    }
}

impl<C: CurveGroup> std::ops::Mul<&C::ScalarField> for &'_ ShamirPointShare<C> {
    type Output = ShamirPointShare<C>;

    fn mul(self, scalar: &C::ScalarField) -> Self::Output {
        Self::Output { a: self.a * scalar }
    }
}

impl<C: CurveGroup> std::ops::Mul<&ShamirPointShare<C>> for ShamirPrimeFieldShare<C::ScalarField> {
    type Output = ShamirPointShare<C>;

    // Result has higher degree than the inputs
    fn mul(self, rhs: &ShamirPointShare<C>) -> Self::Output {
        ShamirPointShare::<C> { a: rhs.a * self.a }
    }
}
