use ark_ec::CurveGroup;

use crate::protocols::rep3::Rep3PrimeFieldShare;

use super::Rep3PointShare;

impl<C: CurveGroup> std::ops::Add for Rep3PointShare<C> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl<C: CurveGroup> std::ops::Add<&Rep3PointShare<C>> for Rep3PointShare<C> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl<C: CurveGroup> std::ops::Add<&Rep3PointShare<C>> for &'_ Rep3PointShare<C> {
    type Output = Rep3PointShare<C>;

    fn add(self, rhs: &Rep3PointShare<C>) -> Self::Output {
        Rep3PointShare::<C> {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl<C: CurveGroup> std::ops::AddAssign<&Rep3PointShare<C>> for Rep3PointShare<C> {
    fn add_assign(&mut self, rhs: &Self) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl<C: CurveGroup> std::ops::Sub for Rep3PointShare<C> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<C: CurveGroup> std::ops::Sub<&Rep3PointShare<C>> for Rep3PointShare<C> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        Self {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}
impl<C: CurveGroup> std::ops::Sub<&Rep3PointShare<C>> for &'_ Rep3PointShare<C> {
    type Output = Rep3PointShare<C>;

    fn sub(self, rhs: &Rep3PointShare<C>) -> Self::Output {
        Rep3PointShare::<C> {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<C: CurveGroup> std::ops::SubAssign<&Rep3PointShare<C>> for Rep3PointShare<C> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.a -= rhs.a;
        self.b -= rhs.b;
    }
}

impl<C: CurveGroup> std::ops::Mul<C::ScalarField> for &'_ Rep3PointShare<C> {
    type Output = Rep3PointShare<C>;

    fn mul(self, scalar: C::ScalarField) -> Self::Output {
        Self::Output {
            a: self.a * scalar,
            b: self.b * scalar,
        }
    }
}

impl<C: CurveGroup> std::ops::Mul<&Rep3PointShare<C>> for Rep3PrimeFieldShare<C::ScalarField> {
    type Output = C;

    // Local part of mul only
    fn mul(self, rhs: &Rep3PointShare<C>) -> Self::Output {
        rhs.a * self.a + rhs.b * self.a + rhs.a * self.b
    }
}
