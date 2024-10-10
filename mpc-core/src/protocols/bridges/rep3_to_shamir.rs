use crate::protocols::{
    rep3::{Rep3PointShare, Rep3PrimeFieldShare},
    shamir::{network::ShamirNetwork, ShamirPointShare, ShamirPrimeFieldShare, ShamirProtocol},
};
use ark_ec::CurveGroup;
use ark_ff::PrimeField;

impl<F: PrimeField, N: ShamirNetwork> ShamirProtocol<F, N> {
    /// Translate a Rep3 prime field share into a 3-party Shamir prime field share, where the underlying sharing polynomial is of degree 1 (i.e., the threshold t = 1).
    pub async fn translate_primefield_repshare(
        &mut self,
        input: Rep3PrimeFieldShare<F>,
    ) -> std::io::Result<ShamirPrimeFieldShare<F>> {
        // Essentially, a mul function
        let my_lagrange_coeff = self.open_lagrange_2t[0]
            .inverse()
            .expect("lagrange coeff must be invertible");
        let mul = input.a * my_lagrange_coeff;
        self.degree_reduce(mul).await
    }

    /// Translate a Rep3 prime field share vector into a 3-party Shamir prime field share vector, where the underlying sharing polynomial is of degree 1 (i.e., the threshold t = 1).
    pub async fn translate_primefield_repshare_vec(
        &mut self,
        input: Vec<Rep3PrimeFieldShare<F>>,
    ) -> std::io::Result<Vec<ShamirPrimeFieldShare<F>>> {
        // Essentially, a mul_vec function
        let my_lagrange_coeff = self.open_lagrange_2t[0]
            .inverse()
            .expect("lagrange coeff must be invertible");
        // TODO maybe we do not collect here? we can just provide the iter
        // to the next function?
        let muls = input
            .into_iter()
            .map(|rep_share| rep_share.a * my_lagrange_coeff)
            .collect::<Vec<_>>();
        self.degree_reduce_vec(muls).await
    }

    /// Translate a Rep3 point share into a 3-party Shamir point share, where the underlying sharing polynomial is of degree 1 (i.e., the threshold t = 1).
    pub async fn translate_point_repshare<C>(
        &mut self,
        input: Rep3PointShare<C>,
    ) -> std::io::Result<ShamirPointShare<C>>
    where
        C: CurveGroup + std::ops::Mul<F, Output = C> + for<'a> std::ops::Mul<&'a F, Output = C>,
    {
        // Essentially, a scalar_mul function
        let my_lagrange_coeff = self.open_lagrange_2t[0]
            .inverse()
            .expect("lagrange coeff must be invertible");
        let mul = input.a * my_lagrange_coeff;
        self.degree_reduce_point(mul).await
    }
}
