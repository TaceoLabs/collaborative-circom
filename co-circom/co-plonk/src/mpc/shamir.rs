use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly::univariate::DensePolynomial;
use ark_poly::EvaluationDomain;
use ark_poly::Polynomial;

use mpc_core::protocols::shamir::{
    arithmetic, network::ShamirNetwork, pointshare, ShamirPointShare, ShamirPrimeFieldShare,
    ShamirProtocol,
};

use super::{CircomPlonkProver, IoResult};

/// A Plonk driver unsing shamir secret sharing
///
/// Contains two [`ShamirProtocol`]s, `protocol0` for the main execution and `protocol0` for parts that can run concurrently.
pub struct ShamirPlonkDriver<F: PrimeField, N: ShamirNetwork> {
    protocol0: ShamirProtocol<F, N>,
    protocol1: ShamirProtocol<F, N>,
}

impl<F: PrimeField, N: ShamirNetwork> ShamirPlonkDriver<F, N> {
    /// Create a new [`ShamirPlonkDriver`] with two [`ShamirProtocol`]s
    pub fn new(protocol0: ShamirProtocol<F, N>, protocol1: ShamirProtocol<F, N>) -> Self {
        Self {
            protocol0,
            protocol1,
        }
    }
}

impl<P: Pairing, N: ShamirNetwork> CircomPlonkProver<P> for ShamirPlonkDriver<P::ScalarField, N> {
    type ArithmeticShare = ShamirPrimeFieldShare<P::ScalarField>;
    type PointShareG1 = ShamirPointShare<P::G1>;
    type PointShareG2 = ShamirPointShare<P::G2>;

    type PartyID = usize;

    type IoContext = ShamirProtocol<P::ScalarField, N>;

    async fn close_network(self) -> IoResult<()> {
        self.protocol0.network.shutdown().await?;
        self.protocol1.network.shutdown().await?;
        Ok(())
    }

    fn rand(&mut self) -> IoResult<Self::ArithmeticShare> {
        self.protocol0.rand()
    }

    fn get_party_id(&self) -> Self::PartyID {
        self.protocol0.network.get_id()
    }

    fn add(a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare {
        arithmetic::add(a, b)
    }

    fn add_with_public(
        _party_id: Self::PartyID,
        shared: Self::ArithmeticShare,
        public: <P as Pairing>::ScalarField,
    ) -> Self::ArithmeticShare {
        arithmetic::add_public(shared, public)
    }

    fn sub(a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare {
        arithmetic::sub(a, b)
    }

    fn neg_vec_in_place(&mut self, a: &mut [Self::ArithmeticShare]) {
        for a in a.iter_mut() {
            *a = arithmetic::neg(*a);
        }
    }

    fn mul_with_public(
        shared: Self::ArithmeticShare,
        public: <P as Pairing>::ScalarField,
    ) -> Self::ArithmeticShare {
        arithmetic::mul_public(shared, public)
    }

    fn local_mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> Vec<P::ScalarField> {
        arithmetic::local_mul_vec(a, b)
    }

    async fn io_round_mul_vec(
        &mut self,
        a: Vec<P::ScalarField>,
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        self.protocol0.degree_reduce_vec(a).await
    }

    async fn mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        arithmetic::mul_vec(a, b, &mut self.protocol0).await
    }

    async fn mul_vecs(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
        c: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        let tmp = arithmetic::mul_vec(a, b, &mut self.protocol0).await?;
        arithmetic::mul_vec(&tmp, c, &mut self.protocol0).await
    }

    async fn add_mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
        c: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        let mut result = arithmetic::mul_vec(b, c, &mut self.protocol0).await?;
        arithmetic::add_vec_assign(&mut result, a);
        Ok(result)
    }

    async fn mul_open_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<<P as Pairing>::ScalarField>> {
        arithmetic::mul_open_vec(a, b, &mut self.protocol0).await
    }

    async fn open_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<<P as Pairing>::ScalarField>> {
        arithmetic::open_vec(a, &mut self.protocol0).await
    }

    async fn inv_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        arithmetic::inv_vec(a, &mut self.protocol0).await
    }

    fn promote_to_trivial_share(
        _party_id: Self::PartyID,
        public_value: <P as Pairing>::ScalarField,
    ) -> Self::ArithmeticShare {
        arithmetic::promote_to_trivial_share(public_value)
    }

    fn fft<D: EvaluationDomain<<P as Pairing>::ScalarField>>(
        data: &[Self::ArithmeticShare],
        domain: &D,
    ) -> Vec<Self::ArithmeticShare> {
        domain.fft(data)
    }

    fn ifft<D: EvaluationDomain<<P as Pairing>::ScalarField>>(
        data: &[Self::ArithmeticShare],
        domain: &D,
    ) -> Vec<Self::ArithmeticShare> {
        domain.ifft(data)
    }

    async fn open_point_g1(&mut self, a: Self::PointShareG1) -> IoResult<<P as Pairing>::G1> {
        pointshare::open_point(&a, &mut self.protocol0).await
    }

    async fn open_point_vec_g1(
        &mut self,
        a: &[Self::PointShareG1],
    ) -> IoResult<Vec<<P as Pairing>::G1>> {
        pointshare::open_point_many(a, &mut self.protocol0).await
    }

    fn msm_public_points_g1(
        points: &[<P as Pairing>::G1Affine],
        scalars: &[Self::ArithmeticShare],
    ) -> Self::PointShareG1 {
        pointshare::msm_public_points(points, scalars)
    }

    fn evaluate_poly_public(
        coeffs: Vec<Self::ArithmeticShare>,
        point: P::ScalarField,
    ) -> (Self::ArithmeticShare, Vec<Self::ArithmeticShare>) {
        let poly = DensePolynomial {
            coeffs: Self::ArithmeticShare::convert_vec(coeffs),
        };
        let result = Self::ArithmeticShare::new(poly.evaluate(&point));
        let coeffs = Self::ArithmeticShare::convert_vec_rev(poly.coeffs);
        (result, coeffs)
    }

    // To reduce the number of communication rounds, we implement the array_prod_mul macro according to https://www.usenix.org/system/files/sec22-ozdemir.pdf, p11 first paragraph.
    // TODO parallelize these? With a different network structure this might not be needed though
    async fn array_prod_mul(
        io_context: &mut Self::IoContext,
        inv: bool,
        arr1: &[Self::ArithmeticShare],
        arr2: &[Self::ArithmeticShare],
        arr3: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        let arr = arithmetic::mul_vec(arr1, arr2, io_context).await?;
        let arr = arithmetic::mul_vec(&arr, arr3, io_context).await?;
        // Do the multiplications of inp[i] * inp[i-1] in constant rounds
        let len = arr.len();

        let mut r = Vec::with_capacity(len + 1);
        for _ in 0..=len {
            r.push(io_context.rand()?);
        }
        let r_inv = arithmetic::inv_vec(&r, io_context).await?;
        let r_inv0 = vec![r_inv[0]; len];
        let mut unblind = arithmetic::mul_vec(&r_inv0, &r[1..], io_context).await?;

        let mul = arithmetic::mul_vec(&r[..len], &arr, io_context).await?;
        let mut open = arithmetic::mul_open_vec(&mul, &r_inv[1..], io_context).await?;

        for i in 1..open.len() {
            open[i] = open[i] * open[i - 1];
        }

        for (unblind, open) in unblind.iter_mut().zip(open.into_iter()) {
            *unblind = arithmetic::mul_public(*unblind, open);
        }
        if inv {
            Ok(arithmetic::inv_vec(&unblind, io_context).await?)
        } else {
            Ok(unblind)
        }
    }

    async fn array_prod_mul2(
        &mut self,
        n1: &[Self::ArithmeticShare],
        n2: &[Self::ArithmeticShare],
        n3: &[Self::ArithmeticShare],
        d1: &[Self::ArithmeticShare],
        d2: &[Self::ArithmeticShare],
        d3: &[Self::ArithmeticShare],
    ) -> IoResult<(Vec<Self::ArithmeticShare>, Vec<Self::ArithmeticShare>)> {
        let (num, den) = tokio::join!(
            <Self as CircomPlonkProver<P>>::array_prod_mul(&mut self.protocol0, false, n1, n2, n3),
            <Self as CircomPlonkProver<P>>::array_prod_mul(&mut self.protocol1, true, d1, d2, d3),
        );
        Ok((num?, den?))
    }
}
