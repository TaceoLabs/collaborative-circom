use ark_ec::pairing::Pairing;
use ark_poly::EvaluationDomain;
use mpc_core::protocols::rep3::{
    arithmetic,
    id::PartyID,
    network::{IoContext, Rep3Network},
    pointshare, poly, Rep3PointShare, Rep3PrimeFieldShare,
};

use super::{CircomPlonkProver, IoResult};

/// A Plonk driver for REP3 secret sharing
///
/// Contains two [`IoContext`]s, `io_context0` for the main execution and `io_context1` for parts that can run concurrently.
pub struct Rep3PlonkDriver<N: Rep3Network> {
    io_context0: IoContext<N>,
    io_context1: IoContext<N>,
}

impl<N: Rep3Network> Rep3PlonkDriver<N> {
    /// Create a new [`Rep3PlonkDriver`] with two [`IoContext`]s
    pub fn new(io_context0: IoContext<N>, io_context1: IoContext<N>) -> Self {
        Self {
            io_context0,
            io_context1,
        }
    }
}

impl<P: Pairing, N: Rep3Network> CircomPlonkProver<P> for Rep3PlonkDriver<N> {
    type ArithmeticShare = Rep3PrimeFieldShare<P::ScalarField>;
    type PointShareG1 = Rep3PointShare<P::G1>;
    type PointShareG2 = Rep3PointShare<P::G2>;

    type PartyID = PartyID;

    type IoContext = IoContext<N>;

    async fn close_network(self) -> IoResult<()> {
        self.io_context0.network.shutdown().await?;
        self.io_context1.network.shutdown().await?;
        Ok(())
    }

    fn rand(&mut self) -> IoResult<Self::ArithmeticShare> {
        Ok(Self::ArithmeticShare::rand(&mut self.io_context0))
    }

    fn get_party_id(&self) -> Self::PartyID {
        self.io_context0.id
    }

    fn add(a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare {
        arithmetic::add(a, b)
    }

    fn add_with_public(
        party_id: Self::PartyID,
        shared: Self::ArithmeticShare,
        public: P::ScalarField,
    ) -> Self::ArithmeticShare {
        arithmetic::add_public(shared, public, party_id)
    }

    fn sub(a: Self::ArithmeticShare, b: Self::ArithmeticShare) -> Self::ArithmeticShare {
        arithmetic::sub(a, b)
    }

    fn neg_vec_in_place(&mut self, vec: &mut [Self::ArithmeticShare]) {
        #[allow(unused_mut)]
        for mut a in vec.iter_mut() {
            *a = arithmetic::neg(*a);
        }
    }

    fn mul_with_public(
        shared: Self::ArithmeticShare,
        public: P::ScalarField,
    ) -> Self::ArithmeticShare {
        arithmetic::mul_public(shared, public)
    }

    fn local_mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> Vec<P::ScalarField> {
        arithmetic::local_mul_vec::<P::ScalarField>(a, b, &mut self.io_context0.rngs)
    }

    async fn io_round_mul_vec(
        &mut self,
        a: Vec<P::ScalarField>,
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        arithmetic::io_mul_vec(a, &mut self.io_context0).await
    }

    async fn mul_vec(
        &mut self,
        lhs: &[Self::ArithmeticShare],
        rhs: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        arithmetic::mul_vec(lhs, rhs, &mut self.io_context0).await
    }

    async fn mul_vecs(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
        c: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        let tmp = arithmetic::mul_vec(a, b, &mut self.io_context0).await?;
        arithmetic::mul_vec(&tmp, c, &mut self.io_context1).await
    }

    async fn add_mul_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
        c: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        let mut result = arithmetic::mul_vec(b, c, &mut self.io_context0).await?;
        arithmetic::add_vec_assign(&mut result, a);
        Ok(result)
    }

    async fn mul_open_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
        b: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<P::ScalarField>> {
        arithmetic::mul_open_vec(a, b, &mut self.io_context0).await
    }

    async fn open_vec(&mut self, a: &[Self::ArithmeticShare]) -> IoResult<Vec<P::ScalarField>> {
        arithmetic::open_vec(a, &mut self.io_context0).await
    }

    async fn inv_vec(
        &mut self,
        a: &[Self::ArithmeticShare],
    ) -> IoResult<Vec<Self::ArithmeticShare>> {
        arithmetic::inv_vec(a, &mut self.io_context0).await
    }

    fn promote_to_trivial_share(
        party_id: Self::PartyID,
        public_value: P::ScalarField,
    ) -> Self::ArithmeticShare {
        Self::ArithmeticShare::promote_from_trivial(&public_value, party_id)
    }

    fn fft<D: EvaluationDomain<P::ScalarField>>(
        data: &[Self::ArithmeticShare],
        domain: &D,
    ) -> Vec<Self::ArithmeticShare> {
        domain.fft(data)
    }

    fn ifft<D: ark_poly::EvaluationDomain<P::ScalarField>>(
        data: &[Self::ArithmeticShare],
        domain: &D,
    ) -> Vec<Self::ArithmeticShare> {
        domain.ifft(data)
    }

    async fn open_point_g1(&mut self, a: Self::PointShareG1) -> IoResult<P::G1> {
        pointshare::open_point(&a, &mut self.io_context0).await
    }

    async fn open_point_vec_g1(&mut self, a: &[Self::PointShareG1]) -> IoResult<Vec<P::G1>> {
        pointshare::open_point_many(a, &mut self.io_context0).await
    }

    fn msm_public_points_g1(
        points: &[P::G1Affine],
        scalars: &[Self::ArithmeticShare],
    ) -> Self::PointShareG1 {
        pointshare::msm_public_points(points, scalars)
    }

    fn evaluate_poly_public(
        coeffs: Vec<Self::ArithmeticShare>,
        point: P::ScalarField,
    ) -> (Self::ArithmeticShare, Vec<Self::ArithmeticShare>) {
        let result = poly::eval_poly(&coeffs, point);
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
            r.push(arithmetic::rand(io_context));
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
            <Self as CircomPlonkProver<P>>::array_prod_mul(
                &mut self.io_context0,
                false,
                n1,
                n2,
                n3
            ),
            <Self as CircomPlonkProver<P>>::array_prod_mul(&mut self.io_context1, true, d1, d2, d3),
        );
        Ok((num?, den?))
    }
}
