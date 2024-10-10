//! # Shamir
//!
//! This module implements the shamir share and combine opertions and shamir preprocessing

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use itertools::izip;
use network::ShamirNetwork;
use rngs::ShamirRng;
use std::time::Instant;

use rand::{CryptoRng, Rng, SeedableRng};

use crate::RngType;

pub mod arithmetic;
pub mod core;
pub mod network;
pub mod pointshare;
mod rngs;

pub use arithmetic::types::ShamirPrimeFieldShare;
pub use pointshare::types::ShamirPointShare;

type IoResult<T> = std::io::Result<T>;
type ShamirShare<F> = ShamirPrimeFieldShare<F>;

/// Share a field element into Shamir shares with given `degree` and `num_parties`
pub fn share_field_element<F: PrimeField, R: Rng + CryptoRng>(
    val: F,
    degree: usize,
    num_parties: usize,
    rng: &mut R,
) -> Vec<ShamirShare<F>> {
    let shares = core::share(val, num_parties, degree, rng);
    ShamirShare::convert_vec_rev(shares)
}

/// Reconstructs a field element from its Shamir shares and lagrange coefficients. Thereby at least `degree` + 1 shares need to be present.
pub fn combine_field_element<F: PrimeField>(
    shares: &[ShamirShare<F>],
    coeffs: &[usize],
    degree: usize,
) -> eyre::Result<F> {
    if shares.len() != coeffs.len() {
        eyre::bail!(
            "Number of shares ({}) does not match number of party indices ({})",
            shares.len(),
            coeffs.len()
        );
    }
    if shares.len() <= degree {
        eyre::bail!(
            "Not enough shares to reconstruct the secret. Expected {}, got {}",
            degree + 1,
            shares.len()
        );
    }

    let lagrange = core::lagrange_from_coeff(&coeffs[..=degree]);
    let shares = ShamirShare::convert_slice(shares);
    let rec = core::reconstruct(&shares[..=degree], &lagrange);
    Ok(rec)
}

/// Secret shares a vector of field element using Shamir secret sharing and the provided random number generator. The field elements are split into num_parties shares each, where each party holds just one. The outputs are of type [ShamirShareVec]. The degree of the sharing polynomial (i.e., the threshold of maximum number of tolerated colluding parties) is specified by the degree parameter.
pub fn share_field_elements<F: PrimeField, R: Rng + CryptoRng>(
    vals: &[F],
    degree: usize,
    num_parties: usize,
    rng: &mut R,
) -> Vec<Vec<ShamirShare<F>>> {
    let mut result = (0..num_parties)
        .map(|_| Vec::with_capacity(vals.len()))
        .collect::<Vec<_>>();

    for val in vals {
        let shares = core::share(*val, num_parties, degree, rng);
        let shares = ShamirShare::convert_vec_rev(shares);
        for (r, s) in izip!(&mut result, shares) {
            r.push(s);
        }
    }

    result
}

/// Reconstructs a vector of field elements from its Shamir shares and lagrange coefficients. The input is structured as one [ShamirShareVec] per party. Thus, shares\[i\]\[j\] represents the j-th share of party i. Thereby at least `degree` + 1 shares need to be present per field element (i.e., i > degree).
pub fn combine_field_elements<F: PrimeField>(
    shares: &[Vec<ShamirShare<F>>],
    coeffs: &[usize],
    degree: usize,
) -> eyre::Result<Vec<F>> {
    if shares.len() != coeffs.len() {
        eyre::bail!(
            "Number of shares ({}) does not match number of party indices ({})",
            shares.len(),
            coeffs.len()
        );
    }
    if shares.len() <= degree {
        eyre::bail!(
            "Not enough shares to reconstruct the secret. Expected {}, got {}",
            degree + 1,
            shares.len()
        );
    }

    let num_vals = shares[0].len();
    for share in shares.iter().skip(1) {
        if share.len() != num_vals {
            eyre::bail!(
                "Number of shares ({}) does not match number of shares in first party ({})",
                share.len(),
                num_vals
            );
        }
    }
    let mut result = Vec::with_capacity(num_vals);

    let lagrange = core::lagrange_from_coeff(&coeffs[..=degree]);

    for i in 0..num_vals {
        let s = shares
            .iter()
            .take(degree + 1)
            .map(|s| s[i].a)
            .collect::<Vec<_>>();
        let rec = core::reconstruct(&s, &lagrange);
        result.push(rec);
    }
    Ok(result)
}

/// Secret shares a curve point using Shamir secret sharing and the provided random number generator. The point is split into num_parties shares, where each party holds just one. The outputs are of type [ShamirPointShare]. The degree of the sharing polynomial (i.e., the threshold of maximum number of tolerated colluding parties) is specified by the degree parameter.
pub fn share_curve_point<C: CurveGroup, R: Rng + CryptoRng>(
    val: C,
    degree: usize,
    num_parties: usize,
    rng: &mut R,
) -> Vec<ShamirPointShare<C>> {
    let shares = core::share_point(val, num_parties, degree, rng);
    ShamirPointShare::convert_vec_rev(shares)
}

/// Reconstructs a curve point from its Shamir shares and lagrange coefficients. Thereby at least `degree` + 1 shares need to be present.
pub fn combine_curve_point<C: CurveGroup>(
    shares: &[ShamirPointShare<C>],
    coeffs: &[usize],
    degree: usize,
) -> eyre::Result<C> {
    if shares.len() != coeffs.len() {
        eyre::bail!(
            "Number of shares ({}) does not match number of party indices ({})",
            shares.len(),
            coeffs.len()
        );
    }
    if shares.len() <= degree {
        eyre::bail!(
            "Not enough shares to reconstruct the secret. Expected {}, got {}",
            degree + 1,
            shares.len()
        );
    }

    let lagrange = core::lagrange_from_coeff(&coeffs[..=degree]);
    let shares = ShamirPointShare::convert_slice(shares);
    let rec = core::reconstruct_point(&shares[..=degree], &lagrange);

    Ok(rec)
}

/// This type is used to construct a [`SahmirProtocol`].
/// Preprocess `amount` number of corre;ated randomness pairs that are consumed while using the protocol.
pub struct ShamirPreprocessing<F: PrimeField, N: ShamirNetwork> {
    threshold: usize,
    rng_buffer: ShamirRng<F>,
    network: N,
}

impl<F: PrimeField, N: ShamirNetwork> ShamirPreprocessing<F, N> {
    /// Construct a new [`ShamirPreprocessing`] type and generate `amount` number of corr rand pairs
    pub async fn new(threshold: usize, mut network: N, amount: usize) -> eyre::Result<Self> {
        let num_parties = network.get_num_parties();

        if 2 * threshold + 1 > num_parties {
            eyre::bail!("Threshold too large for number of parties")
        }

        let num_parties = network.get_num_parties();

        let seed: [u8; crate::SEED_SIZE] = RngType::from_entropy().gen();
        let mut rng_buffer = ShamirRng::new(seed, threshold, num_parties);

        tracing::info!(
            "Party {}: generating correlated randomness..",
            network.get_id()
        );
        let start = Instant::now();
        // buffer_triple generates amount * (t + 1), so we ceil dive the amount we want
        let amount = amount.div_ceil(threshold + 1);
        rng_buffer.buffer_triples(&mut network, amount).await?;
        tracing::info!(
            "Party {}: generating took {} ms",
            network.get_id(),
            start.elapsed().as_millis()
        );

        Ok(Self {
            threshold,
            rng_buffer,
            network,
        })
    }
}

impl<F: PrimeField, N: ShamirNetwork> From<ShamirPreprocessing<F, N>> for ShamirProtocol<F, N> {
    fn from(value: ShamirPreprocessing<F, N>) -> Self {
        let num_parties = value.network.get_num_parties();
        // We send in circles, so we need to receive from the last parties
        let id = value.network.get_id();
        let open_lagrange_t = core::lagrange_from_coeff(
            &(0..value.threshold + 1)
                .map(|i| (id + num_parties - i) % num_parties + 1)
                .collect::<Vec<_>>(),
        );
        let open_lagrange_2t = core::lagrange_from_coeff(
            &(0..2 * value.threshold + 1)
                .map(|i| (id + num_parties - i) % num_parties + 1)
                .collect::<Vec<_>>(),
        );

        let mul_lagrange_2t =
            core::lagrange_from_coeff(&(1..=2 * value.threshold + 1).collect::<Vec<_>>());

        ShamirProtocol {
            threshold: value.threshold,
            open_lagrange_t,
            open_lagrange_2t,
            mul_lagrange_2t,
            rng: value.rng_buffer.rng,
            r_t: value.rng_buffer.r_t,
            r_2t: value.rng_buffer.r_2t,
            network: value.network,
        }
    }
}

/// This struct handles the Shamir MPC protocol, including proof generation. Thus, it implements the [PrimeFieldMpcProtocol], [EcMpcProtocol], [PairingEcMpcProtocol], [FFTProvider], and [MSMProvider] traits.
pub struct ShamirProtocol<F: PrimeField, N: ShamirNetwork> {
    /// The threshold, degree of polynomial
    pub threshold: usize,
    /// The open lagrange coeffs
    pub open_lagrange_t: Vec<F>,
    pub(crate) open_lagrange_2t: Vec<F>,
    mul_lagrange_2t: Vec<F>,
    rng: RngType,
    pub(crate) r_t: Vec<F>,
    pub(crate) r_2t: Vec<F>,
    /// The underlying [`ShamirNetwork`]
    pub network: N,
}

impl<F: PrimeField, N: ShamirNetwork> ShamirProtocol<F, N> {
    const KING_ID: usize = 0;

    /// Gracefully shutdown the netowork. Waits until all data is sent and received
    pub async fn close_network(self) -> std::io::Result<()> {
        self.network.shutdown().await
    }

    /// Create a forked [`ShamirProtocol`] that consumes `amount` number of corr rand pairs from its parent
    pub async fn fork_with_pairs(&mut self, amount: usize) -> std::io::Result<Self> {
        Ok(Self {
            threshold: self.threshold,
            open_lagrange_t: self.open_lagrange_t.clone(),
            open_lagrange_2t: self.open_lagrange_2t.clone(),
            mul_lagrange_2t: self.mul_lagrange_2t.clone(),
            rng: RngType::from_seed(self.rng.gen()),
            r_t: self.r_t.drain(0..amount).collect(),
            r_2t: self.r_2t.drain(0..amount).collect(),
            network: self.network.fork().await?,
        })
    }

    /// Get a correlated randomness pair
    pub fn get_pair(&mut self) -> std::io::Result<(F, F)> {
        if let (Some(r_t), Some(r_2t)) = (self.r_t.pop(), self.r_2t.pop()) {
            Ok((r_t, r_2t))
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "not enough correlated random pairs",
            ))
        }
    }

    /// Generates a random field element and returns it as a share.
    pub fn rand(&mut self) -> std::io::Result<ShamirPrimeFieldShare<F>> {
        self.get_pair().map(|(r, _)| ShamirPrimeFieldShare::new(r))
    }

    pub(crate) async fn degree_reduce(&mut self, mut input: F) -> std::io::Result<ShamirShare<F>> {
        let (r_t, r_2t) = self.get_pair()?;
        input += r_2t;

        let my_id = self.network.get_id();
        let my_share = if my_id == Self::KING_ID {
            // Accumulate the result
            let mut acc = F::zero();
            for (other_id, lagrange) in self.mul_lagrange_2t.iter().enumerate() {
                if other_id == Self::KING_ID {
                    acc += input * lagrange;
                } else {
                    let r = self.network.recv::<F>(other_id).await?;
                    acc += r * lagrange;
                }
            }
            // So far parties who do not require sending, do not send, so no receive here

            // Send fresh shares
            let shares = core::share(
                acc,
                self.network.get_num_parties(),
                self.threshold,
                &mut self.rng,
            );
            let mut my_share = F::default();
            for (other_id, share) in shares.into_iter().enumerate() {
                if my_id == other_id {
                    my_share = share;
                } else {
                    self.network.send(other_id, share).await?;
                }
            }
            my_share
        } else {
            if my_id <= self.threshold * 2 {
                // Only send if my items are required
                self.network.send(Self::KING_ID, input).await?;
            }
            self.network.recv(Self::KING_ID).await?
        };

        Ok(ShamirShare::new(my_share - r_t))
    }

    /// Degree reduce all inputs
    pub async fn degree_reduce_vec(
        &mut self,
        mut inputs: Vec<F>,
    ) -> std::io::Result<Vec<ShamirShare<F>>> {
        let len = inputs.len();
        let mut r_ts = Vec::with_capacity(len);

        for inp in inputs.iter_mut() {
            let (r_t, r_2t) = self.get_pair()?;
            *inp += r_2t;
            r_ts.push(r_t);
        }

        let my_id = self.network.get_id();
        let mut my_shares = if my_id == Self::KING_ID {
            // Accumulate the result
            let mut acc = vec![F::zero(); len];
            for (other_id, lagrange) in self.mul_lagrange_2t.iter().enumerate() {
                if other_id == Self::KING_ID {
                    for (acc, muls) in izip!(&mut acc, &inputs) {
                        *acc += *muls * lagrange;
                    }
                } else {
                    let r = self.network.recv_many::<F>(other_id).await?;
                    if r.len() != len {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,"During execution of degree_reduce_vec in MPC: Invalid number of elements received",
                        ));
                    }
                    for (acc, muls) in izip!(&mut acc, r) {
                        *acc += muls * lagrange;
                    }
                }
            }
            // So far parties who do not require sending, do not send, so no receive here

            // Send fresh shares
            let mut shares = (0..self.network.get_num_parties())
                .map(|_| Vec::with_capacity(len))
                .collect::<Vec<_>>();

            for acc in acc {
                let s = core::share(
                    acc,
                    self.network.get_num_parties(),
                    self.threshold,
                    &mut self.rng,
                );
                for (des, src) in izip!(&mut shares, s) {
                    des.push(src);
                }
            }

            let mut my_share = Vec::new();
            for (other_id, share) in shares.into_iter().enumerate() {
                if my_id == other_id {
                    my_share = share;
                } else {
                    self.network.send_many(other_id, &share).await?;
                }
            }
            my_share
        } else {
            if my_id <= self.threshold * 2 {
                // Only send if my items are required
                self.network.send_many(Self::KING_ID, &inputs).await?;
            }
            let r = self.network.recv_many::<F>(Self::KING_ID).await?;
            if r.len() != len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,"During execution of degree_reduce_vec in MPC: Invalid number of elements received",
                ));
            }
            r
        };

        for (share, r) in izip!(&mut my_shares, r_ts) {
            *share -= r;
        }
        Ok(ShamirShare::convert_vec_rev(my_shares))
    }

    pub(crate) async fn degree_reduce_point<C>(
        &mut self,
        mut input: C,
    ) -> std::io::Result<ShamirPointShare<C>>
    where
        C: CurveGroup + std::ops::Mul<F, Output = C> + for<'a> std::ops::Mul<&'a F, Output = C>,
    {
        let (r_t, r_2t) = self.get_pair()?;
        let r_t = C::generator().mul(r_t);
        let r_2t = C::generator().mul(r_2t);

        input += r_2t;
        let my_id = self.network.get_id();

        let my_share = if my_id == Self::KING_ID {
            // Accumulate the result
            let mut acc = C::zero();
            for (other_id, lagrange) in self.mul_lagrange_2t.iter().enumerate() {
                if other_id == Self::KING_ID {
                    acc += input * lagrange;
                } else {
                    let r = self.network.recv::<C>(other_id).await?;
                    acc += r * lagrange;
                }
            }
            // So far parties who do not require sending, do not send, so no receive here

            // Send fresh shares
            let shares = core::share_point(
                acc,
                self.network.get_num_parties(),
                self.threshold,
                &mut self.rng,
            );
            let mut my_share = C::default();
            for (other_id, share) in shares.into_iter().enumerate() {
                if my_id == other_id {
                    my_share = share;
                } else {
                    self.network.send(other_id, share).await?;
                }
            }
            my_share
        } else {
            if my_id <= self.threshold * 2 {
                // Only send if my items are required
                self.network.send(Self::KING_ID, input).await?;
            }
            self.network.recv(Self::KING_ID).await?
        };

        Ok(ShamirPointShare::new(my_share - r_t))
    }
}
