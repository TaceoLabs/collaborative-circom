use std::cmp::Ordering;

use ark_ff::PrimeField;
use itertools::{izip, Itertools};

use crate::RngType;
use rand::{Rng, SeedableRng};

use super::network::ShamirNetwork;

pub(super) struct ShamirRng<F> {
    pub(super) id: usize,
    pub(super) rng: RngType,
    pub(super) threshold: usize,
    pub(super) num_parties: usize,
    pub(super) shared_rngs: Vec<RngType>,
    pub(super) atlas_dn_matrix: Vec<Vec<F>>,
    pub(super) r_t: Vec<F>,
    pub(super) r_2t: Vec<F>,
}

impl<F: PrimeField> ShamirRng<F> {
    pub async fn new<N: ShamirNetwork>(
        seed: [u8; crate::SEED_SIZE],
        threshold: usize,
        network: &mut N,
    ) -> std::io::Result<Self> {
        let mut rng = RngType::from_seed(seed);
        let num_parties = network.get_num_parties();

        let shared_rngs = Self::get_shared_rngs(network, &mut rng).await?;

        let atlas_dn_matrix = Self::generate_atlas_dn_matrix(num_parties, threshold);

        Ok(Self {
            id: network.get_id(),
            rng,
            threshold,
            num_parties,
            shared_rngs,
            atlas_dn_matrix,
            r_t: Vec::new(),
            r_2t: Vec::new(),
        })
    }

    async fn get_shared_rngs<N: ShamirNetwork>(
        network: &mut N,
        rng: &mut RngType,
    ) -> std::io::Result<Vec<RngType>> {
        type SeedType = [u8; crate::SEED_SIZE];
        let id = network.get_id();
        let num_parties = network.get_num_parties();

        let mut rngs = Vec::with_capacity(num_parties - 1);
        let mut seeds = vec![<SeedType>::default(); num_parties];
        let to_interact_with_parties = num_parties - 1;

        let mut send = to_interact_with_parties / 2;
        if to_interact_with_parties & 1 == 1 && id < num_parties / 2 {
            send += 1;
        }
        let receive = to_interact_with_parties - send;
        for id_off in 1..=send {
            let rcv_id = (id + id_off) % num_parties;
            let seed: SeedType = rng.gen();
            seeds[rcv_id] = seed;
            network.send(rcv_id, seed).await?;
        }
        for id_off in 1..=receive {
            let send_id = (id + num_parties - id_off) % num_parties;
            let seed = network.recv(send_id).await?;
            seeds[send_id] = seed;
        }

        let after = seeds.split_off(id);
        for seed in seeds {
            debug_assert_ne!(seed, SeedType::default());
            rngs.push(RngType::from_seed(seed));
        }
        debug_assert_eq!(after[0], SeedType::default());
        for seed in after.into_iter().skip(1) {
            debug_assert_ne!(seed, SeedType::default());
            rngs.push(RngType::from_seed(seed));
        }

        Ok(rngs)
    }

    // We use the following (t+1 x n) Vandermonde matrix for DN07:
    // [1, 1  , 1  , 1  , ..., 1  ]
    // [1, 2  , 3  , 4  , ..., n  ]
    // [1, 2^2, 3^2, 4^2, ..., n^2]
    // ...
    // [1, 2^t, 3^t, 4^t, ..., n^t]

    // We use the following (n x t+1) Vandermonde matrix for Atlas:
    // [1, 1  , 1  , 1  , ..., 1  ]
    // [1, 2  , 3  , 4  , ..., t  ]
    // [1, 2^2, 3^2, 4^2, ..., t^2]
    // ...
    // [1, 2^n, 3^n, 4^n, ..., t^n]

    // This gives the resulting (n x n) matrix = Atlas x DN07: Each cell (row, col) has the value: sum_{i=0}^{t} (i + 1) ^ row * (col + 1) ^ i
    fn generate_atlas_dn_matrix(num_parties: usize, threshold: usize) -> Vec<Vec<F>> {
        let mut result = Vec::with_capacity(num_parties);
        for row in 0..num_parties {
            let mut row_result = Vec::with_capacity(num_parties);
            for col in 0..num_parties {
                let mut val = F::zero();
                for i in 0..=threshold {
                    val += F::from(i as u64 + 1).pow([row as u64])
                        * F::from(col as u64 + 1).pow([i as u64]);
                }
                row_result.push(val);
            }
            result.push(row_result);
        }

        result
    }

    fn mamtul(mat: &[Vec<F>], inp: &[F], outp: &mut [F]) {
        debug_assert_eq!(outp.len(), mat.len());
        for (res, row) in outp.iter_mut().zip(mat.iter()) {
            debug_assert_eq!(row.len(), inp.len());
            for (v, cell) in inp.iter().cloned().zip(row.iter()) {
                *res += v * cell;
            }
        }
    }

    // get shared_rng_mut
    fn get_rng_mut(&mut self, other_id: usize) -> &mut RngType {
        match other_id.cmp(&self.id) {
            Ordering::Less => &mut self.shared_rngs[other_id],
            Ordering::Greater => &mut self.shared_rngs[other_id - 1],
            Ordering::Equal => &mut self.rng,
        }
    }

    async fn random_t_share<N: ShamirNetwork>(
        &mut self,
        amount: usize,
        network: &mut N,
    ) -> std::io::Result<Vec<Vec<F>>> {
        let mut rcv = vec![vec![F::default(); self.num_parties]; amount];

        // These are the parties for which I act as a receiver using the seeds
        for i in 1..=self.threshold {
            let send_id = (self.id + self.num_parties - i) & self.num_parties;
            for r in rcv.iter_mut() {
                r[send_id] = F::rand(self.get_rng_mut(send_id));
            }
        }
        // for my share I will use the seed for the next parties alongside mine
        let mut ids = Vec::with_capacity(self.threshold + 1);
        let mut shares = (0..amount)
            .map(|_| Vec::with_capacity(self.threshold + 1))
            .collect_vec();
        ids.push(0); // my randomness acts as the secret
        for s in shares.iter_mut() {
            s.push(F::rand(&mut self.rng));
        }
        for i in 1..=self.threshold {
            let rcv_id = (self.id + i) & self.num_parties;
            ids.push(rcv_id);
            for s in shares.iter_mut() {
                s.push(F::rand(self.get_rng_mut(rcv_id)));
            }
        }
        // Interpolate polys
        let polys = shares
            .into_iter()
            .map(|s| super::core::interpolate_poly::<F>(&s, &ids))
            .collect_vec();
        // Set my share
        for (r, p) in rcv.iter_mut().zip(polys.iter()) {
            r[self.id] = super::core::evaluate_poly(&p, F::from(self.id as u64 + 1));
        }
        // Send the share of my ranomness
        let sending = self.num_parties - self.threshold - 1;
        let mut to_send = vec![F::zero(); self.threshold + 1];
        for i in 1..=sending {
            let rcv_id = (self.id + i + self.threshold) & self.num_parties;
            for (des, p) in to_send.iter_mut().zip(polys.iter()) {
                *des = super::core::evaluate_poly(&p, F::from(rcv_id as u64 + 1));
            }
            network.send_many(rcv_id, &to_send).await?;
        }
        // Receive the remaining shares
        for i in 1..=sending {
            let send_id = (self.id + self.num_parties - self.threshold - i) & self.num_parties;
            let shares = network.recv_many(send_id).await?;
            for (r, s) in rcv.iter_mut().zip(shares.iter()) {
                r[send_id] = *s;
            }
        }

        Ok(rcv)
    }

    // Generates amount * num_parties random double shares
    // We use DN07 to generate t+1 double shares from the randomness of the n parties. Then we use Atlas to generate n double shares from the t+1 double shares. Without changing the King server in the Multiplications this only works in an semi-honest setting.
    // The matrix we use is a combined version of the DN07 and Atlas matrix, so we only have one matrix multiplication for both.
    pub(super) async fn buffer_triples<N: ShamirNetwork>(
        &mut self,
        network: &mut N,
        amount: usize,
    ) -> std::io::Result<()> {
        let rand = (0..amount)
            .map(|_| F::rand(&mut self.rng))
            .collect::<Vec<_>>();

        let mut send = (0..self.num_parties)
            .map(|_| Vec::with_capacity(amount * 2))
            .collect::<Vec<_>>();

        for r in rand {
            let shares_t = super::core::share(r, self.num_parties, self.threshold, &mut self.rng);
            let shares_2t =
                super::core::share(r, self.num_parties, 2 * self.threshold, &mut self.rng);

            for (des, src1, src2) in izip!(&mut send, shares_t, shares_2t) {
                des.push(src1);
                des.push(src2);
            }
        }

        let mut rcv_rt = (0..amount)
            .map(|_| Vec::with_capacity(self.num_parties))
            .collect_vec();
        let mut rcv_r2t = (0..amount)
            .map(|_| Vec::with_capacity(self.num_parties))
            .collect_vec();

        // TODO this sometimes runs fast, but often 1 party is fast and the rest take a lot longer
        let recv = network.send_and_recv_each_many(send).await?;

        for r in recv.into_iter() {
            for (des_r, des_r2, src) in izip!(&mut rcv_rt, &mut rcv_r2t, r.chunks_exact(2)) {
                des_r.push(src[0]);
                des_r2.push(src[1]);
            }
        }

        // reserve buffer
        let mut r_t = vec![F::default(); amount * self.num_parties];
        let mut r_2t = vec![F::default(); amount * self.num_parties];

        // Now make the matrix multiplication
        let r_t_chunks = r_t.chunks_exact_mut(self.num_parties);
        let r_2t_chunks = r_2t.chunks_exact_mut(self.num_parties);

        for (des, src) in izip!(r_t_chunks, rcv_rt) {
            Self::mamtul(&self.atlas_dn_matrix, &src, des);
        }
        for (des, src) in izip!(r_2t_chunks, rcv_r2t) {
            Self::mamtul(&self.atlas_dn_matrix, &src, des);
        }

        self.r_t.extend(r_t);
        self.r_2t.extend(r_2t);

        Ok(())
    }
}
