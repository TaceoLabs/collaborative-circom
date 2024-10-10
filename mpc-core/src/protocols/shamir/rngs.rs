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

    fn receive_seeded(&mut self, degree: usize, output: &mut [Vec<F>]) {
        for i in 1..=degree {
            let send_id = (self.id + self.num_parties - i) % self.num_parties;
            let rng = self.get_rng_mut(send_id);
            for r in output.iter_mut() {
                r[send_id] = F::rand(rng);
            }
        }
    }

    fn receive_seeded_prev(&mut self, degree: usize, output: &mut [Vec<F>]) {
        for i in 1..=degree {
            let send_id = (self.id + self.num_parties - i) % self.num_parties;
            if send_id > self.id {
                continue;
            }
            let rng = self.get_rng_mut(send_id);
            for r in output.iter_mut() {
                r[send_id] = F::rand(rng);
            }
        }
    }

    fn receive_seeded_next(&mut self, degree: usize, output: &mut [Vec<F>]) {
        for i in 1..=degree {
            let send_id = (self.id + self.num_parties - i) % self.num_parties;
            if send_id < self.id {
                continue;
            }
            let rng = self.get_rng_mut(send_id);
            for r in output.iter_mut() {
                r[send_id] = F::rand(rng);
            }
        }
    }

    fn get_interpolation_polys(&mut self, my_rands: &[F], degree: usize) -> Vec<Vec<F>> {
        let amount = my_rands.len();
        let mut ids = Vec::with_capacity(degree + 1);
        let mut shares = (0..amount)
            .map(|_| Vec::with_capacity(degree + 1))
            .collect_vec();
        ids.push(0); // my randomness acts as the secret
        for (s, r) in shares.iter_mut().zip(my_rands.iter()) {
            s.push(*r);
        }
        for i in 1..=degree {
            let rcv_id = (self.id + i) % self.num_parties;
            ids.push(rcv_id + 1);
            let rng = self.get_rng_mut(rcv_id);
            for s in shares.iter_mut() {
                s.push(F::rand(rng));
            }
        }
        // Interpolate polys
        shares
            .into_iter()
            .map(|s| super::core::interpolate_poly::<F>(&s, &ids))
            .collect_vec()
    }

    fn set_my_share(&self, output: &mut [Vec<F>], polys: &[Vec<F>]) {
        for (r, p) in output.iter_mut().zip(polys.iter()) {
            r[self.id] = super::core::evaluate_poly(p, F::from(self.id as u64 + 1));
        }
    }

    async fn send_share_of_randomness<N: ShamirNetwork>(
        &self,
        degree: usize,
        polys: &[Vec<F>],
        network: &mut N,
    ) -> std::io::Result<()> {
        let sending = self.num_parties - degree - 1;
        let mut to_send = vec![F::zero(); polys.len()]; // Allocate buffer only once
        for i in 1..=sending {
            let rcv_id = (self.id + i + degree) % self.num_parties;
            for (des, p) in to_send.iter_mut().zip(polys.iter()) {
                *des = super::core::evaluate_poly(p, F::from(rcv_id as u64 + 1));
            }
            network.send_many(rcv_id, &to_send).await?;
        }
        Ok(())
    }

    async fn receive_share_of_randomness<N: ShamirNetwork>(
        &self,
        degree: usize,
        output: &mut [Vec<F>],
        network: &mut N,
    ) -> std::io::Result<()> {
        let sending = self.num_parties - degree - 1;
        for i in 1..=sending {
            let send_id = (self.id + self.num_parties - degree - i) % self.num_parties;
            let shares = network.recv_many(send_id).await?;
            for (r, s) in output.iter_mut().zip(shares.iter()) {
                r[send_id] = *s;
            }
        }
        Ok(())
    }

    async fn random_double_share<N: ShamirNetwork>(
        &mut self,
        amount: usize,
        network: &mut N,
    ) -> std::io::Result<(Vec<Vec<F>>, Vec<Vec<F>>)> {
        let mut rcv_t = vec![vec![F::default(); self.num_parties]; amount];
        let mut rcv_2t = vec![vec![F::default(); self.num_parties]; amount];

        // These are the parties for which I act as a receiver using the seeds
        self.receive_seeded(self.threshold, &mut rcv_t);

        // for my share I will use the seed for the next parties alongside mine
        let my_rands = (0..amount)
            .map(|_| F::rand(&mut self.rng))
            .collect::<Vec<_>>();
        let polys_t = self.get_interpolation_polys(&my_rands, self.threshold);

        // Do the same for rcv_2t (do afterwards due to seeds being used here)
        // Be careful about the order of calling the rngs
        self.receive_seeded_next(self.threshold * 2, &mut rcv_2t);
        let polys_2t = self.get_interpolation_polys(&my_rands, self.threshold * 2);
        self.receive_seeded_prev(self.threshold * 2, &mut rcv_2t);

        // Set my share
        self.set_my_share(&mut rcv_t, &polys_t);
        self.set_my_share(&mut rcv_2t, &polys_2t);

        // Send the share of my randomness
        self.send_share_of_randomness(self.threshold, &polys_t, network)
            .await?;
        self.send_share_of_randomness(self.threshold * 2, &polys_2t, network)
            .await?;

        // Receive the remaining shares
        self.receive_share_of_randomness(self.threshold, &mut rcv_t, network)
            .await?;
        self.receive_share_of_randomness(self.threshold * 2, &mut rcv_2t, network)
            .await?;

        Ok((rcv_t, rcv_2t))
    }

    fn get_random_double_shares_3_party(&mut self, amount: usize) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
        assert_eq!(self.num_parties, 3);
        assert_eq!(self.threshold, 1);

        let mut rcv_t = vec![vec![F::default(); 3]; amount];
        let mut rcv_2t = vec![vec![F::default(); 3]; amount];

        // These are the parties for which I act as a receiver using the seeds
        // Be careful about the order of calling the rngs
        self.receive_seeded_next(2, &mut rcv_t);

        // Generate
        let mut ids = [0; 2];
        let mut shares = vec![[F::zero(); 2]; amount];
        for i in 1..=2 {
            let rcv_id = (self.id + i) % 3;
            ids[i - 1] = rcv_id + 1;
            let rng = self.get_rng_mut(rcv_id);
            for s in shares.iter_mut() {
                s[i - 1] = F::rand(rng);
            }
        }

        // Receive the remaining now to clock rngs in the correct order
        self.receive_seeded_prev(2, &mut rcv_t);

        // Interpolate polys
        let polys_t = shares
            .into_iter()
            .map(|s| super::core::interpolate_poly::<F>(&s, &ids))
            .collect_vec();

        // Set my rand on the polynomial and calculate the share
        let mut rands = Vec::with_capacity(amount);
        for (r, p) in rcv_t.iter_mut().zip(polys_t.iter()) {
            r[self.id] = super::core::evaluate_poly(p, F::from(self.id as u64 + 1));
            rands.push(super::core::evaluate_poly(p, F::zero()));
        }

        // Do the same for rcv_2t (do afterwards due to seeds being used here)
        // Be careful about the order of calling the rngs
        Self::receive_seeded_next(self, 2, &mut rcv_2t);
        let polys_2t = self.get_interpolation_polys(&rands, 2);
        Self::receive_seeded_prev(self, 2, &mut rcv_2t);
        self.set_my_share(&mut rcv_2t, &polys_2t);

        (rcv_t, rcv_2t)
    }

    // Generates amount * num_parties random double shares
    // We use DN07 to generate t+1 double shares from the randomness of the n parties. Then we use Atlas to generate n double shares from the t+1 double shares. Without changing the King server in the Multiplications this only works in an semi-honest setting.
    // The matrix we use is a combined version of the DN07 and Atlas matrix, so we only have one matrix multiplication for both.
    pub(super) async fn buffer_triples<N: ShamirNetwork>(
        &mut self,
        network: &mut N,
        amount: usize,
    ) -> std::io::Result<()> {
        let (rcv_rt, rcv_r2t) = if self.num_parties == 3 && self.threshold == 1 {
            self.get_random_double_shares_3_party(amount)
        } else {
            self.random_double_share(amount, network).await?
        };

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
