use ark_ff::PrimeField;
use itertools::{izip, Itertools};

use crate::RngType;
use rand::{Rng, SeedableRng};

use super::network::ShamirNetwork;

pub(super) struct ShamirRng<F> {
    pub(super) rng: RngType,
    pub(super) threshold: usize,
    pub(super) num_parties: usize,
    pub(super) shared_rngs: Vec<RngType>,
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

        Ok(Self {
            rng,
            threshold,
            num_parties,
            shared_rngs,
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

    // I use the following matrix:
    // [1, 1  , 1  , 1  , ..., 1  ]
    // [1, 2  , 3  , 4  , ..., n  ]
    // [1, 2^2, 3^2, 4^2, ..., n^2]
    // ...
    // [1, 2^t, 3^t, 4^t, ..., n^t]
    fn vandermonde_mul(inputs: &[F], res: &mut [F], num_parties: usize, threshold: usize) {
        debug_assert_eq!(inputs.len(), num_parties);
        debug_assert_eq!(res.len(), threshold + 1);

        let row = (1..=num_parties as u64).map(F::from).collect::<Vec<_>>();
        let mut current_row = row.clone();

        res[0] = inputs.iter().sum();

        for ri in res.iter_mut().skip(1) {
            *ri = F::zero();
            for (c, r, i) in izip!(&mut current_row, &row, inputs) {
                *ri += *c * i;
                *c *= r; // Update current_row
            }
        }
    }

    // Generates amount * (self.threshold + 1) random double shares
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
        let mut r_t = Vec::with_capacity(amount * (self.threshold + 1));
        let mut r_2t = Vec::with_capacity(amount * (self.threshold + 1));

        r_t.resize(amount * (self.threshold + 1), F::default());
        r_2t.resize(amount * (self.threshold + 1), F::default());

        // Now make vandermonde multiplication
        let r_t_chunks = r_t.chunks_exact_mut(self.threshold + 1);
        let r_2t_chunks = r_2t.chunks_exact_mut(self.threshold + 1);

        for (r_t_des, r_2t_des, r_t_src, r_2t_src) in
            izip!(r_t_chunks, r_2t_chunks, rcv_rt, rcv_r2t)
        {
            Self::vandermonde_mul(&r_t_src, r_t_des, self.num_parties, self.threshold);
            Self::vandermonde_mul(&r_2t_src, r_2t_des, self.num_parties, self.threshold);
        }

        self.r_t.extend(r_t);
        self.r_2t.extend(r_2t);

        Ok(())
    }
}
