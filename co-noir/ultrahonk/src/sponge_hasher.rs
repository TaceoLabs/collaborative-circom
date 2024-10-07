use ark_ff::{One, PrimeField};
use num_bigint::BigUint;

pub trait FieldHash<F: PrimeField, const T: usize> {
    #[allow(unused)]
    fn permutation(&self, input: &[F; T]) -> [F; T] {
        let mut state = *input;
        self.permutation_in_place(&mut state);
        state
    }
    fn permutation_in_place(&self, input: &mut [F; T]);
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SpongeMode {
    Absorb,
    Squeeze,
}

pub struct FieldSponge<F: PrimeField, const T: usize, const R: usize, H: FieldHash<F, T>> {
    state: [F; T],
    cache: [F; R],
    cache_size: usize,
    mode: SpongeMode,
    hasher: H,
}

impl<F: PrimeField, const T: usize, const R: usize, H: FieldHash<F, T>> FieldSponge<F, T, R, H>
where
    H: Default,
{
    pub(crate) fn new(iv: F) -> Self {
        assert!(R < T);
        let mut state = [F::zero(); T];
        state[R] = iv;

        Self {
            state,
            cache: [F::zero(); R],
            cache_size: 0,
            mode: SpongeMode::Absorb,
            hasher: H::default(),
        }
    }

    fn perform_duplex(&mut self) -> [F; R] {
        // zero-pad the cache
        for i in self.cache_size..R {
            self.cache[i] = F::zero();
        }
        // add the cache into sponge state
        for i in 0..R {
            self.state[i] += self.cache[i];
        }
        self.hasher.permutation_in_place(&mut self.state);
        // return `rate` number of field elements from the sponge state.
        let mut output = [F::zero(); R];
        output.copy_from_slice(&self.state[..R]);

        output
    }

    fn absorb(&mut self, input: &F) {
        if self.mode == SpongeMode::Absorb && self.cache_size == R {
            // If we're absorbing, and the cache is full, apply the sponge permutation to compress the cache
            self.perform_duplex();
            self.cache[0] = *input;
            self.cache_size = 1;
        } else if self.mode == SpongeMode::Absorb && self.cache_size < R {
            // If we're absorbing, and the cache is not full, add the input into the cache
            self.cache[self.cache_size] = *input;
            self.cache_size += 1;
        } else if self.mode == SpongeMode::Squeeze {
            // If we're in squeeze mode, switch to absorb mode and add the input into the cache.
            // N.B. I don't think this code path can be reached?!
            self.cache[0] = *input;
            self.cache_size = 1;
            self.mode = SpongeMode::Absorb;
        }
    }

    fn squeeze(&mut self) -> F {
        if self.mode == SpongeMode::Squeeze && self.cache_size == 0 {
            // If we're in squeze mode and the cache is empty, there is nothing left to squeeze out of the sponge!
            // Switch to absorb mode.
            self.mode = SpongeMode::Absorb;
            self.cache_size = 0;
        }
        if self.mode == SpongeMode::Absorb {
            // If we're in absorb mode, apply sponge permutation to compress the cache, populate cache with compressed
            // state and switch to squeeze mode. Note: this code block will execute if the previous `if` condition was
            // matched
            self.cache = self.perform_duplex();
            self.cache_size = R;
        }
        // By this point, we should have a non-empty cache. Pop one item off the top of the cache and return it.
        let result = self.cache[0];
        for i in 1..self.cache_size {
            self.cache[i - 1] = self.cache[i];
        }
        self.cache_size -= 1;
        self.cache[self.cache_size] = F::zero();
        result
    }

    /**
     * @brief Use the sponge to hash an input string
     *
     * @tparam out_len
     * @tparam is_variable_length. Distinguishes between hashes where the preimage length is constant/not constant
     * @param input
     * @return std::array<FF, out_len>
     */
    pub(crate) fn hash_internal<const OUT_LEN: usize, const IS_VAR_LEN: bool>(
        input: &[F],
    ) -> [F; OUT_LEN] {
        let in_len = input.len();
        let iv = (BigUint::from(in_len) << 64) + OUT_LEN - BigUint::one();

        let mut sponge = Self::new(F::from(iv));
        for input in input.iter() {
            sponge.absorb(input);
        }

        // In the case where the hash preimage is variable-length, we append `1` to the end of the input, to distinguish
        // from fixed-length hashes. (the combination of this additional field element + the hash IV ensures
        // fixed-length and variable-length hashes do not collide)
        if IS_VAR_LEN {
            sponge.absorb(&F::one());
        }

        let mut res = [F::zero(); OUT_LEN];
        for r in res.iter_mut() {
            *r = sponge.squeeze();
        }
        res
    }

    pub(crate) fn hash_fixed_lenth<const OUT_LEN: usize>(input: &[F]) -> [F; OUT_LEN] {
        Self::hash_internal::<OUT_LEN, false>(input)
    }

    #[allow(unused)]
    pub(crate) fn hash_variable_length<const OUT_LEN: usize>(input: &[F]) -> [F; OUT_LEN] {
        Self::hash_internal::<OUT_LEN, true>(input)
    }
}
