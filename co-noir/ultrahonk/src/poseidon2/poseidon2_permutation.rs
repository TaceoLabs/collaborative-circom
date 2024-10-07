use super::poseidon2_params::Poseidon2Params;
use crate::sponge_hasher::FieldHash;
use ark_ff::PrimeField;

#[derive(Clone, Debug)]
pub struct Poseidon2<F: PrimeField, const T: usize, const D: u64> {
    pub(crate) params: &'static Poseidon2Params<F, T, D>,
}

impl<F: PrimeField, const T: usize, const D: u64> Poseidon2<F, T, D> {
    pub(crate) fn new(params: &'static Poseidon2Params<F, T, D>) -> Self {
        Self { params }
    }

    fn sbox(input: &mut [F; T]) {
        input.iter_mut().for_each(Self::single_sbox);
    }

    fn single_sbox(input: &mut F) {
        match D {
            3 => {
                let input2 = input.square();
                *input *= input2;
            }
            5 => {
                let input2 = input.square();
                let input4 = input2.square();
                *input *= input4;
            }
            7 => {
                let input2 = input.square();
                let input4 = input2.square();
                *input *= input4;
                *input *= input2;
            }
            _ => {
                *input = input.pow([D]);
            }
        }
    }

    /**
     * hardcoded algorithm that evaluates matrix multiplication using the following MDS matrix:
     * /         \
     * | 5 7 1 3 |
     * | 4 6 1 1 |
     * | 1 3 5 7 |
     * | 1 1 4 6 |
     * \         /
     *
     * Algorithm is taken directly from the Poseidon2 paper.
     */
    fn matmul_m4(input: &mut [F; 4]) {
        let t_0 = input[0] + input[1]; // A + B
        let t_1 = input[2] + input[3]; // C + D
        let t_2 = input[1].double() + t_1; // 2B + C + D
        let t_3 = input[3].double() + t_0; // A + B + 2D
        let t_4 = t_1.double().double() + t_3; // A + B + 4C + 6D
        let t_5 = t_0.double().double() + t_2; // 4A + 6B + C + D
        let t_6 = t_3 + t_5; // 5A + 7B + C + 3D
        let t_7 = t_2 + t_4; // A + 3B + 5C + 7D
        input[0] = t_6;
        input[1] = t_5;
        input[2] = t_7;
        input[3] = t_4;
    }

    fn matmul_external(input: &mut [F; T]) {
        match T {
            2 => {
                // Matrix circ(2, 1)
                let sum = input[0] + input[1];
                input[0] += &sum;
                input[1] += sum;
            }
            3 => {
                // Matrix circ(2, 1, 1)
                let sum = input[0] + input[1] + input[2];
                input[0] += &sum;
                input[1] += &sum;
                input[2] += sum;
            }
            4 => {
                Self::matmul_m4(input.as_mut_slice().try_into().unwrap());
            }
            8 | 12 | 16 | 20 | 24 => {
                // Applying cheap 4x4 MDS matrix to each 4-element part of the state
                for state in input.chunks_exact_mut(4) {
                    Self::matmul_m4(state.try_into().unwrap());
                }

                // Applying second cheap matrix for t > 4
                let mut stored = [F::zero(); 4];
                for l in 0..4 {
                    stored[l] = input[l];
                    for j in 1..T / 4 {
                        stored[l] += input[4 * j + l];
                    }
                }
                for i in 0..T {
                    input[i] += stored[i % 4];
                }
            }
            _ => {
                panic!("Invalid Statesize");
            }
        }
    }

    fn matmul_internal(&self, input: &mut [F; T]) {
        // Compute input sum
        let sum: F = input.iter().sum();
        // Add sum + diag entry * element to each element

        for (s, m) in input
            .iter_mut()
            .zip(self.params.mat_internal_diag_m_1.iter())
        {
            *s *= m;
            *s += sum;
        }
    }

    fn add_rc_external(&self, input: &mut [F; T], rc_offset: usize) {
        for (s, rc) in input
            .iter_mut()
            .zip(self.params.round_constants_external[rc_offset].iter())
        {
            *s += rc;
        }
    }

    fn add_rc_internal(&self, input: &mut [F; T], rc_offset: usize) {
        input[0] += &self.params.round_constants_internal[rc_offset];
    }
}

impl<F: PrimeField, const T: usize, const D: u64> FieldHash<F, T> for Poseidon2<F, T, D> {
    fn permutation_in_place(&self, state: &mut [F; T]) {
        // Linear layer at beginning
        Self::matmul_external(state);

        // First set of external rounds
        for r in 0..self.params.rounds_f_beginning {
            self.add_rc_external(state, r);
            Self::sbox(state);
            Self::matmul_external(state);
        }

        // Internal rounds
        for r in 0..self.params.rounds_p {
            self.add_rc_internal(state, r);
            Self::single_sbox(&mut state[0]);
            self.matmul_internal(state);
        }

        // Remaining external rounds
        for r in self.params.rounds_f_beginning
            ..self.params.rounds_f_beginning + self.params.rounds_f_end
        {
            self.add_rc_external(state, r);
            Self::sbox(state);
            Self::matmul_external(state);
        }
    }
}

impl Default for Poseidon2<ark_bn254::Fr, 4, 5> {
    fn default() -> Self {
        let params = &crate::poseidon2::poseidon2_bn254::POSEIDON2_BN254_T4_PARAMS;
        Poseidon2::new(params)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{poseidon2::poseidon2_bn254::POSEIDON2_BN254_T4_PARAMS, Utils};
    use rand::thread_rng;

    const TESTRUNS: usize = 10;

    fn poseidon2_kat<F: PrimeField, const T: usize, const D: u64>(
        params: &'static Poseidon2Params<F, T, D>,
        input: &[F; T],
        expected: &[F; T],
    ) {
        let poseidon2 = Poseidon2::new(params);
        let result = poseidon2.permutation(input);
        assert_eq!(&result, expected);
    }

    fn poseidon2_consistent_perm<F: PrimeField, const T: usize, const D: u64>(
        params: &'static Poseidon2Params<F, T, D>,
    ) {
        let mut rng = &mut thread_rng();
        let input1: Vec<F> = (0..T).map(|_| F::rand(&mut rng)).collect();
        let mut input2 = input1.clone();
        input2.rotate_right(T / 2);

        let poseidon2 = Poseidon2::new(params);
        let perm1 = poseidon2.permutation(input1.as_slice().try_into().unwrap());
        let perm2 = poseidon2.permutation(&input1.try_into().unwrap());
        let perm3 = poseidon2.permutation(&input2.try_into().unwrap());

        assert_eq!(perm1, perm2);
        assert_ne!(perm1, perm3);
    }

    #[test]
    fn posedon2_bn254_t4_consistent_perm() {
        for _ in 0..TESTRUNS {
            poseidon2_consistent_perm(&POSEIDON2_BN254_T4_PARAMS);
        }
    }

    #[test]
    fn posedon2_bn254_t4_kat1() {
        let input = [
            ark_bn254::Fr::from(0u64),
            ark_bn254::Fr::from(1u64),
            ark_bn254::Fr::from(2u64),
            ark_bn254::Fr::from(3u64),
        ];
        let expected = [
            Utils::field_from_hex_string(
                "0x01bd538c2ee014ed5141b29e9ae240bf8db3fe5b9a38629a9647cf8d76c01737",
            )
            .unwrap(),
            Utils::field_from_hex_string(
                "0x239b62e7db98aa3a2a8f6a0d2fa1709e7a35959aa6c7034814d9daa90cbac662",
            )
            .unwrap(),
            Utils::field_from_hex_string(
                "0x04cbb44c61d928ed06808456bf758cbf0c18d1e15a7b6dbc8245fa7515d5e3cb",
            )
            .unwrap(),
            Utils::field_from_hex_string(
                "0x2e11c5cff2a22c64d01304b778d78f6998eff1ab73163a35603f54794c30847a",
            )
            .unwrap(),
        ];

        poseidon2_kat(&POSEIDON2_BN254_T4_PARAMS, &input, &expected);
    }

    #[test]
    fn posedon2_bn254_t4_kat2() {
        let input = [
            Utils::field_from_hex_string(
                "9a807b615c4d3e2fa0b1c2d3e4f56789fedcba9876543210abcdef0123456789",
            )
            .unwrap(),
            Utils::field_from_hex_string(
                "9a807b615c4d3e2fa0b1c2d3e4f56789fedcba9876543210abcdef0123456789",
            )
            .unwrap(),
            Utils::field_from_hex_string(
                "0x9a807b615c4d3e2fa0b1c2d3e4f56789fedcba9876543210abcdef0123456789",
            )
            .unwrap(),
            Utils::field_from_hex_string(
                "0x9a807b615c4d3e2fa0b1c2d3e4f56789fedcba9876543210abcdef0123456789",
            )
            .unwrap(),
        ];
        let expected = [
            Utils::field_from_hex_string(
                "0x2bf1eaf87f7d27e8dc4056e9af975985bccc89077a21891d6c7b6ccce0631f95",
            )
            .unwrap(),
            Utils::field_from_hex_string(
                "0x0c01fa1b8d0748becafbe452c0cb0231c38224ea824554c9362518eebdd5701f",
            )
            .unwrap(),
            Utils::field_from_hex_string(
                "0x018555a8eb50cf07f64b019ebaf3af3c925c93e631f3ecd455db07bbb52bbdd3",
            )
            .unwrap(),
            Utils::field_from_hex_string(
                "0x0cbea457c91c22c6c31fd89afd2541efc2edf31736b9f721e823b2165c90fd41",
            )
            .unwrap(),
        ];

        poseidon2_kat(&POSEIDON2_BN254_T4_PARAMS, &input, &expected);
    }
}
