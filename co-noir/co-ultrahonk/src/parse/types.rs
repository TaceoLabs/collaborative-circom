use super::CoUltraCircuitBuilder;
use crate::{mpc::NoirUltraHonkProver, types::ProvingKey};
use ark_ec::pairing::Pairing;
use ultrahonk::prelude::{CycleNode, CyclicPermutation, Polynomial, NUM_SELECTORS, NUM_WIRES};

pub(crate) struct TraceData<'a, T: NoirUltraHonkProver<P>, P: Pairing> {
    pub(crate) wires: [&'a mut Polynomial<T::ArithmeticShare>; NUM_WIRES],
    pub(crate) selectors: [&'a mut Polynomial<P::ScalarField>; NUM_SELECTORS],
    pub(crate) copy_cycles: Vec<CyclicPermutation>,
    pub(crate) ram_rom_offset: u32,
    pub(crate) pub_inputs_offset: u32,
}

impl<'a, T: NoirUltraHonkProver<P>, P: Pairing> TraceData<'a, T, P> {
    pub(crate) fn new(
        builder: &CoUltraCircuitBuilder<T, P>,
        proving_key: &'a mut ProvingKey<T, P>,
    ) -> Self {
        let mut iter = proving_key.polynomials.witness.get_wires_mut().iter_mut();
        let wires = [
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
        ];

        let mut iter = proving_key
            .polynomials
            .precomputed
            .get_selectors_mut()
            .iter_mut();
        let selectors = [
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
        ];
        let copy_cycles = vec![vec![]; builder.variables.len()];

        Self {
            wires,
            selectors,
            copy_cycles,
            ram_rom_offset: 0,
            pub_inputs_offset: 0,
        }
    }

    pub(crate) fn construct_trace_data(
        &mut self,
        id: T::PartyID,
        builder: &mut CoUltraCircuitBuilder<T, P>,
        is_structured: bool,
    ) {
        tracing::info!("Construct trace data");
        // Complete the public inputs execution trace block from builder.public_inputs
        builder.populate_public_inputs_block();

        let mut offset = 1; // Offset at which to place each block in the trace polynomials
                            // For each block in the trace, populate wire polys, copy cycles and selector polys

        for block in builder.blocks.get() {
            let block_size = block.len();

            // Update wire polynomials and copy cycles
            // NB: The order of row/column loops is arbitrary but needs to be row/column to match old copy_cycle code

            for block_row_idx in 0..block_size {
                for wire_idx in 0..NUM_WIRES {
                    let var_idx = block.wires[wire_idx][block_row_idx] as usize; // an index into the variables array
                    let real_var_idx = builder.real_variable_index[var_idx] as usize;
                    let trace_row_idx = block_row_idx + offset;
                    // Insert the real witness values from this block into the wire polys at the correct offset
                    self.wires[wire_idx][trace_row_idx] =
                        builder.get_variable(var_idx).get_as_shared(id);
                    // Add the address of the witness value to its corresponding copy cycle
                    self.copy_cycles[real_var_idx].push(CycleNode {
                        wire_index: wire_idx as u32,
                        gate_index: trace_row_idx as u32,
                    });
                }
            }

            // Insert the selector values for this block into the selector polynomials at the correct offset
            // AZTEC TODO(https://github.com/AztecProtocol/barretenberg/issues/398): implicit arithmetization/flavor consistency
            for (selector_poly, selector) in self.selectors.iter_mut().zip(block.selectors.iter()) {
                debug_assert_eq!(selector.len(), block_size);

                for (src, des) in selector.iter().zip(selector_poly.iter_mut().skip(offset)) {
                    *des = *src;
                }
            }

            // Store the offset of the block containing RAM/ROM read/write gates for use in updating memory records
            if block.has_ram_rom {
                self.ram_rom_offset = offset as u32;
            }
            // Store offset of public inputs block for use in the pub(crate)input mechanism of the permutation argument
            if block.is_pub_inputs {
                self.pub_inputs_offset = offset as u32;
            }

            // If the trace is structured, we populate the data from the next block at a fixed block size offset
            if is_structured {
                offset += block.get_fixed_size() as usize;
            } else {
                // otherwise, the next block starts immediately following the previous one
                offset += block_size;
            }
        }
    }
}
