use super::CoUltraCircuitBuilder;
use crate::mpc::NoirUltraHonkProver;
use crate::parse::types::TraceData;
use crate::types::Polynomials;
use crate::types::ProverWitnessEntities;
use crate::types::ProvingKey;
use ark_ec::pairing::Pairing;
use ark_ff::One;
use eyre::Result;
use std::marker::PhantomData;
use ultrahonk::prelude::Crs;
use ultrahonk::prelude::HonkProofResult;
use ultrahonk::prelude::PrecomputedEntities;
use ultrahonk::prelude::ProverCrs;
use ultrahonk::prelude::ProvingKey as PlainProvingKey;
use ultrahonk::prelude::UltraCircuitVariable;
use ultrahonk::prelude::VerifyingKey;
use ultrahonk::Utils;

impl<T: NoirUltraHonkProver<P>, P: Pairing> ProvingKey<T, P> {
    const PUBLIC_INPUT_WIRE_INDEX: usize =
        ProverWitnessEntities::<T::ArithmeticShare, P::ScalarField>::W_R;

    // We ignore the TraceStructure for now (it is None in barretenberg for UltraHonk)
    pub fn create(
        id: T::PartyID,
        mut circuit: CoUltraCircuitBuilder<T, P>,
        crs: ProverCrs<P>,
    ) -> Self {
        tracing::info!("ProvingKey create");
        circuit.add_gates_to_ensure_all_polys_are_non_zero();
        circuit.finalize_circuit();

        let dyadic_circuit_size = circuit.compute_dyadic_size();
        let mut proving_key = Self::new(dyadic_circuit_size, circuit.public_inputs.len(), crs);
        // Construct and add to proving key the wire, selector and copy constraint polynomials
        proving_key.populate_trace(id, &mut circuit, false);

        // First and last lagrange polynomials (in the full circuit size)
        proving_key.polynomials.precomputed.lagrange_first_mut()[0] = P::ScalarField::one();
        proving_key.polynomials.precomputed.lagrange_last_mut()[dyadic_circuit_size - 1] =
            P::ScalarField::one();

        PlainProvingKey::construct_lookup_table_polynomials(
            proving_key
                .polynomials
                .precomputed
                .get_table_polynomials_mut(),
            &circuit,
            dyadic_circuit_size,
            0,
        );
        PlainProvingKey::construct_lookup_read_counts(
            proving_key
                .polynomials
                .witness
                .lookup_read_counts_and_tags_mut()
                .try_into()
                .unwrap(),
            &mut circuit,
            dyadic_circuit_size,
        );

        // Construct the public inputs array
        let block = circuit.blocks.get_pub_inputs();
        assert!(block.is_pub_inputs);
        for var_idx in block.wires[Self::PUBLIC_INPUT_WIRE_INDEX]
            .iter()
            .take(proving_key.num_public_inputs as usize)
            .cloned()
        {
            let var = circuit.get_variable(var_idx as usize);
            proving_key.public_inputs.push(var.public_into_field());
        }

        proving_key
    }

    pub fn create_keys(
        id: T::PartyID,
        circuit: CoUltraCircuitBuilder<T, P>,
        crs: Crs<P>,
    ) -> HonkProofResult<(Self, VerifyingKey<P>)> {
        let prover_crs = ProverCrs {
            monomials: crs.monomials,
        };
        let verifier_crs = crs.g2_x;

        let pk = ProvingKey::create(id, circuit, prover_crs);
        let circuit_size = pk.circuit_size;

        let mut commitments = PrecomputedEntities::default();
        for (des, src) in commitments
            .iter_mut()
            .zip(pk.polynomials.precomputed.iter())
        {
            let comm = Utils::commit(src.as_ref(), &pk.crs)?;
            *des = P::G1Affine::from(comm);
        }

        // Create and return the VerifyingKey instance
        let vk = VerifyingKey {
            crs: verifier_crs,
            circuit_size,
            num_public_inputs: pk.num_public_inputs,
            pub_inputs_offset: pk.pub_inputs_offset,
            commitments,
        };

        Ok((pk, vk))
    }

    pub fn get_public_inputs(&self) -> Vec<P::ScalarField> {
        self.public_inputs.clone()
    }

    pub fn get_prover_crs(
        circuit: &CoUltraCircuitBuilder<T, P>,
        path_g1: &str,
    ) -> Result<ProverCrs<P>> {
        PlainProvingKey::get_prover_crs(circuit, path_g1)
    }

    pub fn get_crs(
        circuit: &CoUltraCircuitBuilder<T, P>,
        path_g1: &str,
        path_g2: &str,
    ) -> Result<Crs<P>> {
        PlainProvingKey::get_crs(circuit, path_g1, path_g2)
    }

    fn new(circuit_size: usize, num_public_inputs: usize, crs: ProverCrs<P>) -> Self {
        tracing::info!("ProvingKey new");
        let polynomials = Polynomials::new(circuit_size);

        Self {
            crs,
            circuit_size: circuit_size as u32,
            public_inputs: Vec::with_capacity(num_public_inputs),
            num_public_inputs: num_public_inputs as u32,
            pub_inputs_offset: 0,
            polynomials,
            memory_read_records: Vec::new(),
            memory_write_records: Vec::new(),
            phantom: PhantomData,
        }
    }

    fn populate_trace(
        &mut self,
        id: T::PartyID,
        builder: &mut CoUltraCircuitBuilder<T, P>,
        is_strucutred: bool,
    ) {
        tracing::info!("Populating trace");

        let mut trace_data = TraceData::new(builder, self);
        trace_data.construct_trace_data(id, builder, is_strucutred);

        let ram_rom_offset = trace_data.ram_rom_offset;
        let copy_cycles = trace_data.copy_cycles;
        self.pub_inputs_offset = trace_data.pub_inputs_offset;

        PlainProvingKey::add_memory_records_to_proving_key(
            ram_rom_offset,
            builder,
            &mut self.memory_read_records,
            &mut self.memory_write_records,
        );

        // Compute the permutation argument polynomials (sigma/id) and add them to proving key
        PlainProvingKey::compute_permutation_argument_polynomials(
            &mut self.polynomials.precomputed,
            builder,
            copy_cycles,
            self.circuit_size as usize,
            self.pub_inputs_offset as usize,
        );
    }
}
