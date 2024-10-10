pub(crate) mod auxiliary_relation;
pub(crate) mod delta_range_constraint_relation;
pub(crate) mod elliptic_relation;
pub(crate) mod logderiv_lookup_relation;
pub(crate) mod permutation_relation;
pub(crate) mod poseidon2_external_relation;
pub(crate) mod poseidon2_internal_relation;
pub(crate) mod ultra_arithmetic_relation;

use crate::mpc::NoirUltraHonkProver;

use super::{
    types::{ProverUnivariates, RelationParameters},
    univariates::SharedUnivariate,
};
use ark_ec::pairing::Pairing;
use auxiliary_relation::{AuxiliaryRelation, AuxiliaryRelationAcc};
use delta_range_constraint_relation::{
    DeltaRangeConstraintRelation, DeltaRangeConstraintRelationAcc,
};
use elliptic_relation::{EllipticRelation, EllipticRelationAcc};
use logderiv_lookup_relation::{LogDerivLookupRelation, LogDerivLookupRelationAcc};
use permutation_relation::{UltraPermutationRelation, UltraPermutationRelationAcc};
use poseidon2_external_relation::{Poseidon2ExternalRelation, Poseidon2ExternalRelationAcc};
use poseidon2_internal_relation::{Poseidon2InternalRelation, Poseidon2InternalRelationAcc};
use ultra_arithmetic_relation::{UltraArithmeticRelation, UltraArithmeticRelationAcc};
use ultrahonk::prelude::{HonkCurve, HonkProofResult, TranscriptFieldType, Univariate};

pub(crate) trait Relation<T: NoirUltraHonkProver<P>, P: HonkCurve<TranscriptFieldType>> {
    type Acc: Default;
    const SKIPPABLE: bool;

    fn check_skippable() {
        if !Self::SKIPPABLE {
            panic!("Cannot skip this relation");
        }
    }

    fn skip(input: &ProverUnivariates<T, P>) -> bool;
    async fn accumulate(
        driver: &mut T,
        univariate_accumulator: &mut Self::Acc,
        input: &ProverUnivariates<T, P>,
        relation_parameters: &RelationParameters<P::ScalarField>,
        scaling_factor: &P::ScalarField,
    ) -> HonkProofResult<()>;
}

pub(crate) const NUM_SUBRELATIONS: usize = UltraArithmeticRelation::NUM_RELATIONS
    + UltraPermutationRelation::NUM_RELATIONS
    + DeltaRangeConstraintRelation::NUM_RELATIONS
    + EllipticRelation::NUM_RELATIONS
    + AuxiliaryRelation::NUM_RELATIONS
    + LogDerivLookupRelation::NUM_RELATIONS
    + Poseidon2ExternalRelation::NUM_RELATIONS
    + Poseidon2InternalRelation::NUM_RELATIONS;

pub const CRAND_PAIRS_FACTOR: usize = AuxiliaryRelation::CRAND_PAIRS_FACTOR
    + DeltaRangeConstraintRelation::CRAND_PAIRS_FACTOR
    + EllipticRelation::CRAND_PAIRS_FACTOR
    + LogDerivLookupRelation::CRAND_PAIRS_FACTOR
    + UltraPermutationRelation::CRAND_PAIRS_FACTOR
    + Poseidon2ExternalRelation::CRAND_PAIRS_FACTOR
    + Poseidon2InternalRelation::CRAND_PAIRS_FACTOR
    + UltraArithmeticRelation::CRAND_PAIRS_FACTOR;

pub(crate) struct AllRelationAcc<T: NoirUltraHonkProver<P>, P: Pairing> {
    pub(crate) r_arith: UltraArithmeticRelationAcc<T, P>,
    pub(crate) r_perm: UltraPermutationRelationAcc<T, P>,
    pub(crate) r_delta: DeltaRangeConstraintRelationAcc<T, P>,
    pub(crate) r_elliptic: EllipticRelationAcc<T, P>,
    pub(crate) r_aux: AuxiliaryRelationAcc<T, P>,
    pub(crate) r_lookup: LogDerivLookupRelationAcc<T, P>,
    pub(crate) r_pos_ext: Poseidon2ExternalRelationAcc<T, P>,
    pub(crate) r_pos_int: Poseidon2InternalRelationAcc<T, P>,
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> Default for AllRelationAcc<T, P> {
    fn default() -> Self {
        Self {
            r_arith: Default::default(),
            r_perm: Default::default(),
            r_delta: Default::default(),
            r_elliptic: Default::default(),
            r_aux: Default::default(),
            r_lookup: Default::default(),
            r_pos_ext: Default::default(),
            r_pos_int: Default::default(),
        }
    }
}

impl<T: NoirUltraHonkProver<P>, P: Pairing> AllRelationAcc<T, P> {
    pub(crate) fn scale(
        &mut self,
        driver: &mut T,
        first_scalar: P::ScalarField,
        elements: &[P::ScalarField],
    ) {
        assert!(elements.len() == NUM_SUBRELATIONS - 1);
        self.r_arith.scale(driver, &[first_scalar, elements[0]]);
        self.r_perm.scale(driver, &elements[1..3]);
        self.r_delta.scale(driver, &elements[3..7]);
        self.r_elliptic.scale(driver, &elements[7..9]);
        self.r_aux.scale(driver, &elements[9..15]);
        self.r_lookup.scale(driver, &elements[15..17]);
        self.r_pos_ext.scale(driver, &elements[17..21]);
        self.r_pos_int.scale(driver, &elements[21..]);
    }

    pub(crate) fn extend_and_batch_univariates<const SIZE: usize>(
        &self,
        driver: &mut T,
        result: &mut SharedUnivariate<T, P, SIZE>,
        extended_random_poly: &Univariate<P::ScalarField, SIZE>,
        partial_evaluation_result: &P::ScalarField,
    ) {
        self.r_arith.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
        );
        self.r_perm.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
        );
        self.r_delta.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
        );
        self.r_elliptic.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
        );
        self.r_aux.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
        );
        self.r_lookup.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
        );
        self.r_pos_ext.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
        );
        self.r_pos_int.extend_and_batch_univariates(
            driver,
            result,
            extended_random_poly,
            partial_evaluation_result,
        );
    }
}
