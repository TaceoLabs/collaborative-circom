use acir::{native_types::Expression, AcirField};
use mpc_core::traits::NoirWitnessExtensionProtocol;

use crate::types::WitnessState;

use super::{CoAcvmError, CoAcvmResult, CoSolver};

impl<T, F> CoSolver<T, F>
where
    T: NoirWitnessExtensionProtocol<F>,
    F: AcirField,
{
    fn evaluate_mul_terms(
        &mut self,
        expr: &Expression<F>,
        acc: &mut Expression<T::AcvmType>,
    ) -> CoAcvmResult<()> {
        tracing::trace!("evaluating mul terms for simplification");
        if expr.mul_terms.is_empty() {
            tracing::trace!("no mul term. we are done");
            Ok(())
        } else if expr.mul_terms.len() == 1 {
            let (c, lhs, rhs) = &expr.mul_terms[0];
            tracing::trace!("looking at mul term {c} * _{} * _{}", lhs.0, rhs.0);
            if c.is_zero() {
                tracing::trace!("c is zero. We can skip this mul term");
                Ok(())
            } else {
                match (self.witness_map.get(lhs), self.witness_map.get(rhs)) {
                    // we could batch this multiplication but our currently planed network design
                    // should solve this without batching
                    (WitnessState::Known(lhs), WitnessState::Known(rhs)) => {
                        tracing::trace!("solving mul term...");
                        self.driver.solve_mul_term(*c, lhs, rhs, &mut acc.q_c)?;
                    }
                    (WitnessState::Known(lhs), WitnessState::Unknown) => {
                        tracing::trace!("partially solving mul term...");
                        let partly_solved = self.driver.acvm_mul_with_public(*c, lhs)?;
                        acc.linear_combinations.push((partly_solved, *rhs));
                    }
                    (WitnessState::Unknown, WitnessState::Known(rhs)) => {
                        tracing::trace!("partially solving mul term...");
                        let partly_solved = self.driver.acvm_mul_with_public(*c, rhs)?;
                        acc.linear_combinations.push((partly_solved, *lhs));
                    }
                    (WitnessState::Unknown, WitnessState::Unknown) => {
                        tracing::debug!(
                            "two unknowns in evaluate mul term. Not solvable for expr: {:?}",
                            expr
                        );
                        return Err(CoAcvmError::TooManyUnknowns);
                    }
                };
                tracing::trace!("after eval mul term: {acc:?}");
                Ok(())
            }
        } else {
            tracing::debug!("more than one mul term found!");
            Err(CoAcvmError::TooManyMulTerm(expr.mul_terms.len()))
        }
    }

    fn evaluate_linear_terms(&mut self, expr: &Expression<F>, acc: &mut Expression<T::AcvmType>) {
        for term in expr.linear_combinations.iter() {
            let (q_l, w_l) = term;
            tracing::trace!("looking at linear term: {q_l} * _{}..", w_l.0);
            match self.witness_map.get(w_l) {
                WitnessState::Known(w_l) => {
                    tracing::trace!("is known! reduce it");
                    self.driver.solve_linear_term(*q_l, w_l, &mut acc.q_c);
                }
                WitnessState::Unknown => {
                    tracing::trace!("is unknown!");
                    acc.linear_combinations
                        .push((T::AcvmType::from(*q_l), *w_l))
                }
            }
        }
    }

    fn simplify_expression(
        &mut self,
        expr: &Expression<F>,
    ) -> CoAcvmResult<Expression<T::AcvmType>> {
        tracing::trace!("simplifying expression...");
        // default implementation not exposed if we not have AcirField trait bound
        let mut simplified = Expression {
            mul_terms: vec![],
            linear_combinations: vec![],
            q_c: T::public_zero(),
        };
        // evaluate mul terms
        self.evaluate_mul_terms(expr, &mut simplified)?;
        // evaluate linear terms
        self.evaluate_linear_terms(expr, &mut simplified);
        // add constants
        self.driver
            .acvm_add_assign_with_public(expr.q_c, &mut simplified.q_c);

        Ok(simplified)
    }

    pub(super) fn solve_assert_zero(&mut self, expr: &Expression<F>) -> CoAcvmResult<()> {
        //first evaluate the already existing terms
        tracing::trace!("solving assert zero: {}", expr);
        let simplified = self.simplify_expression(expr)?;
        tracing::trace!("simplified expr:     {:?}", simplified);
        // if we are here, we do not have any mul terms
        debug_assert!(simplified.mul_terms.is_empty());
        // also if we are here and have more than one linear combination, we
        // cannot solve the expression
        if simplified.linear_combinations.is_empty() {
            //I think we are done??????
            Ok(())
        } else if simplified.linear_combinations.len() == 1 {
            //we can solve it!
            let (q_l, w_l) = simplified.linear_combinations[0].clone();
            let witness = self.driver.solve_equation(q_l, simplified.q_c)?;
            self.witness_map.insert(&w_l, witness);
            Ok(())
        } else {
            tracing::debug!("too many unknowns. not solvable for expression: {:?}", expr);
            Err(CoAcvmError::TooManyUnknowns)
        }
    }
}
