use acir::{acir_field::GenericFieldElement, native_types::Expression, AcirField};
use ark_ff::PrimeField;

use crate::{mpc::NoirWitnessExtensionProtocol, solver::solver_utils};

use super::{CoAcvmResult, CoSolver};

impl<T, F> CoSolver<T, F>
where
    T: NoirWitnessExtensionProtocol<F>,
    F: PrimeField,
{
    fn evaluate_mul_terms(
        &mut self,
        expr: &Expression<GenericFieldElement<F>>,
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
                match (
                    self.witness().get(lhs).cloned(),
                    self.witness().get(rhs).cloned(),
                ) {
                    // we could batch this multiplication but our currently planed network design
                    // should solve this without batching
                    (Some(lhs), Some(rhs)) => {
                        tracing::trace!("solving mul term...");
                        self.driver
                            .solve_mul_term(c.into_repr(), lhs, rhs, &mut acc.q_c)?;
                    }
                    (Some(lhs), None) => {
                        tracing::trace!("partially solving mul term...");
                        let partly_solved = self.driver.acvm_mul_with_public(c.into_repr(), lhs);
                        acc.linear_combinations.push((partly_solved, *rhs));
                    }
                    (None, Some(rhs)) => {
                        tracing::trace!("partially solving mul term...");
                        let partly_solved = self.driver.acvm_mul_with_public(c.into_repr(), rhs);
                        acc.linear_combinations.push((partly_solved, *lhs));
                    }
                    (None, None) => Err(eyre::eyre!(
                        "two unknowns in evaluate mul term. Not solvable for expr: {:?}",
                        expr
                    ))?,
                };
                tracing::trace!("after eval mul term: {acc:?}");
                Ok(())
            }
        } else {
            Err(eyre::eyre!("more than one mul term found!"))?
        }
    }

    fn evaluate_linear_terms(
        &mut self,
        expr: &Expression<GenericFieldElement<F>>,
        acc: &mut Expression<T::AcvmType>,
    ) {
        for term in expr.linear_combinations.iter() {
            let (q_l, w_l) = term;
            tracing::trace!("looking at linear term: {q_l} * _{}..", w_l.0);
            if let Some(w_l) = self.witness().get(w_l).cloned() {
                tracing::trace!("is known! reduce it");
                self.driver
                    .solve_linear_term(q_l.into_repr(), w_l, &mut acc.q_c);
            } else {
                tracing::trace!("is unknown!");
                acc.linear_combinations
                    .push((T::AcvmType::from(q_l.into_repr()), *w_l))
            }
        }
    }

    pub(crate) fn simplify_expression(
        &mut self,
        expr: &Expression<GenericFieldElement<F>>,
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
            .acvm_add_assign_with_public(expr.q_c.into_repr(), &mut simplified.q_c);

        Ok(simplified)
    }

    pub(super) fn solve_assert_zero(
        &mut self,
        expr: &Expression<GenericFieldElement<F>>,
    ) -> CoAcvmResult<()> {
        //first evaluate the already existing terms
        tracing::trace!(
            "solving assert zero: {}",
            solver_utils::expr_to_string(expr)
        );
        let simplified = self.simplify_expression(expr)?;
        tracing::trace!("simplified expr:     {:?}", simplified);
        // if we are here, we do not have any mul terms
        debug_assert!(simplified.mul_terms.is_empty());
        // also if we are here and have more than one linear combination, we
        // cannot solve the expression
        if simplified.linear_combinations.is_empty() {
            // we are done
            tracing::trace!("nothing to do for us");
            Ok(())
        } else if simplified.linear_combinations.len() == 1 {
            //we can solve it!
            tracing::trace!("solving equation...");
            let (q_l, w_l) = simplified.linear_combinations[0].clone();
            let witness = self.driver.solve_equation(q_l, simplified.q_c)?;
            self.witness().insert(w_l, witness);
            tracing::trace!("we did it!");
            Ok(())
        } else {
            Err(eyre::eyre!(
                "too many unknowns. not solvable for expression: {:?}",
                expr
            ))?
        }
    }

    pub(crate) fn evaluate_expression(
        &mut self,
        expr: &Expression<GenericFieldElement<F>>,
    ) -> CoAcvmResult<T::AcvmType> {
        Ok(self
            .simplify_expression(expr)?
            .to_const()
            .cloned()
            .ok_or(eyre::eyre!(
                "cannot evaluate expression to const - has unknown"
            ))?)
    }
}
