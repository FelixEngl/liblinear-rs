//! Types and traits that wrap LIBLINEAR hyper parameters.

use std::marker::PhantomData;
use crate::{
    errors::ModelError,
    solver::{
        traits::{
            CanDisableBiasRegularization, IsSingleClassSolver, IsSupportVectorRegressionSolver,
            IsTrainableSolver, Solver, SupportsInitialSolutions,
        },
        SolverOrdinal,
    },
};

use self::traits::{
    SetBiasRegularization, SetInitialSolutions, SetOutlierRatio, SetRegressionLossSensitivity,
};

#[cfg(feature = "serde")]
use ::serde::{Deserialize, Serialize};

/// Traits implemented by [`Parameters`].
pub mod traits {
    /// Implemented for parameters with solvers that implement the
    /// [`SupportsInitialSolutions`](crate::solver::traits::SupportsInitialSolutions) trait.
    pub trait SetInitialSolutions {
        /// Set the initial solution specification.
        ///
        /// Default: `None`
        fn initial_solutions(&mut self, init_solutions: Vec<f64>) -> &mut Self;
    }

    /// Implemented for parameters with solvers that implement the
    /// [`CanDisableBiasRegularization`](crate::solver::traits::CanDisableBiasRegularization) trait.
    pub trait SetBiasRegularization {
        /// Toggle bias regularization during training.
        ///
        /// If set to `false`, the bias value will automatically be set to `1`.
        ///
        /// Default: `true`
        fn bias_regularization(&mut self, bias_regularization: bool) -> &mut Self;
    }

    /// Implemented for parameters with solvers that implement the
    /// [`IsSingleClassSolver`](crate::solver::traits::IsSingleClassSolver) trait.
    pub trait SetOutlierRatio {
        /// Set the fraction of data that is to be classified as outliers (parameter `nu`).
        ///
        /// Default: `0.5`
        fn outlier_ratio(&mut self, nu: f64) -> &mut Self;
    }

    /// Implemented for parameters with solvers that implement the
    /// [`IsSupportVectorRegressionSolver`](crate::solver::traits::IsSupportVectorRegressionSolver) trait.
    pub trait SetRegressionLossSensitivity {
        /// Set the tolerance margin/loss sensitivity of support vector regression (parameter `p`).
        ///
        /// Default: `0.1`
        fn regression_loss_sensitivity(&mut self, p: f64) -> &mut Self;
    }
}

/// Represents the tunable parameters of a LIBLINEAR model.
///
/// This struct is generic on the [`Solver`](crate::solver::traits::Solver) trait and
/// its descendents, using them to implement solver-specific functionality.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "SolverT: serde::SupportsParametersCreation"))]
#[cfg_attr(feature = "serde", serde(try_from = "serde::GenericParameters"))]
#[cfg_attr(feature = "serde", serde(into = "serde::GenericParameters"))]
pub struct Parameters<SolverT> {
    pub(crate) _solver: PhantomData<SolverT>,
    pub(crate) epsilon: f64,
    pub(crate) cost: f64,
    pub(crate) p: f64,
    pub(crate) nu: f64,
    pub(crate) cost_penalty: Vec<(i32, f64)>,
    pub(crate) initial_solutions: Vec<f64>,
    pub(crate) bias: f64,
    pub(crate) regularize_bias: bool,
}

impl<SolverT> Parameters<SolverT>  {
    pub(crate) const EPSILON: f64 = 0.01;
    pub(crate) const COST: f64 = 1.0;
    pub(crate) const P: f64 = 0.1;
    pub(crate) const NU: f64 = 0.5;
    pub(crate) const BIAS: f64 = -1f64;
    pub(crate) const REGULARIZE_BIAS: bool = true;
}

impl<SolverT> Default for Parameters<SolverT> {
    fn default() -> Self {
        Self {
            _solver: PhantomData,
            epsilon: Self::EPSILON,
            cost: Self::COST,
            p: Self::P,
            nu: Self::NU,
            cost_penalty: Vec::new(),
            initial_solutions: Vec::new(),
            bias: Self::BIAS,
            regularize_bias: Self::REGULARIZE_BIAS,
        }
    }
}

impl<SolverT> Parameters<SolverT>
where
    SolverT: IsTrainableSolver,
{
    /// Set tolerance of termination criterion for optimization (parameter `e`).
    ///
    /// Default: `0.01`
    pub fn stopping_tolerance(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Set cost of constraints violation (parameter `C`).
    //
    /// Rules the trade-off between regularization and correct classification on data.
    /// It can be seen as the inverse of a regularization constant.
    ///
    /// Default: `1.0`
    pub fn constraints_violation_cost(&mut self, cost: f64) -> &mut Self {
        self.cost = cost;
        self
    }

    /// Set weights to adjust the cost of constraints violation for specific classes. Each element
    /// is a tuple where the first value is the label and the second its corresponding weight penalty.
    ///
    /// Useful when training classifiers on unbalanced input data or with asymmetric mis-classification cost.
    pub fn cost_penalty(&mut self, cost_penalty: Vec<(i32, f64)>) -> &mut Self {
        self.cost_penalty = cost_penalty;
        self
    }

    /// Set the bias of the training data. If `bias >= 0`, it's appended to the feature vector of each training data instance.
    ///
    /// Default: `-1.0`
    pub fn bias(&mut self, bias: f64) -> &mut Self {
        self.bias = bias;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), ModelError> {
        if self.epsilon <= 0f64 {
            return Err(ModelError::InvalidParameters(format!(
                "epsilon must be > 0, but got '{}'",
                self.epsilon
            )));
        }

        if self.cost <= 0f64 {
            return Err(ModelError::InvalidParameters(format!(
                "constraints violation cost must be > 0, but got '{}'",
                self.cost
            )));
        }

        if self.p < 0f64 {
            return Err(ModelError::InvalidParameters(format!(
                "regression loss sensitivity must be >= 0, but got '{}'",
                self.p
            )));
        }

        if self.bias >= 0f64 && <SolverT as Solver>::ordinal() == SolverOrdinal::ONECLASS_SVM {
            return Err(ModelError::InvalidParameters(format!(
                "bias term must be < 0 for single-class SVM, but got '{}'",
                self.bias
            )));
        }

        if !self.regularize_bias && self.bias != 1f64 {
            return Err(ModelError::InvalidParameters(format!(
                "bias term must be `1.0` when regularization is disabled, but got '{}'",
                self.bias
            )));
        }

        Ok(())
    }
}

impl<SolverT> SetInitialSolutions for Parameters<SolverT>
where
    SolverT: IsTrainableSolver + SupportsInitialSolutions,
{
    fn initial_solutions(&mut self, initial_solutions: Vec<f64>) -> &mut Self {
        self.initial_solutions = initial_solutions;
        self
    }
}

impl<SolverT> SetBiasRegularization for Parameters<SolverT>
where
    SolverT: IsTrainableSolver + CanDisableBiasRegularization,
{
    fn bias_regularization(&mut self, bias_regularization: bool) -> &mut Self {
        self.regularize_bias = bias_regularization;
        if !self.regularize_bias {
            self.bias = 1f64;
        }
        self
    }
}

impl<SolverT> SetOutlierRatio for Parameters<SolverT>
where
    SolverT: IsTrainableSolver + IsSingleClassSolver,
{
    fn outlier_ratio(&mut self, nu: f64) -> &mut Self {
        self.nu = nu;
        self
    }
}

impl<SolverT> SetRegressionLossSensitivity for Parameters<SolverT>
where
    SolverT: IsTrainableSolver + IsSupportVectorRegressionSolver,
{
    fn regression_loss_sensitivity(&mut self, p: f64) -> &mut Self {
        self.p = p;
        self
    }
}

#[cfg(feature = "serde")]
pub mod serde {
    use serde::{Deserialize, Serialize};
    use crate::errors::ModelError;
    use crate::parameter::traits::{SetBiasRegularization, SetInitialSolutions, SetOutlierRatio, SetRegressionLossSensitivity};
    use crate::Parameters;
    use crate::solver::traits::{CanDisableBiasRegularization, IsSingleClassSolver, IsSupportVectorRegressionSolver, IsTrainableSolver, Solver, SupportsInitialSolutions};

    /// Allows to create parameters from the serializable variant
    pub trait SupportsParametersCreation: IsTrainableSolver + Clone {

        /// Creates some parameters from the provided [generic_params] for the solver type.
        fn create_parameters(generic_params: &GenericParameters) -> Parameters<Self> where Self: Sized {
            let mut params = Parameters::default();
            Self::configure_parameters(generic_params, &mut params);
            params
        }

        /// Configures some kind of [params] with the provided [generic_params] for the solver type.
        fn configure_parameters(generic_params: &GenericParameters, params: &mut Parameters<Self>) where Self: Sized;
    }

    /// A generic view of [Parameters] for easier serialisation
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(default)]
    pub struct GenericParameters {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub epsilon: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cost: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub p: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub nu: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub cost_penalty: Option<Vec<(i32, f64)>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub initial_solutions: Option<Vec<f64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub bias: Option<f64>,
        #[serde(skip_serializing_if = "is_true")]
        pub regularize_bias: bool,
    }

    #[inline(always)] fn is_true(value: &bool) -> bool { *value }

    impl Default for GenericParameters {
        fn default() -> Self {
            Self {
                epsilon: None,
                cost: None,
                p: None,
                nu: None,
                cost_penalty: None,
                initial_solutions: None,
                bias: None,
                regularize_bias: true,
            }
        }
    }

    impl GenericParameters {
        /// Helps to set the trainable information to [params]
        pub fn set_trainable<SolverT: IsTrainableSolver>(&self, params: &mut Parameters<SolverT>) {
            if let Some(epsilon) = self.epsilon {
                params.stopping_tolerance(epsilon);
            }
            if let Some(cost) = self.cost {
                params.constraints_violation_cost(cost);
            }
            if let Some(cost_penalty) = &self.cost_penalty {
                params.cost_penalty(cost_penalty.clone());
            }
            if let Some(bias) = self.bias {
                params.bias(bias);
            }
        }

        /// Helps to set the initial solution to [params]
        pub fn set_initial_solution<SolverT: IsTrainableSolver + SupportsInitialSolutions>(&self, params: &mut Parameters<SolverT>) {
            if let Some(initial_solutions) = &self.initial_solutions {
                params.initial_solutions(initial_solutions.clone());
            }
        }

        /// Helps to set the bias regularisation to [params]
        pub fn set_bias_regularization<SolverT: IsTrainableSolver + CanDisableBiasRegularization>(&self, params: &mut Parameters<SolverT>) {
            params.bias_regularization(self.regularize_bias);
        }

        /// Helps to set the outlier ratio to [params]
        pub fn set_outlier_ratio<SolverT: IsTrainableSolver + IsSingleClassSolver>(&self, params: &mut Parameters<SolverT>) {
            if let Some(nu) = self.nu {
                params.outlier_ratio(nu);
            }
        }

        /// Helps to set the regression loss sensitivity to [params]
        pub fn set_regression_loss_sensitivity<SolverT: IsTrainableSolver + IsSupportVectorRegressionSolver>(&self, params: &mut Parameters<SolverT>) {
            if let Some(p) = self.p {
                params.regression_loss_sensitivity(p);
            }
        }
    }

    impl<SolverT> From<Parameters<SolverT>> for GenericParameters {
        fn from(value: Parameters<SolverT>) -> Self {
            Self {
                epsilon: (value.epsilon != Parameters::<SolverT>::EPSILON).then_some(value.epsilon),
                cost: (value.cost != Parameters::<SolverT>::COST).then_some(value.cost),
                p: (value.p != Parameters::<SolverT>::P).then_some(value.p),
                nu: (value.nu != Parameters::<SolverT>::NU).then_some(value.nu),
                cost_penalty: (!value.cost_penalty.is_empty()).then_some(value.cost_penalty),
                initial_solutions: (!value.initial_solutions.is_empty()).then_some(value.initial_solutions),
                bias: (value.bias != Parameters::<SolverT>::BIAS).then_some(value.bias),
                regularize_bias: value.regularize_bias,
            }
        }
    }

    impl<SolverT> TryFrom<GenericParameters> for Parameters<SolverT> where SolverT: SupportsParametersCreation {
        type Error = ModelError;

        fn try_from(value: GenericParameters) -> Result<Self, Self::Error> {
            let params = SolverT::create_parameters(&value);
            params.validate()?;
            Ok(params)
        }
    }
}





#[cfg(test)]
mod tests {
    use crate::{errors::ModelError, solver};
    use super::traits::*;
    use super::Parameters;

    #[test]
    fn test_parameter_runtime_validation() {
        let mut params = Parameters::<solver::L2R_LR>::default();
        params.stopping_tolerance(-1f64);
        assert!(matches!(
            params.validate(),
            Err(ModelError::InvalidParameters { .. })
        ));

        let mut params = Parameters::<solver::L1R_LR>::default();
        params.constraints_violation_cost(0f64);
        assert!(matches!(
            params.validate(),
            Err(ModelError::InvalidParameters { .. })
        ));

        let mut params = Parameters::<solver::L2R_L2LOSS_SVR>::default();
        params.regression_loss_sensitivity(-1f64);
        assert!(matches!(
            params.validate(),
            Err(ModelError::InvalidParameters { .. })
        ));

        let mut params = Parameters::<solver::ONECLASS_SVM>::default();
        params.bias(10f64).outlier_ratio(1f64);
        assert!(matches!(
            params.validate(),
            Err(ModelError::InvalidParameters { .. })
        ));

        let mut params = Parameters::<solver::L1R_L2LOSS_SVC>::default();
        params.bias_regularization(false).bias(10f64);
        assert!(matches!(
            params.validate(),
            Err(ModelError::InvalidParameters { .. })
        ));

        let mut params = Parameters::<solver::L2R_L2LOSS_SVR>::default();
        params
            .cost_penalty(Vec::new())
            .initial_solutions(Vec::new());

        let mut params = Parameters::<solver::L2R_L2LOSS_SVR>::default();
        params.cost_penalty(Vec::new());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialisation(){
        use crate::parameter::serde::SupportsParametersCreation;
        use crate::solver::traits::{IsTrainableSolver, Solver};

        fn test_params<SolverT: SupportsParametersCreation + Solver>(parameters: &Parameters<SolverT>) {
            let serialized = serde_json::to_string(parameters).expect(format!("Can serialize for {:?}", SolverT::ordinal()).as_str());
            let deserialized: Parameters<SolverT> = serde_json::from_str(&serialized).expect(format!("Can deserialize for {:?}", SolverT::ordinal()).as_str());
            float_cmp::assert_approx_eq!(f64, parameters.epsilon, deserialized.epsilon);
            float_cmp::assert_approx_eq!(f64, parameters.cost, deserialized.cost);
            float_cmp::assert_approx_eq!(f64, parameters.p, deserialized.p);
            float_cmp::assert_approx_eq!(f64, parameters.nu, deserialized.nu);
            assert_eq!(parameters.cost_penalty.len(), deserialized.cost_penalty.len());
            for ((a_i, a_f), (b_i, b_f)) in parameters.cost_penalty.iter().cloned().zip(deserialized.cost_penalty.iter().cloned()) {
                assert_eq!(a_i, b_i);
                float_cmp::assert_approx_eq!(f64, a_f, b_f);
            }
            assert_eq!(parameters.initial_solutions.len(), deserialized.initial_solutions.len());
            for (a, b) in parameters.initial_solutions.iter().cloned().zip(deserialized.initial_solutions.iter().cloned()) {
                float_cmp::assert_approx_eq!(f64, a, b);
            }
            float_cmp::assert_approx_eq!(f64, parameters.bias, deserialized.bias);
            assert_eq!(parameters.regularize_bias, deserialized.regularize_bias, "Regularize bias failed for {:?}", SolverT::ordinal());
        }

        fn configure_trainable<SolverT: IsTrainableSolver>(
            params: &mut Parameters<SolverT>,
            epsilon: f64,
            cost: f64,
            cost_penalty: Vec<(i32, f64)>,
            bias: f64
        ) {
            params
                .stopping_tolerance(epsilon)
                .constraints_violation_cost(cost)
                .cost_penalty(cost_penalty)
                .bias(bias);
        }

        let mut params: Parameters<solver::L2R_LR> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        params.initial_solutions(vec![0.5, 4.6]);
        test_params(&params);
        params.bias_regularization(false);
        test_params(&params);

        let mut params: Parameters<solver::L2R_L2LOSS_SVC_DUAL> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        test_params(&params);

        let mut params: Parameters<solver::L2R_L2LOSS_SVC> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        params.initial_solutions(vec![0.5, 4.6]);
        test_params(&params);
        params.bias_regularization(false);
        test_params(&params);

        let mut params: Parameters<solver::L2R_L1LOSS_SVC_DUAL> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        test_params(&params);

        let mut params: Parameters<solver::MCSVM_CS> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        test_params(&params);

        let mut params: Parameters<solver::L1R_L2LOSS_SVC> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        test_params(&params);
        params.bias_regularization(false);
        test_params(&params);

        let mut params: Parameters<solver::L1R_LR> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        test_params(&params);
        params.bias_regularization(false);
        test_params(&params);

        let mut params: Parameters<solver::L2R_LR_DUAL> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        test_params(&params);

        let mut params: Parameters<solver::L2R_L2LOSS_SVR> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        params.regression_loss_sensitivity(10.8);
        params.initial_solutions(vec![0.5, 4.6]);
        test_params(&params);
        params.bias_regularization(false);
        test_params(&params);

        let mut params: Parameters<solver::L2R_L2LOSS_SVR_DUAL> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        params.regression_loss_sensitivity(10.8);
        test_params(&params);

        let mut params: Parameters<solver::L2R_L1LOSS_SVR_DUAL> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        params.regression_loss_sensitivity(10.8);
        test_params(&params);

        let mut params: Parameters<solver::ONECLASS_SVM> = Parameters::default();
        configure_trainable(&mut params, 4.5, 5.5, vec![(1, 3.0), (2, 4.3)], -9.4);
        params.outlier_ratio(10.8);
        test_params(&params);
    }
}
