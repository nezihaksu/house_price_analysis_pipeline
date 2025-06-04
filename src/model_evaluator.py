import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Abstract method to evaluate a model.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics and plots.
        """
        pass


class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def calculate_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate VIF for each feature."""
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])]
        return dict(zip(vif_data["Feature"], vif_data["VIF"]))

    def create_diagnostic_plots(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, plt.Figure]:
        """Create diagnostic plots for regression analysis."""
        plots = {}
        
        # Residual Plot
        residuals = y_true - y_pred
        fig_residual, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        plots['residual_plot'] = fig_residual

        # Actual vs Predicted Plot
        fig_actual_pred, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_true, y_pred)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted')
        plots['actual_vs_predicted'] = fig_actual_pred

        # QQ Plot
        fig_qq, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Normal Q-Q Plot')
        plots['qq_plot'] = fig_qq

        return plots

    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Enhanced evaluation of regression model with comprehensive metrics and diagnostics.

        Parameters:
        model (RegressorMixin): The trained regression model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing comprehensive evaluation metrics and plots.
        """
        logging.info("Starting comprehensive model evaluation.")
        
        # Basic predictions
        y_pred = model.predict(X_test)
        
        # Calculate basic metrics
        metrics = {
            "Mean Squared Error": mean_squared_error(y_test, y_pred),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
            "Mean Absolute Percentage Error": mean_absolute_percentage_error(y_test, y_pred),
            "R-Squared": r2_score(y_test, y_pred),
        }
        
        # Calculate adjusted R-squared
        n = len(y_test)
        p = X_test.shape[1]
        metrics["Adjusted R-Squared"] = 1 - (1 - metrics["R-Squared"]) * (n - 1) / (n - p - 1)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        metrics["Cross Validation Scores"] = {
            "Mean": cv_scores.mean(),
            "Std": cv_scores.std(),
            "Scores": cv_scores.tolist()
        }
        
        # Calculate VIF for multicollinearity
        metrics["VIF"] = self.calculate_vif(X_test)
        
        # Create diagnostic plots
        metrics["Diagnostic Plots"] = self.create_diagnostic_plots(y_test, y_pred)
        
        # Statistical tests
        residuals = y_test - y_pred
        
        # Breusch-Pagan test for heteroscedasticity
        bp_test = sms.het_breuschpagan(residuals, X_test)
        metrics["Breusch-Pagan Test"] = {
            "statistic": bp_test[0],
            "p_value": bp_test[1]
        }
        
        logging.info("Completed comprehensive model evaluation.")
        return metrics


class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
        strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
        strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Executes the comprehensive model evaluation using the current strategy.

        Parameters:
        model (RegressorMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing comprehensive evaluation metrics and plots.
        """
        logging.info("Starting model evaluation with the selected strategy.")
        return self._strategy.evaluate_model(model, X_test, y_test)


# Example usage
if __name__ == "__main__":
    # Example trained model and data (replace with actual trained model and data)
    # model = trained_sklearn_model
    # X_test = test_data_features
    # y_test = test_data_target

    # Initialize model evaluator with a specific strategy
    # model_evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    # evaluation_metrics = model_evaluator.evaluate(model, X_test, y_test)
    # print(evaluation_metrics)

    pass
