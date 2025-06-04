import logging
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.pipeline import Pipeline
import mlflow
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy
from zenml import step


@step(enable_cache=False)
def model_evaluator_step(
    trained_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Dict[str, Any], float]:
    """
    Evaluates the trained model using enhanced ModelEvaluator and RegressionModelEvaluationStrategy.

    Parameters:
    trained_model (Pipeline): The trained pipeline containing the model and preprocessing steps.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    Tuple[Dict[str, Any], float]: A tuple containing:
        - Dictionary with comprehensive evaluation metrics and diagnostic plots
        - RMSE value for model comparison
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    logging.info("Applying preprocessing to the test data.")
    X_test_processed = trained_model.named_steps["preprocessor"].transform(X_test)

    # Initialize the evaluator with the regression strategy
    evaluator = ModelEvaluator(strategy=RegressionModelEvaluationStrategy())

    # Perform comprehensive evaluation
    evaluation_results = evaluator.evaluate(
        trained_model.named_steps["model"], X_test_processed, y_test
    )

    # Log metrics to MLflow
    for metric_name, metric_value in evaluation_results.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(metric_name.lower().replace(" ", "_"), metric_value)
        elif isinstance(metric_value, dict) and "Mean" in metric_value:
            mlflow.log_metric(f"{metric_name.lower().replace(' ', '_')}_mean", metric_value["Mean"])

    # Log diagnostic plots
    if "Diagnostic Plots" in evaluation_results:
        for plot_name, fig in evaluation_results["Diagnostic Plots"].items():
            mlflow.log_figure(fig, f"{plot_name}.png")

    # Return the evaluation metrics and RMSE for model comparison
    rmse = evaluation_results.get("Root Mean Squared Error", None)
    return evaluation_results, rmse
