import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from metrics import predicted_actual_plot, residuals_distribution, plot_combined_distribution, QQplot, importance_plot, calculate_importance
from utils import set_seed
# MLflow connection
mlflow_server = os.getenv("MLFLOW_TRACKING_URI")  # your environment feature for accessing to MLFlow server
mlflow.set_tracking_uri(mlflow_server)


# MLFlow logging
def log_to_mlflow(exp_name, model, model_name, model_params, X_train, X_test, y_train, y_test, logger):
    RANDOM_STATE = set_seed()

    mlflow.set_experiment(exp_name)
    signature = infer_signature(X_train, model.predict(X_train))

    with mlflow.start_run():
        logger.info(f"Loggine model {model_name} to MLFlow")
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=model_name,
            code_paths=["solution/preprocess.py"]  # To store the full pipeline with it
        )
        logger.info("Computing residuals and metrics")
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        model_metrics = {
            "neg_root_mean_squared_error": root_mean_squared_error(y_test, y_pred),
            "neg_mean_absolute_error": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        logger.info("Logging metrics and params")
        mlflow.log_metrics(model_metrics)
        mlflow.log_params(model_params)

        logger.info("Logging Predicted vs actual values")
        # Predicted vs actual values
        mlflow.log_figure(
                        predicted_actual_plot(y_test, y_pred, model_name),
                        "predicted_actual.png"
                    )

        logger.info("Logging distribution of residuals")
        # Distribution of residuals
        mlflow.log_figure(
                residuals_distribution(residuals, model_metrics["r2"]),
                "residuals_distrib.png"
            )

        logger.info("Logging distribution of y_test vs y_pred")
        # Distribution of y_test and y_pred
        fig, ax = plt.subplots()
        plot_combined_distribution(y_test, y_pred, ax=ax, label=f"{model_name} - predicted values", color="steelblue", show_actual=True)
        mlflow.log_figure(fig, "y_distrib.png")

        logger.info("Logging QQ plot")
        # QQ Plot
        mlflow.log_figure(QQplot(y_test, y_pred), "qqplot.png")

        logger.info("Logging importance plot")
        # Importance plot
        mlflow.log_figure(
            importance_plot(
                calculate_importance(X_test, y_test, RANDOM_STATE, model, "r2")
            ),
            "importance.png"
        )
