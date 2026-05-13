# %%
import duckdb
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import s3fs
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import logging
import datetime


# Setting the S3 connection and path generators
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

# MLflow connection
mlflow_server = os.getenv("MLFLOW_TRACKING_URI")  # your environment feature for accessing to MLFlow server
mlflow.set_tracking_uri(mlflow_server)

# Seed
RANDOM_STATE = 202605


# Logging
def setup_logging():
    """Configure logging with both console and file handlers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'veille_ssphub_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# %%
# Metrics
def QQplot(y_test: pd.Series, y_pred: pd.Series, ax=None, label=None, color=None):
    """
    Actual quantiles vs predicted quantiles
    """
    quantiles = np.linspace(0, 100, 1000)
    q_real = np.percentile(y_test, quantiles)
    q_predict = np.percentile(y_pred, quantiles)

    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(q_real, q_predict, alpha=0.5, s=5, label=label or "Quantiles", color=color)
    ax.plot(
        [q_real[0], q_real[-1]],
        [q_real[0], q_real[-1]],
        "r--", linewidth=1.5
    )
    ax.set_xlabel("Actual quantiles")
    ax.set_ylabel("Predicted quantiles")
    ax.set_title("QQ-plot: actual vs predicted")
    ax.legend()
    return ax.get_figure()


def residuals_distribution(residuals: pd.Series, rmse: float, ax=None, label=None, color=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(residuals, bins=100, edgecolor="none", alpha=0.5, label=label or f"RMSE = {rmse:.3f}", color=color)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residuals distribution")
    ax.legend()
    return ax.get_figure()


def predicted_actual_plot(y_test, y_pred_test, model_name):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_test, y_pred_test, alpha=0.3, s=5, label="Predictions")

    lims = [min(y_test.min(), y_pred_test.min()),
            max(y_test.max(), y_pred_test.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual values (log)")
    ax.set_ylabel("Predicted values (log)")
    ax.set_title(f"Comparison of predicted values vs. actual values on the test set\n({model_name})")
    ax.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    return fig


def plot_combined_distribution(y_test: pd.Series, y_pred: pd.Series, ax=None, label=None, color=None, show_actual=True):
    """
    Plots the target distributions of actual and predicted values on the same graph.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if show_actual:
        y_sorted_actual = np.sort(y_test)
        axe_actual = np.linspace(0, 100, len(y_sorted_actual))
        ax.plot(axe_actual, y_sorted_actual, label="Actual Values", color="black")

    y_sorted_pred = np.sort(y_pred)
    axe_pred = np.linspace(0, 100, len(y_sorted_pred))
    ax.plot(axe_pred, y_sorted_pred, label=label or "Predicted Values", color=color)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Price")
    ax.set_title("Target distribution — actual vs predicted values")
    ax.legend()
    return ax


def calculate_importance(X_test, y_test, RANDOM_STATE, model, SCORING):
    X_test_sample = X_test.sample(n=min(100000, len(X_test)), random_state=RANDOM_STATE)
    y_test_sample = y_test.loc[X_test_sample.index]

    perm = permutation_importance(
        model, X_test_sample, y_test_sample,
        n_repeats=5,
        scoring=SCORING,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    importances = (
        pd.Series(perm.importances_mean, index=X_test.columns)
        .sort_values(ascending=False)
    )
    return importances


def importance_plot(importances):
    """
    Permutation importance plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.head(20).plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_title("Permutation importance (top 20)")
    ax.set_xlabel("Mean increase in RMSE")
    plt.tight_layout()
    return fig


# MLFlow logging
def log_to_mlflow(exp_name, model, model_name, model_params, X_train, X_test, y_train, y_test):
    mlflow.set_experiment(exp_name)
    signature = infer_signature(X_train, model.predict(X_train))

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=model_name,
        )

        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        model_metrics = {
            "neg_root_mean_squared_error": root_mean_squared_error(y_test, y_pred),
            "neg_mean_absolute_error": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

        mlflow.log_metrics(model_metrics)
        mlflow.log_params(model_params)

        # Predicted vs actual values
        mlflow.log_figure(
                        predicted_actual_plot(y_test, y_pred, model_name),
                        "predicted_actual.png"
                    )

        # Distribution of residuals
        mlflow.log_figure(
                residuals_distribution(residuals, model_metrics["r2"]),
                "residuals_distrib.png"
            )

        # Distribution of y_test and y_pred
        fig, ax = plt.subplots()
        plot_combined_distribution(y_test, y_pred, ax=ax, label=f"{model_name} - predicted values", color="steelblue", show_actual=True)
        mlflow.log_figure(fig, "y_distrib.png")

        # QQ Plot
        mlflow.log_figure(QQplot(y_test, y_pred), "qqplot.png")

        # Importance plot
        mlflow.log_figure(
            importance_plot(
                calculate_importance(X_test, y_test, RANDOM_STATE, model, "r2")
            ),
            "importance.png"
        )


# %%
logger.info("Importing data")
# Create a non-persistent connection (the database exists only while the connection is alive and disappears when it is closed)
con = duckdb.connect(database=":memory:")

# You need to create a secret table with all the S3 credentials
con.execute(
    f"""
    CREATE SECRET secret_s3 (
    TYPE S3,
    KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
    SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
    ENDPOINT '{os.environ["AWS_S3_ENDPOINT"]}',
    SESSION_TOKEN '{os.environ["AWS_SESSION_TOKEN"]}',
    REGION 'eu-west-1',
    URL_STYLE 'path',
    SCOPE 's3://projet-funathon/'
    );
    """
)


# We load all transactions made in France between 2010 and 2022
trans = con.sql(
    """
        SELECT * FROM read_parquet('s3://projet-funathon/2026/project1/data/1_input/transactions_EN.parquet')
    """).to_df()


trans = trans[trans["prop_loc_dep"].isin(["75", "77", "78", "91", "92", "93", "94", "95"])]
# %%
logger.info("Pre-processing")
# Exercice 2: Analyzing data inputs
trans["price_sqm"] = trans["price"] / trans["farea"]

# Apply some deterministic threshold on the dataframe
trans = trans[(trans["price_sqm"] < 200000) & (trans["price_sqm"] > 100)]


# Apply IQR methods for the outlier removal
def outlier_transform(y, lower=0.1, upper=0.9):
    """
    Transform Y target to log(Y) and remove outliers with IQR method

    Args :
        y : target
        lower: lower quantile for the IQR
        upper: upper quantile for the IQR
    """
    Q_lower = np.quantile(y, lower)
    Q_upper = np.quantile(y, upper)
    IQR = Q_upper - Q_lower

    mask = (y >= Q_lower - 1.5 * IQR) & (y <= Q_upper + 1.5 * IQR)
    return mask


mask = outlier_transform(trans["price_sqm"])
trans = trans[mask].reset_index(drop=True)

trans = trans.dropna(subset="price_sqm")

df = trans.drop(columns=[
    "price", "prop_loc_dep", "prop_loc_citycode", "dist_tosea"
])


# Filtering NA values
df = df.dropna()

df["prop_type"] = pd.Categorical(
    df["prop_type"],
    categories=["1", "2"],
    ordered=False
).rename_categories({"1": "House", "2": "Flat"})


# Replacing year of construction by decade and merging together all years before 1850
df['prop_year_harm_10'] = (df['prop_year_harm'] // 10)*10
df['prop_year_harm_10'] = df['prop_year_harm_10'].where(df['prop_year_harm_10'] >= 1850, 1840)

# Dropping old column
df = df.drop(columns=["prop_year_harm"])
# %%
logger.info("Pipeline")


def date_to_days(X: pd.Series, ref_date: pd.Timestamp):
    # converts a date to a difference to ref_date :
    diff_dt = pd.to_datetime(X) - ref_date
    # Extract days part from datetime object
    diff_dt = diff_dt.dt.days
    # Transform it from a Pandas series to a Numpy nd array, used by scikit learn for input
    diff_dt = diff_dt.to_numpy().reshape(-1, 1)

    return diff_dt


date_transformer = FunctionTransformer(
    date_to_days,
    kw_args={"ref_date": pd.Timestamp('2010-01-01 00:00')}
    )

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["prop_type", "prop_year_harm_10"]),  # one-hot encoder on feature
        ("dat", date_transformer, "trans_date")  # feature time since 01-01-2010
    ],
    remainder="passthrough"  # to keep features not transformed
)


def log_transform(y):
    return np.log10(y)


def inverse_log_transform(y):
    return 10 ** y


y_transformer = FunctionTransformer(
    func=log_transform,
    inverse_func=inverse_log_transform)

logger.info("Setting training data sets")
X = df.drop(columns=["price_sqm"])
y = df["price_sqm"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)


# %%
logger.info("Fitting GB model")
BEST_ITER = 1000
BEST_LR = 0.3
BEST_DEPTH = 20
BEST_MIN_LEAF = 50
BEST_L2 = 0

gb_params = {
    "max_iter": BEST_ITER,
    "learning_rate": BEST_LR,
    "max_depth": BEST_DEPTH,
    "min_samples_leaf": BEST_MIN_LEAF,
    "l2_regularization": BEST_L2,
    "random_state": RANDOM_STATE
}

gb_final = HistGradientBoostingRegressor(
    **gb_params
)

# Wrap in the same pipeline / TransformedTargetRegressor as the RF section
gb_pipeline_best = Pipeline([
    ("preprocessor", preprocessor),  # same preprocessor as defined in the preprocessing section
    ("GB", gb_final),
])

gb_model_final = TransformedTargetRegressor(
    regressor=gb_pipeline_best,
    transformer=y_transformer  # same targettransformer as defined in preprocessing section
)

gb_model_final.fit(X_train, y_train)

# %%
# Saving model to MLFlow
logger.info("Storing GB model to MLFlow")
exp_name = "Funathon - project 1"

log_to_mlflow(
    exp_name=exp_name,
    model=gb_model_final,
    model_name="GB",
    model_params=gb_params,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)
# %%
logger.info("Setting training data sets")
X = df.drop(columns=["price_sqm"])
y = df["price_sqm"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)
# %%
logger.info("Fitting RF model")
# create RandomForestRegressor with tuned hyperparameters
rf_params = {
    "n_estimators": 80,
    "max_features": "sqrt",
    "min_samples_leaf": 40
}

rf_final = RandomForestRegressor(
    **rf_params
)

rf_pipeline_best = Pipeline([
    ("preprocessor", preprocessor),  # same preprocessor as defined in the preprocessing section
    ("RF", rf_final),
])

rf_model_final = TransformedTargetRegressor(
    regressor=rf_pipeline_best,
    transformer=y_transformer  # same targettransformer as defined in preprocessing section
)

# Train the model
rf_model_final.fit(X_train, y_train)

# %%
# Saving model to MLFlow
logger.info("Storing RF model to MLFlow")

log_to_mlflow(
    exp_name=exp_name,
    model=rf_model_final,
    model_name="RF",
    model_params=rf_params,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)

# %%
