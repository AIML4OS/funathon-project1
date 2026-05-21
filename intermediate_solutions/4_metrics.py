# %%
import sys
sys.path.append("..") 

from joblib import load
import os
import s3fs
from solution.utils import set_seed
from solution.preprocess import set_date_transformer, set_preprocessor, set_y_transformer, complete_pre_processing

RANDOM_STATE = set_seed()
df = complete_pre_processing()

# Create filesystem object
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


date_transformer = set_date_transformer()

preprocessor = set_preprocessor()

y_transformer = set_y_transformer()


# Importing fine-tuned models
FILE_PATH_S3 = "projet-funathon/2026/project1/models/rf_model_final.joblib" 

with fs.open(FILE_PATH_S3, mode="rb") as model:
    rf_model_final = load(model)

FILE_PATH_S3 = "projet-funathon/2026/project1/models/gb_model_final.joblib" 

with fs.open(FILE_PATH_S3, mode="rb") as model:
    gb_model_final = load(model)

# %%
## Exercice 10: Compute evaluation metrics
# %%

# best RF
y_pred_RF = rf_model_final.predict(X_test)
rf_residuals = y_test - y_pred_RF

# best GB
y_pred_GB = gb_model_final.predict(X_test)
gb_residuals = y_test - y_pred_GB


# %%
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def print_metrics(model, split, X=X_train, y=y_train):
    """
    Print metrics for trained model
    """
    y_pred = model.predict(X)
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    print(f"{split} — RMSE: {rmse:.2f}  |  MAE: {mae:.2f}  |  R²: {r2:.4f}")

models = [("RF", rf_model_final), ("GB", gb_model_final)]

for name, model in models:
    print_metrics(model, name, X_test, y_test)


# %%
# Disponible à partir de scikit-learn >= 0.24
from sklearn.metrics import mean_absolute_percentage_error

mape_pct_rf = mean_absolute_percentage_error(y_test, y_pred_RF) * 100
mape_pct_gb = mean_absolute_percentage_error(y_test, y_pred_GB) * 100
print(f"MAPE RF: {mape_pct_rf:.2f} %")
print(f"MAPE GB: {mape_pct_gb:.2f} %")


## Exercice 11: Generate diagnostic plots
# %%
import matplotlib.pyplot as plt

def residuals_distribution(residuals: pd.Series, rmse: float, ax=None, label=None, color=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(residuals, bins=100, edgecolor="none", alpha=0.5, label=label or f"RMSE = {rmse:.3f}", color=color)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residuals distribution")
    ax.legend()
    return ax

# %%

fig, ax = plt.subplots()
residuals_distribution(rf_residuals, rmse_rf, ax=ax, label=f"RF (RMSE={rmse_rf:.3f})", color="steelblue")
residuals_distribution(gb_residuals, rmse_gb, ax=ax, label=f"GB (RMSE={rmse_gb:.3f})", color="darkorange")
plt.show()

# %%
import numpy as np

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
    return ax


# %%

fig, ax = plt.subplots()
QQplot(y_test, y_pred_RF, ax=ax, label="Random Forest", color="steelblue")
QQplot(y_test, y_pred_GB, ax=ax, label="Gradient Boosting", color="darkorange")
plt.show()

# %%

def target_distribution(y: pd.Series):
    y_sorted = np.sort(y)
    axe = np.linspace(0, 100, len(y_sorted))   # axe with percentiles

    fig = plt.figure()
    plt.plot(axe, y_sorted)
    plt.xlabel("Percentile")
    plt.ylabel("Value (EUR)")
    plt.title("Distribution")
    return fig


# %%

fig_actual = target_distribution(y_test)
plt.title("Target distribution — actual values")
plt.show()

fig_pred = target_distribution(y_pred_RF)
plt.title("Target distribution — predicted values with RF model")
plt.show()

fig_pred = target_distribution(y_pred_GB)
plt.title("Target distribution — predicted values with GB model")
plt.show()

# %%
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


fig, ax = plt.subplots()
plot_combined_distribution(y_test, y_pred_RF, ax=ax, label="Random Forest", color="steelblue", show_actual=True)
plot_combined_distribution(y_test, y_pred_GB, ax=ax, label="Gradient Boosting", color="darkorange", show_actual=False)
plt.show()

# %%

from sklearn.inspection import permutation_importance

def calculate_importance(X_test, y_test, RANDOM_STATE, final_rf, SCORING):
    X_test_sample = X_test.sample(n=min(100_000, len(X_test)), random_state=RANDOM_STATE)
    y_test_sample = y_test.loc[X_test_sample.index]

    perm = permutation_importance(
        final_rf, X_test_sample, y_test_sample,
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

# %%

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


# %%

score = "r2"

importances = calculate_importance(X_test, y_test, RANDOM_STATE, rf_model_final, score)
fig_importance = importance_plot(importances)
plt.show()

