# %%
# ============================================
# STEP 1 — Import data
# ============================================

import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import warnings

RANDOM_STATE=202605

trans = pl.concat(
    [
        pl.read_parquet('s3://confpns/synthetic-transactions/rawdata/transactions/transactions_houses_final.parquet'),
        pl.read_parquet('s3://confpns/synthetic-transactions/rawdata/transactions/transactions_flats_final.parquet')
    ]
    ).to_pandas()

trans = trans[trans["prop_loc_dep"].isin(["75", "77", "78", "91", "92", "93", "94", "95"])]
trans["price_sqm"] = trans["price"] / trans["farea"]



# %%
# ============================================
# STEP 2 — Preprocessing
#   - Outlier removal
#   - Scaling
#   - One-hot encoding
# ============================================
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

trans = trans.dropna(subset = "price_sqm")
df = trans.drop(columns=[
    'trans_id', "trans_type_code", "trans_type_label", 
    "price", "prop_loc_dep", "prop_loc_citycode", "dist_tosea", "dist_tosea_corr"
])
df = df.dropna()
df["prop_type"] = pd.Categorical(
    df["prop_type"],
    categories=["1", "2"],
    ordered=False
).rename_categories({"1": "House", "2": "Flat"})

df['prop_year_harm_10'] = (df['prop_year_harm'] // 10)*10
df['prop_year_harm_10'] = df['prop_year_harm_10'].where(df['prop_year_harm_10'] >= 1850, 1840)

# Dropping old column
df = df.drop(columns=["prop_year_harm"])


def date_to_days(X: pd.Series, ref_date:pd.Timestamp):
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
        ("dat", date_transformer, "trans_date") # feature time since 01-01-2010
    ],
    remainder="passthrough"  # to keep features not transformed
) 


def log_transform(y):
    return np.log10(y)

def inverse_log_transform(y):
    return 10 ** y

y_transformer = FunctionTransformer(
    func = log_transform,
    inverse_func = inverse_log_transform)

# %%
# ============================================
# STEP 3 — Train / test split, model fitting,
#          and performance evaluation
# ============================================

# Split features / target
X = df.drop(columns="price_sqm")  # X must contain only the features we'll learn from
y = df["price_sqm"]  # target must be a dataframe with 1 column

# Split train / test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)


def rf_error_oob_plot(X_train,
                      y_train,
                      subsample=0.1,
                      min_estimators=15,
                      max_estimators=150,
                      metric='r2',
                      **rf_params):
    """
    Plot error OOB convergence by the number of trees

    Args:
        X_train: features
        y_train: target
        subsample: rate of sample for X_train
        min_estimators: number min of trees
        max_estimators: number max of trees
        metric : 'r2',  'rmse' or 'mae'
    """

# --- Stratified sampling of training set ---
    y_train_df = pd.DataFrame(y_train)
    y_train_df["quantile"] = pd.qcut(y_train_df["price_sqm"], q=100, labels=False) ## allows to discretly cut along quantiles
    y_sub = y_train_df.groupby("quantile").sample(frac=0.1, random_state= RANDOM_STATE)  # sampling by quantile 

    y_sub = y_sub["price_sqm"] # converting to pandas.series
    X_sub = X_train.filter(items=y_sub.index, axis=0).drop(columns=["trans_date", "prop_type"])    # sampling X_train


    rf = RandomForestRegressor(
        oob_score=True,
        warm_start=True,
        **rf_params,
    )

    oob_scores = []
    warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
    for n in range(min_estimators, max_estimators, 10):
        rf.set_params(n_estimators=n)
        rf.fit(X_sub, y_sub)
        if metric == "r2":
            oob_scores.append((n, 1 - rf.oob_score_))
        elif metric == "neg_root_mean_squared_error":
            mse = np.mean((y_sub - rf.oob_prediction_) ** 2)
            oob_scores.append((n, np.sqrt(mse)))
        else:
            mae = np.mean(np.abs(y_sub - rf.oob_prediction_))
            oob_scores.append((n, mae))
    warnings.resetwarnings()

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    xs, ys = zip(*oob_scores)

    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlim(min_estimators, max_estimators)
    ax.set_xlabel("n_trees")
    ax.set_ylabel(f"OOB error ({metric})")
    plt.close(fig)

    return fig


oob_error_ntrees = rf_error_oob_plot(X_train=X_train,
                                     y_train=y_train,
                                     subsample=0.1,
                                     min_estimators=5,
                                     max_estimators=150,
                                     metric="r2")
oob_error_ntrees

# %%
### Gridsearch

rf_params = {
    "max_depth": 8,
    "min_samples_split": 5,
    "random_state": RANDOM_STATE,
}

rf_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('RF', RandomForestRegressor(**rf_params))
])

model = TransformedTargetRegressor(
    regressor=rf_pipeline,
    transformer=y_transformer
)

param_grid = {
    "regressor__RF__n_estimators": [80],
    "regressor__RF__max_features": ["sqrt", "log2"],
    "regressor__RF__min_samples_leaf": [5, 10, 50]
}

# Grid search
grid_search = GridSearchCV(
    estimator=model, # it is the TransformedTargetRegressor created in the preprocessing part
    param_grid=param_grid,
    cv=4,  # number of folds
    scoring="r2", # 'r2' or 'neg_root_mean_squared_error' or 'neg_mean_absolute_error'
    n_jobs=-1,
    verbose=1
)

# Train
grid_search.fit(X_train, y_train)

rf_best = grid_search.best_estimator_
print(type(rf_best))

rf_best.fit(X_train, y_train)

# %%

y_pred = rf_best.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.4f}")

# %%
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(y_test, y_pred, alpha=0.3, s=5, label="Predictions")

lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

ax.set_xlabel("Actual values")
ax.set_ylabel("Predicted values")
ax.set_title("Predicted vs. Actual values on the test set")
ax.legend()
plt.tight_layout()
plt.show()