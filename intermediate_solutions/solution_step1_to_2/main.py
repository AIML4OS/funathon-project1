# %%
# ============================================
# STEP 1 — Generate synthetic regression data
# ============================================

import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

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


rf_params = {
    "n_estimators": 100,
    "max_depth": 5,
    "max_features": "sqrt",
    "min_samples_split": 2,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
    "oob_score": True,
    "n_jobs": -1,  # The number of jobs to run in parallel, -1 using all processors
}

rf_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('RF', RandomForestRegressor(**rf_params))
])

model = TransformedTargetRegressor(
    regressor=rf_pipeline,
    transformer=y_transformer
)
