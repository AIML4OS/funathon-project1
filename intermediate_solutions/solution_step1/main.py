# ============================================
# STEP 1 — Data exploration
# ============================================
# %%
## ============================================
# Exercice 1 -Import data and calculate price per sqm for Paris region
# ## ============================================

import os

import duckdb
import numpy as np
import pandas as pd

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
RANDOM_STATE = 202605

# We load all transactions made in France between 2010 and 2022
trans = con.sql(
    """
        SELECT * FROM read_parquet('s3://projet-funathon/2026/project1/data/transactions_EN.parquet')
    """
).to_df()

# Filtering Paris region
trans = trans[
    trans["prop_loc_dep"].isin(["75", "77", "78", "91", "92", "93", "94", "95"])
]
trans["price_sqm"] = trans["price"] / trans["farea"]

# %%
## ============================================
# Exercice 2 - Data exploration
# ## ============================================
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

# --- features filtering
df = trans.drop(
    columns=[
        "price",
        "prop_loc_dep",
        "prop_loc_citycode",
        "dist_tosea",
        "predicted_price",
    ]
)

# Filtering NA values
df = df.dropna()

# Categorical feature
df["prop_type"] = pd.Categorical(
    df["prop_type"], categories=["1", "2"], ordered=False
).rename_categories({"1": "House", "2": "Flat"})

# Simplifying year of the property
# Replacing year of construction by decade and merging together all years before 1850
df["prop_year_harm_10"] = (df["prop_year_harm"] // 10) * 10
df["prop_year_harm_10"] = df["prop_year_harm_10"].where(
    df["prop_year_harm_10"] >= 1850, 1840
)

# Dropping old column
# df = df.drop(columns=["prop_year_harm"])

## ============================================
# Exercice 3 - A first pipeline
# ## ============================================
# %%

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# Split features / target
X = df.drop(columns="price_sqm")  # X must contain only the features we'll learn from
y = df["price_sqm"]  # target must be a dataframe with 1 column

# Split train / test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)


def date_to_days(X: pd.Series, ref_date: pd.Timestamp):
    # converts a date to a difference to ref_date :
    diff_dt = pd.to_datetime(X) - ref_date
    # Extract days part from datetime object
    diff_dt = diff_dt.dt.days
    # Transform it from a Pandas series to a Numpy nd array, used by scikit learn for input
    diff_dt = diff_dt.to_numpy().reshape(-1, 1)

    return diff_dt


date_transformer = FunctionTransformer(
    date_to_days, kw_args={"ref_date": pd.Timestamp("2010-01-01 00:00")}
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            ["prop_type", "prop_year_harm_10"],
        ),  # one-hot encoder on feature
        ("dat", date_transformer, "trans_date"),  # feature time since 01-01-2010
    ],
    remainder="passthrough",  # to keep features not transformed
)


def log_transform(y):
    return np.log10(y)


def inverse_log_transform(y):
    return 10**y


y_transformer = FunctionTransformer(
    func=log_transform, inverse_func=inverse_log_transform
)

# Other option with Numpy :
# y_transformer = FunctionTransformer(
#     func=np.log,
#     inverse_func=np.exp)

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

rf_pipeline = Pipeline(
    [("preprocessing", preprocessor), ("RF", RandomForestRegressor(**rf_params))]
)

model = TransformedTargetRegressor(regressor=rf_pipeline, transformer=y_transformer)
