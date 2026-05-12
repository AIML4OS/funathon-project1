# %%
from joblib import load
import os
import s3fs
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Create filesystem object
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})
RANDOM_STATE = 202605

# Importing pre-processed data
X_train = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/X_train.parquet')
X_test  = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/X_test.parquet')
y_train = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/y_train.parquet').values
y_test  = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/y_test.parquet').values
df      = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/df.parquet')


# Pipeline
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
        ("dat", date_transformer, "trans_date") # feature time since 01-01-2010
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


# Importing fine-tuned models
FILE_PATH_S3 = "projet-funathon/2026/project1/models/rf_model_final.joblib" 

with fs.open(FILE_PATH_S3, mode="rb") as model:
    rf_model_final = load(model)

FILE_PATH_S3 = "projet-funathon/2026/project1/models/gb_model_final.joblib" 

with fs.open(FILE_PATH_S3, mode="rb") as model:
    gb_model_final = load(model)

# %%