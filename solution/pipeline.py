from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


def date_to_days(X: pd.Series, ref_date: pd.Timestamp):
    # converts a date to a difference to ref_date :
    diff_dt = pd.to_datetime(X) - ref_date
    # Extract days part from datetime object
    diff_dt = diff_dt.dt.days
    # Transform it from a Pandas series to a Numpy nd array, used by scikit learn for input
    diff_dt = diff_dt.to_numpy().reshape(-1, 1)

    return diff_dt


def set_date_transformer():
    return FunctionTransformer(
        date_to_days,
        kw_args={"ref_date": pd.Timestamp('2010-01-01 00:00')}
    )


def set_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["prop_type", "prop_year_harm_10"]),  # one-hot encoder on feature
            ("dat", set_date_transformer(), "trans_date")  # feature time since 01-01-2010
        ],
        remainder="passthrough"  # to keep features not transformed
    )


def log_transform(y):
    return np.log10(y)


def inverse_log_transform(y):
    return 10 ** y


def set_y_transformer():
    return FunctionTransformer(
        func=log_transform,
        inverse_func=inverse_log_transform
    )


def set_pipeline(ml_name, ml_model):
    ml_pipeline = Pipeline([
        ("preprocessor", set_preprocessor()),
        (ml_name, ml_model),
    ])

    ml_model_pipeline = TransformedTargetRegressor(
        regressor=ml_pipeline,
        transformer=set_y_transformer()
    )

    return ml_model_pipeline
