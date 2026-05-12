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
from joblib import dump
import s3fs

# Setting the S3 connection and path generators
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


def generate_file_path_s3(FILE_KEY_OUT_S3: str):
    BUCKET_OUT = "projet-funathon"
    return BUCKET_OUT + "/2026/project1/" + FILE_KEY_OUT_S3


def generate_file_path_s3_models(FILE_KEY_OUT_S3: str):
    return generate_file_path_s3(f"models/{FILE_KEY_OUT_S3}")


def generate_file_path_s3_data(FILE_KEY_OUT_S3: str):
    return generate_file_path_s3(f"data/{FILE_KEY_OUT_S3}")


print("Importing data")
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
        SELECT * FROM read_parquet('s3://projet-funathon/2026/project1/data/1_input/transactions_EN.parquet')
    """).to_df()


trans = trans[trans["prop_loc_dep"].isin(["75", "77", "78", "91", "92", "93", "94", "95"])]
# %%
print("Pre-processing")
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

trans = trans.dropna(subset = "price_sqm")

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
print("Pipeline")
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
    func=log_transform,
    inverse_func=inverse_log_transform)

print("Setting training data sets and storing it")
X = df.drop(columns=["price_sqm"])
y = df["price_sqm"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# Storing datasets
datasets = {"X_train": X_train, "X_test": X_test, "y_train": y_train.to_frame(), "y_test":y_test.to_frame()}

for name, data in datasets.items():
    with fs.open(generate_file_path_s3_data(f"2_preprocessing/{name}.parquet"), 'wb') as file_out:
        data.to_parquet(file_out, index=True)

# %%
print("Fitting GB model")
BEST_ITER = 1000
BEST_LR = 0.3
BEST_DEPTH = 20
BEST_MIN_LEAF = 50
BEST_L2 = 0

gb_final = HistGradientBoostingRegressor(
    max_iter=BEST_ITER,
    learning_rate=BEST_LR,
    max_depth=BEST_DEPTH,
    min_samples_leaf=BEST_MIN_LEAF,
    l2_regularization=BEST_L2,
    random_state=RANDOM_STATE,
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

print("Saving fitted GB model")
# Save the model to a file
with fs.open(generate_file_path_s3_models("gb_model_final.joblib"), 'wb') as file_out:
    dump(gb_model_final, file_out)


print("Fitting RF model")
# create RandomForestRegressor with tuned hyperparameters
rf_final = RandomForestRegressor(
    n_estimators=80,
    max_features="sqrt",
    min_samples_leaf=50
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

print("Saving fitted RF model")
# Save the model to a file
with fs.open(generate_file_path_s3_models("rf_model_final.joblib"), 'wb') as file_out:
    dump(rf_model_final, file_out)

