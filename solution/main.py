# %%
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from preprocessing import load_data, outlier_transform, pre_process_raw_data
from logging import log_to_mlflow
from pipeline import set_pipeline
import s3fs
from utils import setup_logging, set_seed


# Setting the S3 connection and path generators
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


logger = setup_logging()

trans = load_data()

trans = trans[trans["prop_loc_dep"].isin(["75", "77", "78", "91", "92", "93", "94", "95"])]

trans["price_sqm"] = trans["price"] / trans["farea"]

# Apply some deterministic threshold on the dataframe
trans = trans[(trans["price_sqm"] < 200000) & (trans["price_sqm"] > 100)]


# Apply IQR methods for the outlier removal
mask = outlier_transform(trans["price_sqm"])
trans = trans[mask].reset_index(drop=True)

trans = trans.dropna(subset="price_sqm")
df = trans.drop(columns=[
    "price", "prop_loc_dep", "prop_loc_citycode", "dist_tosea"
])

df = pre_process_raw_data(df)
# %%
# %%
logger.info("Pipeline")

logger.info("Setting training data sets")
X = df.drop(columns=["price_sqm"])
y = df["price_sqm"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=set_seed()
)


# %%


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
    "random_state": set_seed()
}

gb_model_final = set_pipeline(
    "GB",
    HistGradientBoostingRegressor(
        **gb_params
    )
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
    random_state=set_seed()
)
# %%
rf_params = {
        "n_estimators": 80,
        "max_features": "sqrt",
        "min_samples_leaf": 40
    }

rf_model_final = set_pipeline(
    "RF",
    RandomForestRegressor(
        **rf_params
    )
)
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
