import logging
import datetime
import pandas as pd
import mlflow.sklearn
import os
import s3fs
from joblib import dump


def setup_logging():
    """Configure logging with both console and file handlers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'funathon_aiml4os_project1_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ]
    )
    return logging.getLogger(__name__)


def set_seed():
    return 202605


# %%
def check_data(df):

    res_dict = {
        "hash": int(pd.util.hash_pandas_object(df).sum()),
        "n_cols": df.shape[1],
        "n_rows": df.shape[0]
    }

    msg = f"""\
====================================================
hash              : {res_dict["hash"]}
number of columns : {res_dict["n_cols"]}
number of rows    : {res_dict["n_rows"]}

"""

    res_dict["msg"] = msg

    return res_dict


# %%
def set_s3fs():
    # ── Setting the S3 connection and path generators ──────────
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    return fs


def generate_file_path_s3(FILE_KEY_OUT_S3: str):
    BUCKET_OUT = "projet-funathon"
    return BUCKET_OUT + "/2026/project1/" + FILE_KEY_OUT_S3


def generate_file_path_s3_models(FILE_KEY_OUT_S3: str):
    return generate_file_path_s3(f"models/{FILE_KEY_OUT_S3}")


def generate_file_path_s3_data(FILE_KEY_OUT_S3: str):
    return generate_file_path_s3(f"data/{FILE_KEY_OUT_S3}")


def store_datasets(datasets_to_store: dict):
    # ── Storing datasets ────────────────────────────────────
    fs = set_s3fs()
    for name, data in datasets_to_store.items():
        with fs.open(generate_file_path_s3_data(f"2_preprocessing/{name}.parquet"), 'wb') as file_out:
            data.to_parquet(file_out, index=True)


def store_model_mlflow_s3(model_uri: str, s3_name: str):
    # ── Load GB from MLFlow ────────────────────────────────────
    fs = set_s3fs()
    model = mlflow.sklearn.load_model(model_uri)
    # ── Storing GB model to S3 ─────────────────────────────────
    with fs.open(generate_file_path_s3_models(s3_name), 'wb') as file_out:
        dump(model, file_out)

