# %%
import mlflow.sklearn
import os
import s3fs
from joblib import dump


# ── Setting the S3 connection and path generators ──────────
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


def generate_file_path_s3(FILE_KEY_OUT_S3: str):
    BUCKET_OUT = "projet-funathon"
    return BUCKET_OUT + "/2026/project1/" + FILE_KEY_OUT_S3


def generate_file_path_s3_models(FILE_KEY_OUT_S3: str):
    return generate_file_path_s3(f"models/{FILE_KEY_OUT_S3}")


def generate_file_path_s3_data(FILE_KEY_OUT_S3: str):
    return generate_file_path_s3(f"data/{FILE_KEY_OUT_S3}")


# ── Storing datasets ────────────────────────────────────
# datasets = {"df": df}
# for name, data in datasets.items():
#     with fs.open(generate_file_path_s3_data(f"2_preprocessing/{name}.parquet"), 'wb') as file_out:
#         data.to_parquet(file_out, index=True)

# ── Load GB from MLFlow ────────────────────────────────────
MODEL_URI = "models:/GB@latest"
gb_model_final = mlflow.sklearn.load_model(MODEL_URI)
# ── Storing GB model ────────────────────────────────────
with fs.open(generate_file_path_s3_models("gb_model_final.joblib"), 'wb') as file_out:
    dump(gb_model_final, file_out)

# ── Load RF from MLFlow ────────────────────────────────────
MODEL_URI = "models:/RF@latest"
rf_model_final = mlflow.sklearn.load_model(MODEL_URI)
# ── Storing RF model ────────────────────────────────────
with fs.open(generate_file_path_s3_models("rf_model_final.joblib"), 'wb') as file_out:
    dump(rf_model_final, file_out)


# %%
