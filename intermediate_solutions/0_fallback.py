# Run this script to load the dataframe df and final tuned GB and RF models. 
#  %%
import sys
sys.path.append("..")
sys.path.append("../solution")
import pandas as pd
from joblib import load
from solution.utils import set_seed, set_s3fs, generate_file_path_s3_models

RANDOM_STATE = set_seed()

df      = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/df_log.parquet")
X_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_train_log.parquet")
X_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_test_log.parquet")
y_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_train_log.parquet")["price_sqm_log"]
y_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_test_log.parquet")["price_sqm_log"]

# Create filesystem object
fs = set_s3fs()


# Importing fine-tuned RF and GB models
FILE_PATH_S3 = generate_file_path_s3_models("rf_model_log_final.joblib")

with fs.open(FILE_PATH_S3, mode="rb") as model:
    rf_model_final = load(model)

FILE_PATH_S3 = generate_file_path_s3_models("gb_model_log_final.joblib") 

with fs.open(FILE_PATH_S3, mode="rb") as model:
    gb_model_final = load(model)

# %%
