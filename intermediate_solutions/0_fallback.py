# Run this script to load the dataframe df and final tuned GB and RF models. 
#  %%
import sys
sys.path.append("..")
sys.path.append("../solution")
import pandas as pd
from joblib import load
from solution.utils import set_seed, set_s3fs, generate_file_path_s3_models
from solution.pipeline import set_date_transformer, set_preprocessor, set_y_transformer

RANDOM_STATE = set_seed()

df      = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/df.parquet")
X_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_train.parquet")
X_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/X_test.parquet")
y_train = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_train.parquet")["price_sqm"]
y_test  = pd.read_parquet("https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/2_preprocessing/y_test.parquet")["price_sqm"]

# Create filesystem object
fs = set_s3fs()

date_transformer = set_date_transformer()

preprocessor = set_preprocessor()

y_transformer = set_y_transformer()

# Importing fine-tuned RF and GB models
FILE_PATH_S3 = generate_file_path_s3_models("rf_model_final.joblib")

with fs.open(FILE_PATH_S3, mode="rb") as model:
    rf_model_final = load(model)

FILE_PATH_S3 = generate_file_path_s3_models("gb_model_final.joblib") 

with fs.open(FILE_PATH_S3, mode="rb") as model:
    gb_model_final = load(model)

# %%
