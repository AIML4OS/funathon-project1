# %%
import sys
sys.path.append("..") 

from joblib import load
import os
import s3fs
from solution.utils import set_seed
from solution.preprocess import set_date_transformer, set_preprocessor, set_y_transformer, complete_pre_processing

RANDOM_STATE = set_seed()
df = complete_pre_processing()

# Create filesystem object
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


date_transformer = set_date_transformer()

preprocessor = set_preprocessor()

y_transformer = set_y_transformer()


# Importing fine-tuned models
FILE_PATH_S3 = "projet-funathon/2026/project1/models/rf_model_final.joblib" 

with fs.open(FILE_PATH_S3, mode="rb") as model:
    rf_model_final = load(model)

FILE_PATH_S3 = "projet-funathon/2026/project1/models/gb_model_final.joblib" 

with fs.open(FILE_PATH_S3, mode="rb") as model:
    gb_model_final = load(model)

# %%