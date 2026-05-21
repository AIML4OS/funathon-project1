
## Exercice 1: Understanding data inputs
# %%

import duckdb
import os

# Create a non-persistent connection (the database exists only while the connection is alive and disappears when it is closed)
con = duckdb.connect(database=":memory:")

# %%

# We load all transactions made in France between 2010 and 2022
trans = con.sql(
    """
        SELECT * FROM read_parquet('s3://projet-funathon/2026/project1/data/1_input/transactions_EN.parquet')
    """).to_df()



# %%
import pandas as pd

trans = trans[trans["prop_loc_dep"].isin(["75", "77", "78", "91", "92", "93", "94", "95"])]


## Exercice 2: Analyzing data inputs
# %%

trans["price_sqm"] = trans["price"] / trans["farea"]

# %%

import numpy as np
import matplotlib.pyplot as plt

y = trans["price_sqm"]
p = np.percentile(y, 99.5)

fig, axes = plt.subplots(4, 1, figsize=(12, 15))

for ax, (data, label) in zip(axes, [(y, "Y"), (y[y <= p], "Y filtered"), (np.log(y), "log(Y)"), (np.log(y[y <= p]), "log(Y) filtered")]):
    ax.hist(data, bins="auto", edgecolor="white", color="#334887", alpha=0.95)
    ax.set_title(label)
    ax.set_xlabel("Price per square meter")
    ax.set_ylabel("Number of transactions")

plt.tight_layout()
plt.show()

# %%

fig, axes = plt.subplots(2, 1, figsize=(12, 15))

for ax, (data, label) in zip(axes, [(y[y <= 2000], "Y below 2000€ per sqm"), (y[y <= 500], "Y below 500€ per sqm")]):
    ax.hist(data, bins="auto", edgecolor="white", color="#334887", alpha=0.95)
    ax.set_title(label)
    ax.set_xlabel("Price per square meter")
    ax.set_ylabel("Number of transactions")

plt.tight_layout()
plt.show()

# %%

n0 = trans.shape[0]
print(f"{n0} rows before filtering")

# Apply some deterministic threshold on the dataframe
trans = trans[(trans["price_sqm"] < 200000) & (trans["price_sqm"] > 100)]

print(f"{trans.shape[0]} rows after deterministic filtering")

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

n1 = trans.shape[0]

print(f"{n1} rows after deterministic and statistic filtering")


# %%
print(f'Applying these filters methods has dropped about {((n0 - n1)/n0)*100:.2f} % of the transactions.')

# %%

trans = trans.dropna(subset = "price_sqm")

# %%

df = trans.drop(columns=[
    "price", "prop_loc_dep", "prop_loc_citycode", "dist_tosea"
])


# %%
# Printing all rows containing at least one NA
print(df[df.isna().any(axis=1)])

# Filtering NA values
df = df.dropna()

# %%

df["prop_type"] = pd.Categorical(
    df["prop_type"],
    categories=["1", "2"],
    ordered=False
).rename_categories({"1": "House", "2": "Flat"})

# %%

counts = df.value_counts("prop_year_harm").reset_index()
counts[counts["prop_year_harm"] < 1850].describe() # there more than 500 different years of construction, going from 13th century to now. Maybe we can bundle together years before 1850 and group them by decade

counts_10 = ((df["prop_year_harm"] // 10)*10).value_counts().reset_index()  # 82 modalities
counts_10[counts_10["prop_year_harm"] < 1850].describe()  # years before 1850 represent 64 modalities with maximal class of about two thousands operations - ok
counts_10[counts_10["prop_year_harm"] < 1850]["count"].sum()

# Replacing year of construction by decade and merging together all years before 1850
df['prop_year_harm_10'] = (df['prop_year_harm'] // 10)*10
df['prop_year_harm_10'] = df['prop_year_harm_10'].where(df['prop_year_harm_10'] >= 1850, 1840)

# Dropping old column
df = df.drop(columns=["prop_year_harm"])

# %%

from sklearn.model_selection import train_test_split

# Split features / target
X = df.drop(columns=["price_sqm"])  # X must contain only the features we'll learn from
y = df["price_sqm"]  # target must be a dataframe with 1 column

# Split train / test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)


## Exercice 3: The scikit-learn pipeline
# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


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

# %%
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def log_transform(y):
    return np.log10(y)

def inverse_log_transform(y):
    return 10 ** y

y_transformer = FunctionTransformer(
    func=log_transform,
    inverse_func=inverse_log_transform)

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

rf_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('RF', RandomForestRegressor(**rf_params))
])

model = TransformedTargetRegressor(
    regressor=rf_pipeline,
    transformer=y_transformer
)


