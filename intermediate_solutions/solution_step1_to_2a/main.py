
## Exercice 1: Understanding data inputs
# %%

import duckdb
import os

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
X = df.drop(columns="price_sqm")  # X must contain only the features we'll learn from
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
    func = log_transform,
    inverse_func = inverse_log_transform)

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



### Exercice 7: Train your first Gradient Boosting model
# %%
from sklearn.ensemble import HistGradientBoostingRegressor

X = df.drop(columns=["price_sqm", "trans_date", "prop_type"])
y = df["price_sqm"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

gb_baseline = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
gb_baseline.fit(X_train, y_train)

# %%
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def print_metrics(model, split, X=X_train, y=y_train):
    """
    Print metrics for trained model
    """
    y_pred = model.predict(X)
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    print(f"{split} — RMSE: {rmse:.2f}  |  MAE: {mae:.2f}  |  R²: {r2:.4f}")


list = [("Train", X_train, y_train), ("Test", X_test, y_test)]
model = gb_baseline

for split, X, y in list:
    print_metrics(model, split, X, y)




### Exercice 8: Tuning Gradient Boosting hyperparameters
# %%

# Split features / target
X = df.drop(columns="price_sqm")
y = df["price_sqm"]

# Split train / test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

gb_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('GB', HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        early_stopping=True))
])

model = TransformedTargetRegressor(
    regressor=gb_pipeline,
    transformer=y_transformer
)


# %%
from sklearn.model_selection import GridSearchCV

param_grid_step1 = {
    "regressor__GB__max_iter": [200, 500],
    "regressor__GB__learning_rate": [0.1, 0.15, 0.2],
}

gs_step1 = GridSearchCV(
    estimator=model,
    param_grid=param_grid_step1,
    cv=2,
    scoring="r2",
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)
gs_step1.fit(X_train, y_train)

print(f"Best params : {gs_step1.best_params_}")
print(f"Best CV R² : {gs_step1.best_score_:.4f}")

# %%

def results_to_df(results_: dict):
    pattern = "param_regressor__GB__"
    pattern_len=len(pattern)
    params = [key for key in results_.keys() if key.startswith(pattern)]
    matching_keys = params + ["mean_train_score", "mean_test_score"]
    rename_params = {key: key[pattern_len:] for key in params}
    rename_params["mean_train_score"] = "train_r2"
    rename_params["mean_test_score"] = "cross_val_r2"
    results_df_ = pd.DataFrame(results_)[matching_keys].rename(columns=rename_params)

    return results_df_.sort_values("cross_val_r2", ascending=False)



# %%
df_step1 = results_to_df(gs_step1.cv_results_)

# %%
import matplotlib.pyplot as plt


def plot_results_cv(param_x:str, results_df_):
    param_group_by = [column for column in results_df_.columns.to_list() if column not in [param_x, "cross_val_r2", "train_r2"]][0]
    param_group_by_values = results_df_[param_group_by].unique()


    fig, ax = plt.subplots(figsize=(13, 4))

    for param_line in param_group_by_values:
        subset = results_df_[results_df_[param_group_by] == param_line].sort_values(param_x)
        line, = ax.plot(subset[param_x], subset["train_r2"], linestyle="--", marker="o", label=f"Train {param_group_by}={param_line}")
        ax.plot(subset[param_x], subset["cross_val_r2"], marker="x", color=line.get_color(), label=f"Cross-val {param_group_by}={param_line}")

    ax.set_xlabel(f"{param_x}")
    ax.set_ylabel("R²")
    ax.set_title(f"Joint optimisation of {param_group_by} and {param_x}")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()



# %%
plot_results_cv("max_iter", df_step1)

# %%
BEST_ITER = 1000  # to automatically catch the best hyperparameter, set to : gs_step1.best_params_["regressor__GB__max_iter"]
BEST_LR = 0.2  # to automatically catch the best hyperparameter, set to : gs_step1.best_params_["regressor__GB__learning_rate"]

# %%
param_grid_step2 = {
    "regressor__GB__max_depth" : [3, 5, 8],
    "regressor__GB__min_samples_leaf": [10, 20, 50],
}

gb_pipeline_step2 = Pipeline([
    ('preprocessing', preprocessor),
    ('GB', HistGradientBoostingRegressor(
        max_iter=BEST_ITER,
        learning_rate=BEST_LR,
        random_state=RANDOM_STATE,
        early_stopping=True
    ))
])

model_step2 = TransformedTargetRegressor(
    regressor=gb_pipeline_step2,
    transformer=y_transformer
)

gs_step2 = GridSearchCV(
    estimator=model_step2,
    param_grid=param_grid_step2,
    cv=2,
    scoring="r2",
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)

gs_step2.fit(X_train, y_train)

print(f"Best params : {gs_step2.best_params_}")
print(f"Best CV R² : {gs_step2.best_score_:.4f}")

# %%

df_step2 = results_to_df(gs_step2.cv_results_)

plot_results_cv("min_samples_leaf", df_step2)

# %%
BEST_DEPTH = 8 # to automatically catch the best hyperparameter, set to : gs_step2.best_params_["regressor__GB__max_depth"]
BEST_MIN_LEAF = 20 # to automatically catch the best hyperparameter, set to : gs_step2.best_params_["regressor__GB__min_samples_leaf"]

# %%

param_grid_step3 = {
    "regressor__GB__l2_regularization" : [0.0, 0.1, 0.3, 0.5, 1.0],
    "regressor__GB__max_iter" : [BEST_ITER],
}

gb_pipeline_step3 = Pipeline([
    ('preprocessing', preprocessor),
    ('GB', HistGradientBoostingRegressor(
        learning_rate=BEST_LR,
        max_depth=BEST_DEPTH,
        min_samples_leaf=BEST_MIN_LEAF,
        random_state=RANDOM_STATE,
        early_stopping=True
    ))
])

model_step3 = TransformedTargetRegressor(
    regressor=gb_pipeline_step3,
    transformer=y_transformer
)

gs_step3 = GridSearchCV(
    estimator=model_step3,
    param_grid=param_grid_step3,
    cv=2,
    scoring="r2",
    return_train_score=True,
    n_jobs=-1,
    verbose=1,
)

gs_step3.fit(X_train, y_train)

df_step3 = results_to_df(gs_step3.cv_results_)
plot_results_cv("l2_regularization", df_step3)

print(f"Best params : {gs_step3.best_params_}")
print(f"Best CV R² : {gs_step3.best_score_:.4f}")

# %%

BEST_L2 = 0 # to automatically catch the best hyperparameter, set to : gs_step3.best_params_["regressor__GB__l2_regularization"]


# %%

gb_final = HistGradientBoostingRegressor(
    max_iter=BEST_ITER,
    learning_rate=BEST_LR,
    max_depth=BEST_DEPTH,
    min_samples_leaf=BEST_MIN_LEAF,
    l2_regularization=BEST_L2,
    random_state=RANDOM_STATE,
)

X = df.drop(columns=["price_sqm"])
y = df["price_sqm"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
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
print("Final model trained.")


## Exercice 9: Evaluate the final Gradient Boosting model
# %%

list = [("train", X_train, y_train), ("test", X_test, y_test)]

for split, X, y in list:
    print_metrics(gb_model_final, split, X, y)

# %%
import matplotlib.pyplot as plt

y_pred_test = gb_model_final.predict(X_test)

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(y_test, y_pred_test, alpha=0.3, s=5, label="Predictions")

lims = [min(y_test.min(), y_pred_test.min()),
        max(y_test.max(), y_pred_test.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

ax.set_xlabel("Actual values (log)")
ax.set_ylabel("Predicted values (log)")
ax.set_title("Comparison of predicted values vs. actual values on the test set\n(Gradient Boosting)")
ax.legend()
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()


# %%

from joblib import dump

# Save the model to a file
dump(gb_model_final, 'gb_model_final.joblib')

