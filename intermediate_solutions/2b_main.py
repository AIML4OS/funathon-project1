
# Exercice 4: Train your first Random Forest model
# %%

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

RANDOM_STATE = 202605


def log_transform(y):
    return np.log10(y)


def inverse_log_transform(y):
    return 10 ** y


y_transformer = FunctionTransformer(
    func=log_transform,
    inverse_func=inverse_log_transform)


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

# %%
from sklearn.ensemble import RandomForestRegressor

# create RandomForestRegressor instance with selected hyperparameters
rf = RandomForestRegressor(
    n_estimators=50,
    max_features="sqrt",
    min_samples_leaf=10,
    oob_score=True # for calculating total oob error for the RF
)

# Defining train and test sets
X_train = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/X_train.parquet')
X_test  = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/X_test.parquet')

X_train = X_train.drop(columns=["prop_type", "trans_date"])
X_test  =  X_test.drop(columns=["prop_type", "trans_date"])

y_train = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/y_train.parquet')["price_sqm"]
y_test  = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/y_test.parquet')["price_sqm"]

# Train the model
rf.fit(X_train, y_train)

# %%

rf.oob_score_

# %%
from sklearn.metrics import mean_squared_error

# Predictions on train set
y_pred_test = rf.predict(X_test)

# Print the error
print(mean_squared_error(y_test, y_pred_test))



# Exercice 5: Tuning a random forest's hyperparameters
# %%

# Sample the train dataset using Pandas' index
y_train_df = pd.DataFrame(y_train)
y_train_df["quantile"] = pd.qcut(y_train_df["price_sqm"], q=100, labels=False) ## allows to discretly cut along quantiles
y_sub = y_train_df.groupby("quantile").sample(frac=0.1, random_state= RANDOM_STATE)  # sampling by quantile 

y_sub = y_sub["price_sqm"] # converting to pandas.series
X_sub = X_train.filter(items=y_sub.index, axis=0 )  # sampling X_train


# %%
import warnings

metric = "r2"
min_estimators=5
max_estimators=150

rf = RandomForestRegressor(
    warm_start=True,
    **rf_params,
)

oob_scores = []
warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
# filterwarnings remove some warnings messages
for n in range(min_estimators, max_estimators, 20):
    rf.set_params(n_estimators=n)
    rf.fit(X_sub, y_sub)
    if metric == "r2":
        oob_scores.append((n, 1 - rf.oob_score_))
    elif metric == "neg_root_mean_squared_error":
        mse = np.mean((y_sub - rf.oob_prediction_) ** 2)
        oob_scores.append((n, np.sqrt(mse)))
    else:
        mae = np.mean(np.abs(y_sub - rf.oob_prediction_))
        oob_scores.append((n, mae))
warnings.resetwarnings()

# %%
import matplotlib.pyplot as plt

rf_params = {
    "max_depth": 8,
    "max_features": "sqrt",
    "min_samples_split": 5,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
}

def rf_error_oob_plot(X_train,
                      y_train,
                      subsample=0.1,
                      min_estimators=15,
                      max_estimators=150,
                      metric='r2',
                      **rf_params):
    """
    Plot error OOB convergence by the number of trees

    Args:
        X_train: features
        y_train: target
        subsample: rate of sample for X_train
        min_estimators: number min of trees
        max_estimators: number max of trees
        metric : 'r2',  'rmse' or 'mae'
    """

    # --- Stratified sampling of training set ---
    y_train_df = pd.DataFrame(y_train)
    y_train_df["quantile"] = pd.qcut(y_train_df["price_sqm"], q=100, labels=False) ## allows to discretly cut along quantiles
    y_sub = y_train_df.groupby("quantile").sample(frac=0.1, random_state= RANDOM_STATE)  # sampling by quantile 

    y_sub = y_sub["price_sqm"] # converting to pandas.series
    X_sub = X_train.filter(items=y_sub.index, axis=0 )  # sampling X_train

    # --- Training with warm start ---
    rf = RandomForestRegressor(
        oob_score=True,
        warm_start=True,
        **rf_params,
    )

    oob_scores = []
    warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
    for n in range(min_estimators, max_estimators, 5):
        rf.set_params(n_estimators=n)
        rf.fit(X_sub, y_sub)
        if metric == "r2":
            oob_scores.append((n, 1 - rf.oob_score_))
        elif metric == "neg_root_mean_squared_error":
            mse = np.mean((y_sub - rf.oob_prediction_) ** 2)
            oob_scores.append((n, np.sqrt(mse)))
        else:
            mae = np.mean(np.abs(y_sub - rf.oob_prediction_))
            oob_scores.append((n, mae))
    warnings.resetwarnings()

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    xs, ys = zip(*oob_scores)

    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlim(min_estimators, max_estimators)
    ax.set_xlabel("n_trees")
    ax.set_ylabel(f"OOB error ({metric})")
    plt.close(fig)

    return fig


# %%
oob_error_ntrees = rf_error_oob_plot(X_train=X_train,
                                     y_train=y_train,
                                     subsample=0.1,
                                     min_estimators=5,
                                     max_estimators=150,
                                     metric="r2")
oob_error_ntrees

# %%

# Split features / target
X_train = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/X_train.parquet')
X_test  = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/X_test.parquet')
y_train = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/y_train.parquet')["price_sqm"]
y_test  = pd.read_parquet('s3://projet-funathon/2026/project1/data/2_preprocessing/y_test.parquet')["price_sqm"] 


# %%

param_grid = {
    "regressor__RF__n_estimators": [80],
    "regressor__RF__max_features": ["sqrt"],
    "regressor__RF__min_samples_leaf": [40, 50, 75]
}

# %%
from sklearn.model_selection import GridSearchCV

# Grid search
grid_search = GridSearchCV(
    estimator=model, # it is the TransformedTargetRegressor created in the preprocessing part
    param_grid=param_grid,
    cv=4,  # number of folds
    scoring="r2", # 'r2' or 'neg_root_mean_squared_error' or 'neg_mean_absolute_error'
    n_jobs=-1,
    verbose=1
)

# Train
grid_search.fit(X_train, y_train)

# %%

print(grid_search.best_params_)

# %%

rf_model_final = grid_search.best_estimator_
print(type(rf_model_final))

rf_model_final.fit(X_train, y_train)


# Exercice 6: Model evaluation
# %%

y_pred_test = rf_model_final.predict(X_test)

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae  = mean_absolute_error(y_test, y_pred_test)
r2   = r2_score(y_test, y_pred_test)

print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.4f}")

# %%
import matplotlib.pyplot as plt


def predicted_actual_plot(y_test, y_pred_test, model_name):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_test, y_pred_test, alpha=0.3, s=5, label="Predictions")

    lims = [min(y_test.min(), y_pred_test.min()),
            max(y_test.max(), y_pred_test.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual values (log)")
    ax.set_ylabel("Predicted values (log)")
    ax.set_title(f"Comparison of predicted values vs. actual values on the test set\n({model_name})")
    ax.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    return fig


predicted_actual_plot(y_test, y_pred_test, "Random Forest")

