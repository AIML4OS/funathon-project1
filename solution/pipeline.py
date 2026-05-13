from sklearn.pipeline import Pipeline
from preprocessing import preprocessor, y_transformer
from sklearn.compose import TransformedTargetRegressor


def set_pipeline(ml_name, ml_model):
    ml_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (ml_name, ml_model),
    ])

    ml_model_pipeline = TransformedTargetRegressor(
        regressor=ml_pipeline,
        transformer=y_transformer
    )

    return ml_model_pipeline
