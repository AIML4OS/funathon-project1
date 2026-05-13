from sklearn.pipeline import Pipeline
from preprocess import set_preprocessor, set_y_transformer
from sklearn.compose import TransformedTargetRegressor


def set_pipeline(ml_name, ml_model):
    ml_pipeline = Pipeline([
        ("preprocessor", set_preprocessor()),
        (ml_name, ml_model),
    ])

    ml_model_pipeline = TransformedTargetRegressor(
        regressor=ml_pipeline,
        transformer=set_y_transformer()
    )

    return ml_model_pipeline
