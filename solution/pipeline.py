from sklearn.pipeline import Pipeline



def set_pipeline(ml_name, ml_model):
    ml_pipeline = Pipeline([
        (ml_name, ml_model),
    ])

    return ml_pipeline
