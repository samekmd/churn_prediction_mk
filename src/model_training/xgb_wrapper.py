import mlflow.pyfunc 
import pandas as pd

class XGBPipelineWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, features_encoder, target_encoder):
        self.model = model
        self.features_encoder = features_encoder
        self.target_encoder = target_encoder
        
    def predict(self, context, model_input):
        X_encoded = self.features_encoder.transform(model_input)
        preds = self.model.predict(X_encoded)
        return self.target_encoder.inverse_transform(preds.astype(int))
