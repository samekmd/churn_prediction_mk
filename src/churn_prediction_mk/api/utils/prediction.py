import logging

import joblib
import pandas as pd 
from pathlib import Path

logger = logging.getLogger("app.main")

class ModelService:
    def __init__(self) -> None:
        self._load_artifacts()
        
    def _load_artifacts(self) -> None:
        """Load all artifacts from the local project folder"""
        logger.info("Loading artifacts from local project folder")
        
        # Base dir 
        base_dir = Path(__file__).resolve().parents[2]
        
        # define base paths 
        artifacts_dir = base_dir / "artifacts"
        models_dir = base_dir / "models"
        
        # Define paths to the preprocesing artifacts
        features_imputer_path = artifacts_dir / "[features]_ohe.joblib"
        target_encoder = artifacts_dir / "[target]_label.joblib"
        
        # Define path to the model file
        model_path = models_dir / "model.xgboost"
         
        # Loading all required artifacts 
        self.features_encoder = joblib.load(features_imputer_path)
        self.target_encoder = joblib.load(target_encoder)
        self.model = joblib.load(model_path)
        
        print(type(self.target_encoder))
        
        logger.info("Successfully loaded all artifacts")
        
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
         """Make predictions using the full pipeline.

        Args:
            features: DataFrame containing the input features

        Returns:
            Series containing the predictions
         """
         X_encoded = self.features_encoder.transform(features)
         
         # Get model predictions
         y_pred = self.model.predict(X_encoded)
         
         y_proba = self.model.predict_proba(X_encoded)
         
    
         y_decoded = self.target_encoder.inverse_transform(y_pred)
        
     
         results = pd.DataFrame({
            "Prediction": y_decoded,
            "Churn_Probability": y_proba[:, 1],  
            "No_Churn_Probability": y_proba[:, 0]  
         })
        
         return results
    

             
        