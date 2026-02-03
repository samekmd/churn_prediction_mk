import logging
import os

import joblib
import pandas as pd 
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger("app.main")

class ModelService:
    def __init__(self) -> None:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        print(f"MlFLow URI: {mlflow_uri}")
        self._load_artifacts()
        
        
    
        
    
        
    def _load_artifacts(self) -> None:
        """Load the registered model from MLflow Model Registry related artifacts from its run"""
       
        # Get run id from model version metadata 
        client = MlflowClient()
        model_version = client.get_latest_versions(
            name="model",
            stages=["None", "Production"]
        )
           
        print("\n===== MODEL REGISTRY DEBUG =====")
        for v in model_version:
                print("name:", v.name)
                print("version:", v.version)
                print("run_id:", v.run_id)
                print("source:", repr(v.source))
                print("status:", v.status)
                print("--------------------------------")
        
        # Load model from registry
        logger.info("Loading registered model from MLFlow Model Registry")
        self.model = mlflow.xgboost.load_model("models:/model/latest")
        
        run_id = model_version.run_id
        run = client.get_run(run_id)
        
        print("\n===== RUN DEBUG =====")
        print("run_id:", run.info.run_id)
        print("artifact_uri:", run.info.artifact_uri)
        print("lifecycle_stage:", run.info.lifecycle_stage)
            
        # Load related artifacts 
        logger.info(f"Loading artifacts from run {run_id}")
        
        print("\n===== ARTIFACT TREE =====")

        def list_artifacts(path=""):
            artifacts = client.list_artifacts(run_id, path)
            for a in artifacts:
                print(f"{a.path} (is_dir={a.is_dir})")
                if a.is_dir:
                    list_artifacts(a.path)
                    
        list_artifacts()
        
        artifacts_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="encoders")
        
        print(f"Pasta dos artefatos: {artifacts_dir}")
        
        ohe_path = Path(artifacts_dir) / "[features]_ohe.joblib"
        label_path = Path(artifacts_dir) / "[target]_label.joblib"
        
        self.features_encoder = joblib.load(ohe_path)
        self.target_encoder = joblib.load(label_path)
        
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
    

             
        