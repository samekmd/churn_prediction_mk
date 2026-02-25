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
        """
        Load the registered model from MLflow Model Registry.
        """
        client = MlflowClient()
        model_name = "model"

        # 1. Busca as versões disponíveis
        model_versions = client.get_latest_versions(name=model_name)
        if not model_versions:
            raise RuntimeError(f"No registered versions found for model '{model_name}'")

        # 2. Lógica de prioridade de Stage melhorada
        # Usamos .lower() para evitar erros de comparação de string
        stage_priority = ["production", "staging"]
        selected_version = None

        for stage in stage_priority:
            for v in model_versions:
                if v.current_stage and v.current_stage.lower() == stage:
                    selected_version = v
                    break
            if selected_version:
                break

        # Se não achou em nenhum stage, pega a versão mais recente (última da lista)
        if not selected_version:
            selected_version = model_versions[0]

        logger.info(
            f"Selected model version {selected_version.version} "
            f"(stage: {selected_version.current_stage})"
        )

        # 3. CONSTRUÇÃO DA URI (A forma mais segura é usar a versão direta)
        # Em vez de 'models:/model/Production', usamos 'models:/model/1' 
        # Isso evita problemas se o stage mudar enquanto o código carrega
        model_uri = f"models:/{model_name}/{selected_version.version}"

        logger.info(f"Loading model from URI: {model_uri}")

        try:
            # Carrega como pyfunc (já que você salvou como PythonModel wrapper)
            # Isso já traz o XGBoost + Encoders serializados no seu Wrapper
            self.model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model from {model_uri}: {e}")
            # Fallback: tentar carregar direto da source (o caminho físico no S3/Local)
            logger.info(f"Attempting fallback to source: {selected_version.source}")
            self.model = mlflow.pyfunc.load_model(selected_version.source)

        logger.info("Model loaded successfully")
        
    
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
    

             
        