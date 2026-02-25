import logging 
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import mlflow



logger = logging.getLogger("src.model_evaluation.evaluate_model")

def load_model() -> XGBClassifier:
    """
    Load the xgboost model from disk
    
    Returns:
        XGBXClassier model
    """
    model_path = "models/model.xgboost"
    model =  joblib.load(model_path)
    
    return model


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the test data from disk
    
    Returns:
        tuple containing:
            pd.DataFrame: Test Features 
            pd.Series: Test Labels
    """
    X_test_path = "data/processed/X_test_processed.csv"
    y_test_path = "data/processed/y_test_processed.csv"
    logger.info(f"Loading test data from {X_test_path}")
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    return X_test, y_test

def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series 
) -> None:
    """
    Evaluate the model and generate performance metrics
    
    Args:
        model (XGBClassifier): Trained xgboost model
        X (pd.DataFrame): Test Features
        y_true (pd.Series): True Labels
    """
    base_dir = Path(__file__).resolve().parents[2]
    
    # Set up mlflow experiment
    # mlflow.set_tracking_uri(f"file://{base_dir}/mlruns")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    logger.info(f"MLFLOW URI {os.getenv("MLFLOW_TRACKING_URI")}")
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
    
    # Getting run_id for latest MLFlow run  
    runs = mlflow.search_runs(
        experiment_ids=[os.getenv("MLFLOW_EXPERIMENT_ID")],
        order_by=["start_time DESC"]
    )
    run_id = runs.iloc[0].run_id
    
    with mlflow.start_run(run_id=run_id):
    
        # Generate model predictions
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        evaluation = {"classification_report": report, "confusion_matrix": cm}
        
        # Log metrics (DVC)
        logger.info(f"Classification Report: \n{classification_report(y_test, y_pred)}")
        evaluation_path = base_dir / "metrics/evaluation.json"
        with open(evaluation_path, "w") as f:
            json.dump(evaluation, f, indent=2)

        # Log Metrics (MLflow)
        mlflow.log_metrics(
            {
              "test_accuracy":report['accuracy'],
              "test_precision_weighted":report['weighted avg']['precision'],
              "test_recall_weighted":report['weighted avg']['recall'],
              "test_f1_weighted":report['weighted avg']['f1-score'],
            }
        )    
        

def main() -> None:
     """Main function to orchestrate the model evaluation process."""
     model = load_model()
     X_test, y_test = load_test_data()
     evaluate_model(model, X_test, y_test)
     logger.info("Model evaluation completed")
     
if __name__ ==  "__main__":
    main()
    
    