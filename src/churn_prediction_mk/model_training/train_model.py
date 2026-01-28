import json 
import logging 
import os

import joblib 
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier
from pathlib import Path

logger = logging.getLogger("src.model_training.train_model")


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the feature-engineered training data.

    Returns:
        pd.DataFrame: A dataframe containing the training data.
    """
    X_train_path = "data/processed/X_train_processed.csv"
    y_train_path = "data/processed/y_train_processed.csv"
    X_test_path = "data/processed/X_test_processed.csv"
    y_test_path = "data/processed/y_test_processed.csv"
    logger.info(f"Loading feature data from {X_train_path}")
    
    X_train_data = pd.read_csv(X_train_path)
    y_train_data = pd.read_csv(y_train_path)
    X_test_data = pd.read_csv(X_test_path)
    y_test_data = pd.read_csv(y_test_path)
    
    return X_train_data, y_train_data, X_test_data, y_test_data


def load_params() -> dict[str, float | int]:
    """
    Load model hyperparameters for the train stage from params.yaml.

    Returns:
        dict[str, int | float]: dictionary containing model hyperparameters.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]



def save_training_artifacts(model: XGBClassifier) -> None:
    """
    Save model artifacts to disk 
    
    Args:
        model (xgboost.XGBClassifier): Trained xgboost model
    """ 
    models_dir = "models"
    model_path = os.path.join(models_dir, "model.xgboost")
    
    # Save the model
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model,model_path) 
    

def create_train_model(X_train_data: pd.DataFrame, 
                       y_train_data: pd.Series,
                       X_test_data: pd.DataFrame, 
                       y_test_data: pd.Series,
                       params: dict [str, int | float]) -> None:
    """
    Create and train the model
    
    Args:
        train_data: Data for training
        params: hiperparameters for the model
        
    Returns:
        XGBClassifier: Model trained
    """
    # Creating the model
    model = XGBClassifier(
        objective='binary:logistic',
        colsample_bytree=params['colsample_bytree'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        eval_metric=["logloss", "auc", "error"]
    )
    
    # Training the model
    model.fit(X_train_data, 
              y_train_data,
              eval_set=[(X_train_data, y_train_data), (X_test_data, y_test_data)]
              )
    
    # Saving training metrics 
    results = model.evals_result()
    
    metrics = {
        'train_logloss': float(results['validation_0']['logloss'][-1]),
        'val_logloss': float(results['validation_1']['logloss'][-1]),
        'train_auc': float(results['validation_0']['auc'][-1]),
        'val_auc': float(results['validation_1']['auc'][-1]),
        'train_error': float(results['validation_0']['error'][-1]),
        'val_error': float(results['validation_1']['error'][-1]),
        'n_estimators_used': len(results['validation_0']['logloss'])
    }
    
    base_dir = Path(__file__).resolve().parents[3]
    
    metrics_path = base_dir / "metrics/training.json"
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Saving the model
    save_training_artifacts(model)
    
    
    
def main():
    """Main function to orchestrate the model training process."""
    X_train, y_train, X_test, y_test = load_data()
    params = load_params()
    create_train_model(X_train, y_train, X_test, y_test, params)
    logger.info("Model training completed")
 
   
if __name__ == "__main__":
    main()