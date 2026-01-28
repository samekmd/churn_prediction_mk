import json 
import logging 
import os

import joblib 
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

logger = logging.getLogger("src.model_training.train_model")


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the feature-engineered training data.

    Returns:
        pd.DataFrame: A dataframe containing the training data.
    """
    X_train_path = "data/processed/X_train_processed.csv"
    y_train_path = "data/processed/y_train_processed.csv"
    logger.info(f"Loading feature data from {X_train_path}")
    
    X_train_data = pd.read_csv(X_train_path)
    y_train_data = pd.read_csv(y_train_path)
    
    return X_train_data, y_train_data


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
    

def create_train_model(X_train_data: pd.DataFrame, y_train_data: pd.Series, params: dict [str, int | float]) -> None:
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
        n_estimators=params['n_estimators']
    )
    
    # Training the model
    model.fit(X_train_data, y_train_data)
    
    # Saving the model
    save_training_artifacts(model)
    
    
    
def main():
    """Main function to orchestrate the model training process."""
    X_train, y_train = load_data()
    params = load_params()
    create_train_model(X_train, y_train, params)
    logger.info("Model training completed")
 
   
if __name__ == "__main__":
    main()