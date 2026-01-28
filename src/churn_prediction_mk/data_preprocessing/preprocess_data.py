import logging
import os 
import yaml

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


logger = logging.getLogger("src.data_preprocessing.preprocess_data")

def load_data() -> pd.DataFrame:
    """
    Load the raw data from disk
    
    :return: pd.DataFrame
    :rtype: DataFrame
    """
    
    data_path = "data/raw/churn.csv"
    logger.info(f"Loading raw data from {data_path}")
    data = pd.read_csv(data_path)
    return data


def load_params() -> dict[str, float | int]:
    """
    Load preprocessing parameters from params.yaml
    
    Returns:
        dict[str, Any]: dictionary containing preprocessing parameters.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess_data"]


def split_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets using parameters from params.yaml
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    
    params = load_params()
    logger.info("Splitting into train and test sets...")
    X = data.drop(columns=['Churn'])
    y = data['Churn']
    X_train, X_test, y_train, y_test  = train_test_split(
        X,y,
        test_size=params["test_size"],
        random_state=params["random_seed"],
        stratify=y
    )
    
    return  X_train, X_test, y_train, y_test
    
def save_artifacts(X_train: np.ndarray, 
                   X_test: np.ndarray,
                   y_train: np.ndarray,
                   y_test: np.ndarray) -> None:
    """
    Save processed data and preprocessing artifacts
    
    Args:
        train_data (pd.DataFrame): Processed training data
        test_data (pd.DataFrame): Processed test data
    """
    
    # Save processed data 
    data_dir = "data/preprocessed"
    logger.info(f"Saving processed data to {data_dir}")
    
    X_train_data = pd.DataFrame(X_train)
    X_test_data = pd.DataFrame(X_test)
    y_train_data = pd.DataFrame(y_train)
    y_test_data = pd.DataFrame(y_test)
    
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")
    
    X_train_data.to_csv(X_train_path, index=False)
    X_test_data.to_csv(X_test_path, index=False)
    y_train_data.to_csv(y_train_path, index=False)
    y_test_data.to_csv(y_test_path, index=False)
    
    
def main() -> None:
    """Main function to orchestrate the preprocessing pipeline."""
    raw_data = load_data()
    X_train, X_test, y_train, y_test = split_data(raw_data)
    save_artifacts(X_train, X_test, y_train, y_test)
    logger.info("Data preprocessing completed")
    
if __name__ == "__main__":
    main()