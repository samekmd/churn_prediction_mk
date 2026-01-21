import logging
import os 
import yaml

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger("src.data_preprocessing.preprocess_data")

def load_data() -> pd.DataFrame:
    """
    Load the raw data from disk
    
    :return: pd.DataFrame
    :rtype: DataFrame
    """
    
    data_path = "/home/samuel-machado/churn_prediction_mk/churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    logger.info(f"Loading raw data from {data_path}")
    data = pd.read_csv(data_path, index_col='customerID')
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


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets using parameters from params.yaml
    
    Args:
        data (pd.DataFrame): Input dataset
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    
    params = load_params()
    logger.info("Splitting into train and test sets...")
    train_data, test_data = train_test_split(
        data, test_size=params["test_size"],
        random_state=params["random_seed"]
    )
    
    return train_data, test_data
    
def save_artifacts(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """
    Save processed data and preprocessing artifacts
    
    Args:
        train_data (pd.DataFrame): Processed training data
        test_data (pd.DataFrame): Processed test data
    """
    
    # Save processed data 
    data_dir = "data/preprocessed"
    logger.info(f"Saving processed data to {data_dir}")
    
    train_path = os.path.join(data_dir, "train_preprocessed.csv")
    test_path = os.path.join(data_dir, "test_preprocessed.csv")
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    
def main() -> None:
    """Main function to orchestrate the preprocessing pipeline."""
    raw_data = load_data()
    train_data, test_data = split_data(raw_data)
    save_artifacts(train_data, test_data)
    logger.info("Data preprocessing completed")
    
if __name__ == "__main__":
    main()