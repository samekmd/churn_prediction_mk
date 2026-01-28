import logging
import os

import joblib
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

logger = logging.getLogger("src.feature_engineering.engineer_features")

def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed train and test DataSets
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    X_train_path = "data/preprocessed/X_train.csv"
    X_test_path = "data/preprocessed/X_test.csv"
    y_train_path = "data/preprocessed/y_train.csv"
    y_test_path = "data/preprocessed/y_test.csv"
    logger.info(f"Loading preprocessed data from {X_train_path}, {X_test_path},{y_train_path} ,{y_test_path}")
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)['Churn']
    y_test = pd.read_csv(y_test_path)['Churn']
    return X_train, X_test, y_train, y_test


def load_params() -> dict[str, float | int]:
    """
    Load preprocessing parameters from params.yaml
    
    Returns:
        dict[str, Any]: dictionary containing preprocessing parameters.
    """
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["columns_process"]


def engineer_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, LabelEncoder]:
    """Apply feature engineering transformations
        Args:
        train_preprocessed (pd.DataFrame): Training dataset
        test_preprocessed (pd.DataFrame): Test dataset

    Returns:
        tuple containing:
            pd.DataFrame: Engineered training features
            pd.DataFrame: Engineered test features
            LabelEncoder: Target Encoder
            ColumnTransformer: Features Encoder
    """
    logger.info("Engineering features...")
    feat_cols = load_params()
    
    cat_cols = feat_cols['cat_cols']
    num_cols = feat_cols['num_cols']
    
    target_encoder = LabelEncoder()
    
    features_encoder = ColumnTransformer(   transformers=[
        (
            'OneHot', OneHotEncoder(handle_unknown='ignore'), cat_cols
        ),
        (
            'Standart', StandardScaler(), num_cols 
        )
    ],
        remainder='drop'
    )

    X_train = X_train.drop(columns=['gender'])
    X_test = X_test.drop(columns=['gender'])
    
    X_train_enc = features_encoder.fit_transform(X_train)
    X_test_enc = features_encoder.transform(X_test)
    
    y_train_enc = target_encoder.fit_transform(y_train) 
    y_test_enc = target_encoder.transform(y_test) 
    
    X_train_preprocessed = pd.DataFrame(X_train_enc)
    X_test_preprocessed = pd.DataFrame(X_test_enc)
    y_train_preprocessed = pd.Series(y_train_enc, name='Churn')
    y_test_preprocessed = pd.Series(y_test_enc, name='Churn')
     
    return X_train_preprocessed, X_test_preprocessed, y_train_preprocessed, y_test_preprocessed, features_encoder, target_encoder


def save_artifacts(
    X_train_processed: pd.DataFrame, 
    X_test_processed: pd.DataFrame,
    y_train_processed: pd.Series,
    y_test_processed: pd.Series,
    feat_encoder: ColumnTransformer,
    target_encoder: LabelEncoder
) -> None:
    """
    Save engineered features and scaler
    
    Args:
        train_processed (pd.DataFrame): Engineered training data
        test_processed (pd.DataFrame): Engineered test data
        feat_encoder: Fitted enconder
    """ 
    
    # Save processed data
    output_dir = "data/processed"
    logger.info(f"Saving engineered features to {output_dir}")
    
    X_train_path = os.path.join(output_dir, "X_train_processed.csv")
    X_test_path = os.path.join(output_dir, "X_test_processed.csv")
    
    y_train_path = os.path.join(output_dir, "y_train_processed.csv")
    y_test_path = os.path.join(output_dir, "y_test_processed.csv")
    
    X_train_processed.to_csv(X_train_path, index=False)
    X_test_processed.to_csv(X_test_path, index=False)
    y_train_processed.to_csv(y_train_path, index=False)
    y_test_processed.to_csv(y_test_path, index=False)
    
    # Save features encoder
    feat_encoder_path = os.path.join("artifacts", "[features]_ohe.joblib")
    logger.info(f"Saving scaler to {feat_encoder_path}")
    joblib.dump(feat_encoder, feat_encoder_path)
    
    # Save target encoder
    target_encoder_path = os.path.join("artifacts", "[target]_label.joblib")
    logger.info(f"Saving scaler to {target_encoder_path}")
    joblib.dump(target_encoder, target_encoder_path)
    
def main() -> None:
    """
    Main function to orchestrate feature engineering pipeline.
    """
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    X_train_enc, X_test_enc, y_train_enc, y_test_enc, feat_encoder, target_encoder = engineer_features(X_train, X_test, y_train, y_test)
    save_artifacts(X_train_enc, X_test_enc, y_train_enc, y_test_enc, feat_encoder, target_encoder)
    logger.info("Feature engineering completed")
    
    
if __name__ == "__main__":
    main()
    