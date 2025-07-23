import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test data."""
    try:
        train_data = pd.read_csv(train_path).dropna(subset=['content'])
        test_data = pd.read_csv(test_path).dropna(subset=['content'])
        logging.info("Processed train and test data loaded.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to load train/test data: {e}")
        raise

def extract_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from DataFrame."""
    try:
        X = df['content'].values
        y = df['sentiment'].values
        return X, y
    except Exception as e:
        logging.error(f"Failed to extract features/labels: {e}")
        raise

def build_bow_features(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Build TF-IDF features for train and test data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        logging.info("TF-IDF features built.")
        return X_train_tfidf, X_test_tfidf, vectorizer
    except Exception as e:
        logging.error(f"Failed to build TF-IDF features: {e}")
        raise

def save_features(X_tfidf, y, path: str) -> None:
    """Save features and labels to CSV."""
    try:
        df = pd.DataFrame(X_tfidf.toarray())
        df['label'] = y
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Features saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save features: {e}")
        raise

def main() -> None:
    """Main function to orchestrate feature building."""
    try:
        params = load_params("params.yaml")
        max_features = params['build_features']['max_features']

        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        X_train, y_train = extract_features_labels(train_data)
        X_test, y_test = extract_features_labels(test_data)

        X_train_tfidf, X_test_tfidf, _ = build_bow_features(X_train, X_test, max_features)

        save_features(X_train_tfidf, y_train, "data/interim/train_tfidf.csv")
        save_features(X_test_tfidf, y_test, "data/interim/test_tfidf.csv")
        logging.info("Feature building completed successfully.")
    except Exception as e:
        logging.error(f"Feature building failed: {e}")
        raise

if __name__ == "__main__":
    main()