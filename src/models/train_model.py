import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
import os

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

def load_train_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data and split into features and labels."""
    try:
        train_data = pd.read_csv(path)
        x_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values
        logging.info(f"Training data loaded from {path}")
        return x_train, y_train
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)
        logging.info("RandomForest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to train model: {e}")
        raise

def save_model(model: RandomForestClassifier, path: str) -> None:
    """Save the trained model to disk."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model training."""
    try:
        params = load_params("params.yaml")
        n_estimators = params['train_model']['n_estimators']
        max_depth = params['train_model']['max_depth']

        x_train, y_train = load_train_data("data/interim/train_bow.csv")
        model = train_model(x_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Model training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()