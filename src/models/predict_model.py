import pandas as pd
import pickle
import json
import logging
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_model(model_path: str):
    """Load a trained model from disk."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def load_test_data(test_path: str) -> pd.DataFrame:
    """Load test data from CSV file."""
    try:
        test_data = pd.read_csv(test_path)
        logging.info(f"Test data loaded from {test_path}")
        return test_data
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info("Model evaluation completed.")
        return metrics_dict
    except Exception as e:
        logging.error(f"Failed to evaluate model: {e}")
        raise

def save_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model prediction and evaluation."""
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_bow.csv")
        X_test = test_data.drop(columns=['label']).values
        y_test = test_data['label'].values
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")
        logging.info("Prediction and evaluation pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Prediction and evaluation pipeline failed: {e}")
        raise
if __name__ == "__main__":
    main()