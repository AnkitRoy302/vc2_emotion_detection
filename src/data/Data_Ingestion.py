import os
import logging
from typing import Tuple
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

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

def load_dataset(url: str) -> pd.DataFrame:
    """Load dataset from a given URL."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Dataset loaded from {url}")
        return df
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset for binary sentiment classification."""
    try:
        df = df.drop(columns=['tweet_id'])
        logging.info("Dropped 'tweet_id' column.")
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        logging.info("Filtered for 'happiness' and 'sadness' sentiments.")
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logging.info("Encoded sentiments: 'happiness' as 1, 'sadness' as 0.")
        return final_df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and testing sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train ({len(train_data)}) and test ({len(test_data)}) sets.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, train_path: str, test_path: str) -> None:
    """Save train and test data to CSV files."""
    try:
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Train data saved to {train_path}")
        logging.info(f"Test data saved to {test_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main() -> None:
    """Main function to orchestrate data ingestion."""
    try:
        params = load_params("params.yaml")
        test_size: float = params['Data_Ingestion']['test_size']
        url: str = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = load_dataset(url)
        final_df = preprocess_dataset(df)
        train_data, test_data = split_data(final_df, test_size)
        save_data(train_data, test_data, 'data/raw/train.csv', 'data/raw/test.csv')
        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")

if __name__ == "__main__":
    main()