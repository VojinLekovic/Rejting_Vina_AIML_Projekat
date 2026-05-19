import argparse
import joblib
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error, median_absolute_error, explained_variance_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


DEFAULT_DATA_PATH = 'winemag-data-130k-v2.csv'
DEFAULT_MODEL_PATH = 'model.joblib'


def parse_args():
    parser = argparse.ArgumentParser(description='Validate a saved wine points regression model')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL_PATH, help='Path to the saved joblib model')
    parser.add_argument('--data', '-d', default=DEFAULT_DATA_PATH, help='Path to dataset CSV for validation')
    parser.add_argument('--sample', '-s', type=int, default=0, help='If >0, validate on a random subset of this many rows')
    parser.add_argument('--output', '-o', help='Path to save prediction results CSV')
    return parser.parse_args()


def load_data(path, sample_size=0):
    if not os.path.exists(path):
        logging.error('Dataset file not found: %s', path)
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path, index_col=0, low_memory=False)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
    except Exception:
        logging.warning('read_csv with index_col=0 failed, retrying without index_col')
        df = pd.read_csv(path, low_memory=False)

    if sample_size and sample_size > 0:
        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        else:
            logging.info('Requested sample size %d is larger than dataset size %d; using full dataset', sample_size, len(df))

    return df


def validate(model, df):
    if 'points' not in df.columns:
        logging.error('Dataset must contain a "points" column for validation.')
        raise ValueError('Missing target column: points')

    required_cols = ['price', 'country', 'province', 'variety', 'winery', 'description']
    for c in required_cols:
        if c not in df.columns:
            logging.error('Dataset missing required column: %s', c)
            raise ValueError(f'Missing required column: {c}')

    X = df[required_cols].copy()
    X['desc_len'] = X['description'].fillna('').str.len()
    X = X.drop(columns=['description'])
    y_true = df['points'].astype(float)

    y_pred = model.predict(X)

    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'ExplainedVariance': explained_variance_score(y_true, y_pred),
        'MaxError': max_error(y_true, y_pred),
        'MedianAE': median_absolute_error(y_true, y_pred),
    }

    return metrics, y_pred


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        logging.error('Model file not found: %s', args.model)
        raise FileNotFoundError(args.model)

    model = joblib.load(args.model)
    df = load_data(args.data, sample_size=args.sample)

    logging.info('Validating model on %d samples', len(df))
    metrics, y_pred = validate(model, df)

    logging.info('Validation metrics:')
    for name, value in metrics.items():
        logging.info('  %s: %.4f', name, value)

    if args.output:
        result_df = df.copy()
        result_df['predicted_points'] = y_pred
        result_df.to_csv(args.output, index=False)
        logging.info('Saved predictions to %s', args.output)


if __name__ == '__main__':
    main()
