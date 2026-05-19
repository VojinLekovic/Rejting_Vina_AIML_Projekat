import os
import sys
import argparse
import inspect
import logging
import pandas as pd
import numpy as np
import sklearn
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


DATA_PATH = 'winemag-data-130k-v2.csv'
MODEL_PATH = 'model.joblib'

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def parse_args():
    p = argparse.ArgumentParser(description='Train wine points prediction model')
    p.add_argument('--data', '-d', default=DATA_PATH, help='Path to dataset CSV')
    p.add_argument('--model', '-m', default=MODEL_PATH, help='Path to save trained model')
    p.add_argument('--sample', '-s', type=int, default=0, help='If >0, use a random sample of this many rows for fast testing')
    return p.parse_args()


def load_data(path=DATA_PATH, sample_size=0):
    if not os.path.exists(path):
        logging.error('Dataset not found at %s', path)
        raise FileNotFoundError(path)

    # Try reading with common options; some CSVs include an extraneous index column.
    try:
        df = pd.read_csv(path, index_col=0, low_memory=False)
        # If first column was not an index (e.g., name 'Unnamed: 0'), drop it
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
    except Exception:
        logging.warning('read_csv with index_col=0 failed, retrying without index_col')
        df = pd.read_csv(path, low_memory=False)

    if sample_size and sample_size > 0:
        logging.info('Sampling %d rows from data for quick testing', sample_size)
        df = df.sample(n=sample_size, random_state=42)

    return df


def prepare_features(df, top_n=30):
    # Select useful columns and drop rows missing the target
    cols = ['points', 'price', 'country', 'province', 'variety', 'winery', 'description']
    df = df[cols].copy()
    df = df.dropna(subset=['points'])

    # Feature: description length (simple text feature)
    df['desc_len'] = df['description'].fillna("").str.len()

    # Reduce cardinality for categorical columns by keeping top_n categories
    for c in ['country', 'province', 'variety', 'winery']:
        top = df[c].value_counts().nlargest(top_n).index
        df[c] = df[c].where(df[c].isin(top), other='Other')

    # Define X and y for regression (predicting points)
    X = df.drop(columns=['points', 'description'])
    y = df['points'].astype(float)
    return X, y


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # OneHotEncoder parameter name changed across sklearn versions
    # sklearn>=1.2 uses `sparse_output`, older versions use `sparse`.
    ohe_kwargs: dict[str, Any] = {'handle_unknown': 'ignore'}
    if 'sparse_output' in inspect.signature(OneHotEncoder).parameters:
        ohe_kwargs['sparse_output'] = False
    else:
        ohe_kwargs['sparse'] = False

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(**ohe_kwargs))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline


def train_and_evaluate(X, y, pipeline):
    logging.info('Dataset contains %d samples with %d features.', X.shape[0], X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info('Training set size: %d samples', X_train.shape[0])
    logging.info('Validation set size: %d samples', X_test.shape[0])
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        logging.exception('Error during pipeline.fit: %s', e)
        raise

    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    logging.info('Evaluation results:')
    logging.info('MAE: %.3f', mae)
    logging.info('RMSE: %.3f', rmse)
    logging.info('R2: %.3f', r2)
    return pipeline, X_train, X_test, y_train, y_test, preds


def save_model(pipeline, path=MODEL_PATH):
    joblib.dump(pipeline, path)
    logging.info('Model saved to %s', path)


def main():
    args = parse_args()
    df = load_data(args.data, sample_size=args.sample)
    X, y = prepare_features(df)

    numeric_features = ['price', 'desc_len']
    categorical_features = ['country', 'province', 'variety', 'winery']

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline, X_train, X_test, y_train, y_test, preds = train_and_evaluate(X, y, pipeline)
    save_model(pipeline, path=args.model)


if __name__ == '__main__':
    main()
