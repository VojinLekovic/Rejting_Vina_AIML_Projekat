import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

DEFAULT_DATA_PATH = 'winemag-data-130k-v2.csv'


def load_data(path, sample_size=0):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset not found: {path}')

    try:
        df = pd.read_csv(path, index_col=0, low_memory=False)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
    except Exception:
        logging.warning('Failed reading with index_col=0, retrying without it')
        df = pd.read_csv(path, low_memory=False)

    if sample_size and sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        logging.info('Using sample of %d records for EDA', len(df))

    return df


def summarize_data(df):
    print('Dataset shape:', df.shape)
    print('\nColumns:')
    print(df.columns.tolist())
    print('\nFirst 10 rows:')
    print(df.head(10).to_string(index=False))
    print('\nLast 5 rows:')
    print(df.tail(5).to_string(index=False))
    print('\nMissing values per column:')
    print(df.isna().sum())
    print('\nBasic numeric statistics:')
    print(df.describe(include='all'))


def plot_histograms(df, output_dir):
    numeric_cols = ['price', 'points']
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(12, 4))
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color='skyblue')
        ax.set_title(f'Histogram of {col}')
    plt.tight_layout()
    path = os.path.join(output_dir, 'histograms.png')
    plt.savefig(path)
    plt.close(fig)
    logging.info('Saved histograms to %s', path)


def plot_correlation(df, output_dir):
    numeric_cols = ['price', 'points']
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation matrix for numeric attributes')
    path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(path)
    plt.close(fig)
    logging.info('Saved correlation matrix to %s', path)


def plot_relationships(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='price', y='points', hue='country', palette='tab10', alpha=0.7, edgecolor=None)
    ax.set_title('Price vs Points by Country')
    ax.set_xlabel('Price')
    ax.set_ylabel('Points')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1)
    path1 = os.path.join(output_dir, 'price_vs_points_by_country.png')
    plt.savefig(path1, bbox_inches='tight')
    plt.close(fig)
    logging.info('Saved scatter plot to %s', path1)

    top_varieties = df['variety'].value_counts().nlargest(8).index.tolist()
    subset = df[df['variety'].isin(top_varieties)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=subset, x='variety', y='points', ax=ax)
    ax.set_title('Points distribution for top varieties')
    ax.set_xlabel('Variety')
    ax.set_ylabel('Points')
    plt.xticks(rotation=45, ha='right')
    path2 = os.path.join(output_dir, 'points_by_top_varieties.png')
    plt.savefig(path2, bbox_inches='tight')
    plt.close(fig)
    logging.info('Saved boxplot to %s', path2)


def prepare_data(df):
    cols = ['points', 'price', 'country', 'province', 'variety', 'winery', 'description']
    df = df[cols].copy()
    df = df.dropna(subset=['points'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['desc_len'] = df['description'].fillna('').str.len()
    df['country'] = df['country'].fillna('Missing')
    df['province'] = df['province'].fillna('Missing')
    df['variety'] = df['variety'].fillna('Missing')
    df['winery'] = df['winery'].fillna('Missing')
    logging.info('After cleaning, dataset shape: %s', df.shape)
    return df


def split_data(df, output_dir=None):
    df['points_bin'] = pd.cut(df['points'], bins=[0, 85, 88, 90, 92, 100], labels=False, include_lowest=True)
    X = df.drop(columns=['points', 'description', 'points_bin'])
    y = df['points'].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['points_bin'])
    logging.info('Completed stratified split: training %d samples, test %d samples', X_train.shape[0], X_test.shape[0])
    if output_dir:
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
        logging.info('Saved train and test splits to %s', output_dir)
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description='Explore wine dataset and prepare it for machine learning')
    parser.add_argument('--data', '-d', default=DEFAULT_DATA_PATH, help='Path to the wine dataset CSV')
    parser.add_argument('--sample', '-s', type=int, default=0, help='Sample a smaller number of rows for quick execution')
    parser.add_argument('--output', '-o', default='eda_output', help='Directory to save plots and split files')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = load_data(args.data, sample_size=args.sample)
    summarize_data(df)
    plot_histograms(df, args.output)
    plot_correlation(df, args.output)
    plot_relationships(df, args.output)
    prepared = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(prepared, output_dir=args.output)

    print('\nSample of prepared predictors:')
    print(X_train.head(5).to_string(index=False))
    print('\nSample of target values:')
    print(y_train.head(5).to_string(index=False))

    logging.info('EDA and preparation complete. Output saved to %s', args.output)


if __name__ == '__main__':
    main()
