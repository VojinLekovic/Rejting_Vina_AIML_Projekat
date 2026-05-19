import argparse
import inspect
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet, SGDClassifier, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor, BaggingClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

DATA_PATH = 'winemag-data-130k-v2.csv'


def load_data(path=DATA_PATH, sample_size=0):
    if not os.path.exists(path):
        logging.error('Dataset not found: %s', path)
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path, index_col=0, low_memory=False)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
    except Exception:
        logging.warning('Failed reading with index_col=0, retrying without it')
        df = pd.read_csv(path, low_memory=False)

    if sample_size and sample_size > 0:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        logging.info('Using a sample of %d records for experiments', len(df))

    return df


def prepare_features(df, top_n=30):
    df = df[['points', 'price', 'country', 'province', 'variety', 'winery', 'description']].copy()
    df = df.dropna(subset=['points'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['desc_len'] = df['description'].fillna('').str.len()

    for c in ['country', 'province', 'variety', 'winery']:
        top = df[c].value_counts().nlargest(top_n).index
        df[c] = df[c].where(df[c].isin(top), other='Other')
        df[c] = df[c].fillna('Missing')

    X = df.drop(columns=['points', 'description'])
    y = df['points'].astype(float)
    return X, y


def make_class_labels(y):
    bins = [0, 85, 89, 93, 101]
    labels = ['low', 'medium', 'high', 'premium']
    y_class = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
    return y_class.astype(str)


def build_preprocessor(numeric_features, categorical_features, polynomial=False):
    if polynomial:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    ohe_kwargs = {'handle_unknown': 'ignore'}
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
    return preprocessor


def get_regression_models():
    return {
        'LinearRegression': LinearRegression(),
        'PolynomialRegression': LinearRegression(),
        'SGDRegressor': SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'SVR_linear': SVR(kernel='linear', C=1.0),
        'SVR_poly': SVR(kernel='poly', degree=2, C=1.0, gamma='scale'),
        'SVR_rbf': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'BaggingTree': BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42), n_estimators=10, bootstrap=False, random_state=42),
    }


def get_classification_models():
    return {
        'SGDClassifier': SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=2000, solver='liblinear', random_state=42),
        'SVC_linear': SVC(kernel='linear', probability=True, random_state=42),
        'SVC_poly': SVC(kernel='poly', degree=2, probability=True, random_state=42),
        'SVC_rbf': SVC(kernel='rbf', probability=True, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'BaggingTree': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=10, bootstrap=False, random_state=42),
    }


def evaluate_regression(models, X_train, X_test, y_train, y_test, output_dir):
    results = []
    for name, estimator in models.items():
        logging.info('Training regression model: %s', name)
        poly = name == 'PolynomialRegression'
        preprocessor = build_preprocessor(['price', 'desc_len'], ['country', 'province', 'variety', 'winery'], polynomial=poly)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', estimator)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append({'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    df = pd.DataFrame(results).sort_values('RMSE')
    df.to_csv(os.path.join(output_dir, 'regression_metrics.csv'), index=False)
    logging.info('Saved regression metrics to %s', os.path.join(output_dir, 'regression_metrics.csv'))
    return df


def evaluate_classification(models, X_train, X_test, y_train, y_test, output_dir):
    results = []
    for name, estimator in models.items():
        logging.info('Training classification model: %s', name)
        preprocessor = build_preprocessor(['price', 'desc_len'], ['country', 'province', 'variety', 'winery'])
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', estimator)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='macro', zero_division=0)
        recall = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        results.append({'model': name, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})
    df = pd.DataFrame(results).sort_values('f1', ascending=False)
    df.to_csv(os.path.join(output_dir, 'classification_metrics.csv'), index=False)
    logging.info('Saved classification metrics to %s', os.path.join(output_dir, 'classification_metrics.csv'))
    return df


def plot_learning_curve_for_model(name, estimator, X, y, output_dir, scoring='neg_mean_squared_error'):
    logging.info('Plotting learning curve for model: %s', name)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
        random_state=42,
    )
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training RMSE')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Validation RMSE')
    plt.title(f'Learning curve: {name}')
    plt.xlabel('Training examples')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    path = os.path.join(output_dir, f'learning_curve_{name}.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logging.info('Saved learning curve to %s', path)


def plot_confusion_matrix_and_curves(model_name, pipeline, X_test, y_test, output_dir):
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f'Confusion Matrix: {model_name}')
    path_cm = os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(path_cm, bbox_inches='tight')
    plt.close(fig)
    logging.info('Saved confusion matrix to %s', path_cm)

    if hasattr(pipeline, 'predict_proba') or hasattr(pipeline.named_steps['model'], 'decision_function'):
        y_bin = LabelBinarizer().fit_transform(y_test)
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        if hasattr(pipeline, 'predict_proba'):
            y_score = pipeline.predict_proba(X_test)
        else:
            decision = pipeline.decision_function(X_test)
            if decision.ndim == 1:
                y_score = np.vstack([1 - decision, decision]).T
            else:
                y_score = decision

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, class_name in enumerate(np.unique(y_test)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            ax.plot(fpr, tpr, label=f'{class_name}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f'ROC Curve: {model_name}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='best')
        path_roc = os.path.join(output_dir, f'roc_curve_{model_name}.png')
        plt.savefig(path_roc, bbox_inches='tight')
        plt.close(fig)
        logging.info('Saved ROC curve to %s', path_roc)

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, class_name in enumerate(np.unique(y_test)):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
            ax.plot(recall, precision, label=f'{class_name}')
        ax.set_title(f'Precision-Recall Curve: {model_name}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='best')
        path_pr = os.path.join(output_dir, f'pr_curve_{model_name}.png')
        plt.savefig(path_pr, bbox_inches='tight')
        plt.close(fig)
        logging.info('Saved PR curve to %s', path_pr)


def main():
    parser = argparse.ArgumentParser(description='Train and compare regression and classification models for wine ratings')
    parser.add_argument('--data', '-d', default=DATA_PATH, help='Path to dataset CSV')
    parser.add_argument('--sample', '-s', type=int, default=0, help='Use a random subset of this many rows')
    parser.add_argument('--output', '-o', default='ml_experiments_output', help='Directory to save results and plots')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = load_data(args.data, sample_size=args.sample)
    logging.info('Loaded dataset with %d rows and %d columns', df.shape[0], df.shape[1])

    X, y = prepare_features(df)
    y_class = make_class_labels(y)
    logging.info('Generated %d classification labels from numeric points', len(y_class))

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
    logging.info('Regression split: %d train, %d test', X_reg_train.shape[0], X_reg_test.shape[0])
    logging.info('Classification split: %d train, %d test', X_clf_train.shape[0], X_clf_test.shape[0])

    reg_models = get_regression_models()
    clf_models = get_classification_models()

    regression_metrics = evaluate_regression(reg_models, X_reg_train, X_reg_test, y_reg_train, y_reg_test, args.output)
    classification_metrics = evaluate_classification(clf_models, X_clf_train, X_clf_test, y_clf_train, y_clf_test, args.output)

    regression_metrics.to_csv(os.path.join(args.output, 'regression_metrics.csv'), index=False)
    classification_metrics.to_csv(os.path.join(args.output, 'classification_metrics.csv'), index=False)

    logging.info('Top regression models by RMSE:')
    logging.info('\n%s', regression_metrics.head(5).to_string(index=False))
    logging.info('Top classification models by F1:')
    logging.info('\n%s', classification_metrics.head(5).to_string(index=False))

    # Plot learning curves for a few representative regression models
    for model_name in ['LinearRegression', 'RandomForest', 'SVR_rbf']:
        estimator = reg_models[model_name]
        pipeline = Pipeline(steps=[('preprocessor', build_preprocessor(['price', 'desc_len'], ['country', 'province', 'variety', 'winery'])), ('model', estimator)])
        plot_learning_curve_for_model(model_name, pipeline, X_reg_train, y_reg_train, args.output)

    # Use the best classifier for curves and confusion matrix
    best_clf_name = classification_metrics.iloc[0]['model']
    best_clf = clf_models[best_clf_name]
    best_pipeline = Pipeline(steps=[('preprocessor', build_preprocessor(['price', 'desc_len'], ['country', 'province', 'variety', 'winery'])), ('model', best_clf)])
    best_pipeline.fit(X_clf_train, y_clf_train)
    plot_confusion_matrix_and_curves(best_clf_name, best_pipeline, X_clf_test, y_clf_test, args.output)

    logging.info('ML experiments complete. Results saved to %s', args.output)


if __name__ == '__main__':
    main()
