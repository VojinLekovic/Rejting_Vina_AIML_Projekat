import argparse
import joblib
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

DEFAULT_MODEL_PATH = 'model.joblib'


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect feature importances of a saved wine model')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL_PATH, help='Path to the saved joblib model')
    parser.add_argument('--top', '-t', type=int, default=20, help='Number of top features to print')
    parser.add_argument('--output', '-o', help='Optional CSV file to save feature importances')
    return parser.parse_args()


def get_feature_names(preprocessor):
    try:
        feature_names = preprocessor.get_feature_names_out()
        return feature_names.tolist()
    except Exception:
        # Fallback for older sklearn versions
        feature_names = []
        if hasattr(preprocessor, 'transformers'):
            for name, transformer, cols in preprocessor.transformers:
                if transformer == 'drop' or transformer == 'passthrough':
                    continue
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        names = transformer.get_feature_names_out(cols)
                        feature_names.extend(names)
                        continue
                    except Exception:
                        pass
                if hasattr(transformer, 'named_steps'):
                    if 'onehot' in transformer.named_steps:
                        this_names = transformer.named_steps['onehot'].get_feature_names_out(cols)
                        feature_names.extend(this_names)
                        continue
                if isinstance(cols, list):
                    feature_names.extend(cols)
                else:
                    feature_names.append(str(cols))
        return feature_names


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model file not found: {args.model}')

    pipeline = joblib.load(args.model)
    if not hasattr(pipeline, 'named_steps') or 'model' not in pipeline.named_steps:
        raise ValueError('Saved object is not a scikit-learn Pipeline with a model step named "model".')

    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps.get('preprocessor')
    if preprocessor is None:
        raise ValueError('Pipeline does not contain a preprocessor step named "preprocessor".')

    if not hasattr(model, 'feature_importances_'):
        raise ValueError('Model does not expose feature_importances_. Try using a tree-based regressor.')

    feature_names = get_feature_names(preprocessor)
    importances = model.feature_importances_

    if len(feature_names) != len(importances):
        logging.warning('Number of feature names (%d) does not match importances (%d). Using generic names.', len(feature_names), len(importances))
        feature_names = [f'feature_{i}' for i in range(len(importances))]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    logging.info('Top %d features by importance:', args.top)
    print(importance_df.head(args.top).to_string(index=False))

    if args.output:
        importance_df.to_csv(args.output, index=False)
        logging.info('Feature importances saved to %s', args.output)


if __name__ == '__main__':
    main()
