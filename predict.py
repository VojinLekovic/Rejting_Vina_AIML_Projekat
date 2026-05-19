import argparse
import json
import joblib
import pandas as pd
import numpy as np
import os
import sys
import logging

MODEL_PATH = 'model.joblib'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        logging.error('Model not found at %s. Run train.py first.', path)
        raise FileNotFoundError(path)
    return joblib.load(path)


def predict_sample(model, sample):
    # sample: dict with keys price,country,province,variety,winery,description
    # Ensure required fields exist; fill missing with sensible defaults
    required = ['price', 'country', 'province', 'variety', 'winery', 'description']
    for r in required:
        if r not in sample:
            sample[r] = '' if r == 'description' else np.nan

    df = pd.DataFrame([sample])
    df['desc_len'] = df['description'].fillna("").str.len()
    X = df.drop(columns=['description'])
    preds = model.predict(X)
    return preds[0]


def prompt_for_input():
    print('Enter wine sample values for prediction:')
    price = input('Price (e.g. 25.0): ').strip()
    country = input('Country (e.g. US): ').strip() or 'US'
    province = input('Province (e.g. California): ').strip() or 'California'
    variety = input('Variety (e.g. Pinot Noir): ').strip() or 'Pinot Noir'
    winery = input('Winery (e.g. Sample Winery): ').strip() or 'Sample Winery'
    description = input('Description (short text): ').strip() or 'Bright cherry and spice with fine acidity.'

    try:
        price_value = float(price)
    except ValueError:
        raise ValueError('Price must be a number, for example 25.0')

    return {
        'price': price_value,
        'country': country,
        'province': province,
        'variety': variety,
        'winery': winery,
        'description': description
    }


def parse_args():
    p = argparse.ArgumentParser(description='Predict wine points using saved model')
    p.add_argument('--model', '-m', default=MODEL_PATH, help='Path to saved model')
    p.add_argument('--sample', '-s', help='JSON string or path to JSON file with sample data')
    p.add_argument('--interactive', '-i', action='store_true', help='Prompt for manual input values')
    return p.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model)
    use_interactive = args.interactive or (not args.sample and sys.stdin.isatty())

    if use_interactive:
        sample = prompt_for_input()
    elif args.sample:
        # try to load JSON file or parse JSON string
        if os.path.exists(args.sample):
            with open(args.sample, 'r', encoding='utf-8') as f:
                sample = json.load(f)
        else:
            sample = json.loads(args.sample)
    else:
        sample = {
            'price': 25.0,
            'country': 'US',
            'province': 'California',
            'variety': 'Pinot Noir',
            'winery': 'Sample Winery',
            'description': 'Bright cherry and spice with fine acidity.'
        }

    pred = predict_sample(model, sample)
    logging.info('Predicted points: %.2f', pred)


if __name__ == '__main__':
    main()
