import argparse
import joblib
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect a saved joblib model pipeline')
    parser.add_argument('--model', '-m', default='model.joblib', help='Path to the saved joblib model')
    return parser.parse_args()


def print_pipeline_info(pipeline):
    print('Pipeline steps:')
    for name, step in pipeline.steps:
        print(f'  - {name}: {type(step).__name__}')

    if hasattr(pipeline, 'named_steps'):
        print('\nNamed steps:')
        for name, step in pipeline.named_steps.items():
            print(f'  {name}: {type(step).__name__}')

    if 'preprocessor' in pipeline.named_steps:
        print('\nPreprocessor details:')
        preprocessor = pipeline.named_steps['preprocessor']
        if hasattr(preprocessor, 'transformers'):
            for name, transformer, cols in preprocessor.transformers:
                print(f'  transformer: {name}')
                print(f'    type: {type(transformer).__name__}')
                print(f'    columns: {cols}')
                if hasattr(transformer, 'named_steps'):
                    for sub_name, sub_step in transformer.named_steps.items():
                        print(f'      sub-step: {sub_name} ({type(sub_step).__name__})')

    if 'model' in pipeline.named_steps:
        model = pipeline.named_steps['model']
        print('\nModel details:')
        print(f'  type: {type(model).__name__}')
        try:
            print(f'  params: {model.get_params()}')
        except Exception as exc:
            print(f'  unable to show params: {exc}')


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model file not found: {args.model}')

    pipeline = joblib.load(args.model)
    print(f'Loaded model from: {args.model}')
    print_pipeline_info(pipeline)


if __name__ == '__main__':
    main()
