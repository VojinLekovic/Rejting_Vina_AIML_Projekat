Ovaj dokument objašnjava `train.py` i `predict.py` liniju po liniju.

## train.py

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

Objašnjenje: importi uvoze potrebne biblioteke. `os` i `sys` koriste se za provjeru datoteka i izlazak.

DATA_PATH = 'winemag-data-130k-v2.csv'
MODEL_PATH = 'model.joblib'

Objašnjenje: konstante definiraju gdje se očekuje CSV i gdje će se spremiti model.

def load_data(path=DATA_PATH):
if not os.path.exists(path):
print(f"Dataset not found at {path}. Put the CSV in the project folder.")
sys.exit(1)
df = pd.read_csv(path, index_col=0)
return df

Objašnjenje: `load_data` provjerava postoji li datoteka, čita CSV u `pandas.DataFrame` i vraća ga. `index_col=0` koristi prvu kolonu kao indeks (CSV iz izvora ima indeksni stupac).

def prepare_features(df, top_n=30): # Select useful columns and drop rows missing the target
cols = ['points', 'price', 'country', 'province', 'variety', 'winery', 'description']
df = df[cols].copy()
df = df.dropna(subset=['points'])

Objašnjenje: odabiru se relevantni stupci i redovi bez ciljne varijable `points` se uklanjaju.

    # Feature: description length (simple text feature)
    df['desc_len'] = df['description'].fillna("").str.len()

Objašnjenje: iz `description` stvaramo novu numeričku značajku `desc_len` kao duljinu teksta.

    # Reduce cardinality for categorical columns by keeping top_n categories
    for c in ['country', 'province', 'variety', 'winery']:
        top = df[c].value_counts().nlargest(top_n).index
        df[c] = df[c].where(df[c].isin(top), other='Other')

Objašnjenje: za svaku kategorijsku kolonu držimo samo `top_n` najčešćih vrijednosti, ostale označimo kao `Other`. To smanjuje broj dummy varijabli nakon OneHot enkodiranja.

    # Define X and y for regression (predicting points)
    X = df.drop(columns=['points', 'description'])
    y = df['points'].astype(float)
    return X, y

Objašnjenje: `description` se uklanja iz X (jer koristimo samo `desc_len`), a `y` je ciljna varijabla konvertirana u float.

def build_pipeline(numeric_features, categorical_features):
numeric_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='median')),
('scaler', StandardScaler())
])

Objašnjenje: numeričke značajke prvo imputer (medijan) pa skaliranje.

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

Objašnjenje: za kategorije najprije zamijenimo prazne vrijednosti s 'Missing', zatim OneHot enkodiramo. `handle_unknown='ignore'` omogućava pipeline-u da primi nove vrijednosti pri predikciji bez greške.

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

Objašnjenje: `ColumnTransformer` primjenjuje različite transformacije različitim skupovima stupaca.

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

Objašnjenje: model je Random Forest s 100 stabala, reproducibilan (`random_state`) i koristi sve dostupne CPU jezgre.

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline

Objašnjenje: Pipeline spaja preprocesor i model tako da se sve transformacije i učenje mogu pozvati kroz jedan objekt.

def train_and_evaluate(X, y, pipeline):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print('Evaluation results:')
    print(f'MAE: {mae:.3f}')
    print(f'RMSE: {rmse:.3f}')
    print(f'R2: {r2:.3f}')
    return pipeline

Objašnjenje: podjela na trening/test, treniranje, dobivanje predikcija i izračun osnovnih metrika regresije.

def save_model(pipeline, path=MODEL_PATH):
joblib.dump(pipeline, path)
print(f'Model saved to {path}')

Objašnjenje: spremanje cijelog pipeline objekta (uključujući preprocesor i model) koristeći `joblib`.

def main():
df = load_data()
X, y = prepare_features(df)

    numeric_features = ['price', 'desc_len']
    categorical_features = ['country', 'province', 'variety', 'winery']

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline = train_and_evaluate(X, y, pipeline)
    save_model(pipeline)

if **name** == '**main**':
main()

---

## predict.py

import joblib
import pandas as pd
import os
import sys

MODEL_PATH = 'model.joblib'

def load_model(path=MODEL_PATH):
if not os.path.exists(path):
print(f'Model not found at {path}. Run train.py first.')
sys.exit(1)
return joblib.load(path)

Objašnjenje: učitava spremljeni pipeline, izlazi ako model ne postoji.

def predict_sample(model, sample): # sample: dict with keys price,country,province,variety,winery,description
df = pd.DataFrame([sample])
df['desc_len'] = df['description'].fillna("").str.len()
X = df.drop(columns=['description'])
preds = model.predict(X)
return preds[0]

Objašnjenje: pretvara ulazni rječnik u DataFrame, izračunava `desc_len` i uklanja `description` prije predikcije.

def main():
model = load_model() # Example sample - replace with real values or load from a file
sample = {
'price': 25.0,
'country': 'US',
'province': 'California',
'variety': 'Pinot Noir',
'winery': 'Sample Winery',
'description': 'Bright cherry and spice with fine acidity.'
}
pred = predict_sample(model, sample)
print(f'Predicted points: {pred:.2f}')

Objašnjenje: primjer poziva koji prikazuje kako dobiti predikciju iz spremljenog modela.

Kraj.
