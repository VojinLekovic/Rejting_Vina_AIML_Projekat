Projekt: Predikcija ocjena vina (points)

Sadržaj:

- `train.py` — skripta za učitavanje podataka, pripremu, treniranje modela i spremanje modela.
- `predict.py` — primjer kako učitati spremljeni model i predvidjeti ocjenu za jedan uzorak.
- `eda.py` — skripta za istraživanje podataka, tabelarni prikaz, vizualizaciju i pripremu.
- `ml_experiments.py` — skripta za treniranje i usporedbu regresijskih i klasifikacijskih modela.
- `validate.py` — skripta za validaciju modela na testnom skupu i dodatne metrike.
- `feature_importance.py` — izvlači i ispisuje važnost značajki iz spremljenog modela.
- `requirements.txt` — ovisnosti za virtualno okruženje.

Kako pokrenuti:

1. Postavite `winemag-data-130k-v2.csv` u isti folder kao `train.py`.
2. Kreirajte virtualno okruženje i instalirajte pakete:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Treniranje modela:

```bash
python train.py
```

Treniranje na manjem uzorku za brzu provjeru:

```bash
python train.py --sample 5000
```

4. Predikcija (primjer):

```bash
python predict.py
```

Predikcija s vlastitim JSON uzorkom (string ili putanja do fajla):

```bash
python predict.py --sample '{"price":30,"country":"US","province":"California","variety":"Chardonnay","winery":"My Winery","description":"Fresh citrus."}'
# ili
python predict.py --sample sample.json
```

Predikcija interaktivnim unosom:

```bash
python predict.py --interactive
```

5. Validacija spremljenog modela:

```bash
python validate.py
```

Validacija na manjim podacima:

```bash
python validate.py --sample 5000
```

Spremanje predikcija iz validacije:

```bash
python validate.py --output validation_predictions.csv
```

6. Istraživanje podataka i priprema:

```bash
python eda.py
```

Pokretanje EDA na manjem uzorku:

```bash
python eda.py --sample 1000
```

Zadržavanje izlaza u posebnom folderu:

```bash
python eda.py --output eda_output
```

7. Pokretanje eksperimenata s modelima:

```bash
python ml_experiments.py --sample 5000 --output ml_experiments_output
```

Ova skripta trenira i uspoređuje:

- regresijske modele: Linear, Polynomial, SGD, Ridge, Lasso, ElasticNet, SVM (linear/poly/rbf), Decision Tree, Random Forest, Bagging Tree
- klasifikacijske modele: SGD, Logistic, SVM (linear/poly/rbf), Decision Tree, Random Forest, Bagging Tree

8. Analiza važnosti značajki:

```bash
python feature_importance.py
```

Spremanje važnosti značajki u CSV:

```bash
python feature_importance.py --output feature_importances.csv
```

Bilješke:

- `train.py` koristi `RandomForestRegressor` za regresiju ocjena (`points`).
- Podaci se jednostavno obrađuju: izračun duljine opisa, smanjenje kardinaliteta kategorija (top N -> Other), skaliranje numeričkih značajki i OneHot za kategorije.
- Spremanje modela u `model.joblib`.
