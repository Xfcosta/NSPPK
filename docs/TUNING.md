# Bayesian hyperparameter tuning

`NSPPKClassifier.tune()` and `NSPPKRegressor.tune()` search NSPPK encoder
settings with Latin-hypercube initial samples followed by Gaussian-process
Bayesian optimization. Install the optional dependency first:

```bash
python -m pip install 'nsppk[tuning]'
```

Pass numeric bounds as two-number tuples and explicit categorical choices as
lists. Integer versus continuous bounds are inferred from the parameter type.
The classifier or regressor supplied when creating the wrapper stays fixed;
`tune()` searches encoder settings only.

```python
from nsppk import NSPPKClassifier

model = NSPPKClassifier(
    estimator="rf",
    estimator_params={"n_estimators": 300, "class_weight": "balanced_subsample"},
    nbits=14,
    dense=False,
    parallel=False,
)
result = model.tune(
    train_graphs,
    y_train,
    search_space={
        "radius": (0, 2),
        "distance": (0, 4),
        "connector": (0, 2),
        "use_simplex_features": [False, True],
    },
    scoring="average_precision",
    cv=5,
    n_samples=40,
    n_initial_points=20,
    random_state=42,
)

print(result.best_params, result.best_score)
predictions = model.predict(test_graphs)
```

If `search_space` is omitted, the default space searches `radius=(0, 2)`,
`distance=(0, 4)`, `connector=(0, 2)`, and
`use_simplex_features=[False, True]`. Other settings, including `nbits`, stay
at their current values. Use a custom space to search those settings.

The default score is average precision for classification and R² for
regression. `cv` defaults to five folds. Classification uses stratified folds;
regression uses shuffled folds. Fold-specific learned attribute preprocessing
is fitted only on each training fold. For structural-only configurations, the
fixed graph encoding is shared across folds to avoid redundant work.

After search, the wrapper is refitted on all graphs and targets passed to
`tune()`. Its parameters are updated to the best encoder configuration. The
returned scikit-optimize `OptimizeResult` includes `best_params`, `best_score`,
`cv_scores`, `fold_scores`, and the complete optimizer history. The wrapper
also exposes `best_params_`, `best_score_`, and `tune_result_`.

`trial_callback`, when supplied, receives a dictionary after each trial with
the parameters, fold scores, mean score, best-so-far values, and an
`is_best` flag. `verbose=True` prints trial progress.
