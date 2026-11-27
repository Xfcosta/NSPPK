"""Bayesian hyperparameter tuning for NSPPK prediction wrappers."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from inspect import Parameter, signature
from numbers import Integral, Real
from typing import Callable

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_consistent_length


_DEFAULT_SEARCH_SPACE = {
    "radius": (0, 2),
    "distance": (0, 4),
    "connector": (0, 2),
    "use_simplex_features": [False, True],
}
_TUNABLE_ENCODER_PARAMETERS = {
    "radius",
    "distance",
    "connector",
    "nbits",
    "degree_threshold",
    "weight_key",
    "node_attribute_key",
    "attribute_dim",
    "attribute_alphabet_size",
    "edge_attribute_key",
    "edge_attribute_dim",
    "edge_attribute_alphabet_size",
    "use_node_kernel",
    "sigma",
    "use_simplex_features",
    "simplex_feature_order",
}
_ALIASES = {
    "r": "radius",
    "d": "distance",
    "c": "connector",
    "simplex_feature_max_order": "simplex_feature_order",
}


def _load_skopt():
    try:
        from skopt import gp_minimize
        from skopt.space import Categorical, Integer, Real as RealDimension
    except ImportError as exc:
        raise ImportError(
            "NSPPK .tune() requires scikit-optimize. Install it with "
            "`python -m pip install 'nsppk[tuning]'`."
        ) from exc
    return gp_minimize, Integer, RealDimension, Categorical


def _make_progress_bar(verbose, total, description):
    if not verbose:
        return None
    from tqdm.auto import tqdm

    return tqdm(total=total, desc=description)


def _resolve_trial_counts(
    n_lhs_trials,
    n_additional_trials,
    n_samples,
    n_initial_points,
    *,
    default_lhs=20,
    default_additional=20,
):
    """Resolve explicit LHS/additional counts and legacy total-count aliases."""
    legacy_used = n_samples is not None or n_initial_points is not None
    explicit_used = (
        (n_lhs_trials is not None and n_lhs_trials != default_lhs)
        or (n_additional_trials is not None and n_additional_trials != default_additional)
    )
    if legacy_used and explicit_used:
        raise ValueError(
            "Use n_lhs_trials/n_additional_trials or the legacy "
            "n_samples/n_initial_points arguments, not both."
        )

    def validate(name, value, *, allow_zero=False):
        minimum = 0 if allow_zero else 1
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < minimum
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer.")
        return int(value)

    if legacy_used:
        total = validate("n_samples", 40 if n_samples is None else n_samples)
        requested_lhs = validate(
            "n_initial_points", default_lhs if n_initial_points is None else n_initial_points
        )
        lhs = min(requested_lhs, total)
        return lhs, total - lhs

    lhs = validate(
        "n_lhs_trials", default_lhs if n_lhs_trials is None else n_lhs_trials
    )
    additional = validate(
        "n_additional_trials",
        default_additional if n_additional_trials is None else n_additional_trials,
        allow_zero=True,
    )
    return lhs, additional


def _normalize_search_space(estimator, search_space):
    if search_space is None:
        search_space = _DEFAULT_SEARCH_SPACE
    if not isinstance(search_space, Mapping) or not search_space:
        raise ValueError("search_space must be a non-empty mapping of parameter ranges.")

    params = estimator.get_params(deep=False)
    normalized = {}
    for name, specification in search_space.items():
        canonical = _ALIASES.get(name, name)
        if canonical in normalized:
            raise ValueError(f"Search space specifies {canonical!r} more than once.")
        if canonical not in _TUNABLE_ENCODER_PARAMETERS or canonical not in params:
            raise ValueError(
                f"Unknown or non-encoder parameter in search_space: {name!r}. "
                "Only NSPPK encoder settings can be tuned."
            )
        normalized[canonical] = specification
    return normalized


def tune_entity_predictor(
    estimator,
    X,
    y,
    *,
    search_space=None,
    scoring=None,
    cv=5,
    n_lhs_trials=20,
    n_additional_trials=20,
    n_samples=None,
    n_initial_points=None,
    random_state=0,
    trial_callback=None,
    verbose=False,
    entity_indices=None,
    validation_data=None,
    validation_entity_indices=None,
):
    """Tune a node/edge wrapper by CV or an explicit validation set."""
    gp_minimize, Integer, RealDimension, Categorical = _load_skopt()

    if isinstance(cv, (bool, np.bool_)) or not isinstance(cv, Integral) or cv < 2:
        raise ValueError("cv must be an integer of at least 2.")
    lhs_count, additional_count = _resolve_trial_counts(
        n_lhs_trials, n_additional_trials, n_samples, n_initial_points
    )
    total_trials = lhs_count + additional_count
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, Integral)
    ):
        raise ValueError("random_state must be an integer or None.")
    if not isinstance(verbose, (bool, np.bool_)):
        raise ValueError("verbose must be a boolean.")
    if trial_callback is not None and not callable(trial_callback):
        raise ValueError("trial_callback must be callable or None.")

    graphs = list(X)
    if not graphs:
        raise ValueError("X must contain at least one graph.")
    train_selectors = estimator._normalize_entity_indices(graphs, entity_indices)
    selected_rows = train_selectors if train_selectors is not None else [None] * len(graphs)
    entity_counts = [
        len(selector)
        if train_selectors is not None
        else int(
            graph.number_of_nodes()
            if estimator._entity_name == "node"
            else graph.number_of_edges()
        )
        for graph, selector in zip(graphs, selected_rows)
    ]
    targets, target_layout = estimator._normalize_targets(y, entity_counts)
    if sum(entity_counts) == 0:
        raise ValueError(f"X contains no {estimator._entity_name} feature rows.")
    estimator._validate_targets(targets)

    validation_graphs = validation_targets = validation_selectors = None
    validation_counts = None
    if validation_data is not None:
        if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
            raise ValueError("validation_data must be a (graphs, targets) pair.")
        validation_graphs = list(validation_data[0])
        if not validation_graphs:
            raise ValueError("validation_data must contain at least one graph.")
        validation_selectors = estimator._normalize_entity_indices(
            validation_graphs, validation_entity_indices
        )
        if validation_selectors is None:
            validation_counts = [
                graph.number_of_nodes()
                if estimator._entity_name == "node"
                else graph.number_of_edges()
                for graph in validation_graphs
            ]
        else:
            validation_counts = [len(selector) for selector in validation_selectors]
        validation_targets, _ = estimator._normalize_targets(
            validation_data[1], validation_counts
        )
        if not len(validation_targets):
            raise ValueError("validation_data contains no selected entities.")
        estimator._validate_targets(validation_targets)

    is_classifier = estimator._task == "classifier"
    if is_classifier:
        classes, counts = np.unique(targets, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Classification tuning requires at least two classes.")
        if validation_data is None and counts.min() < cv:
            raise ValueError(
                f"Stratified {cv}-fold tuning requires at least {cv} examples in every class."
            )
        cv_splitter = (
            StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=random_state)
            if validation_data is None
            else None
        )
        if validation_data is not None and np.unique(validation_targets).size < 2:
            raise ValueError("Validation classification targets must contain at least two classes.")
        if scoring is None:
            score_name = (
                "average_precision"
                if len(classes) == 2 and np.array_equal(classes, np.asarray([0, 1]))
                else "accuracy"
            )
        else:
            score_name = scoring
    else:
        if targets.dtype.kind not in "iuf" or not np.isfinite(targets).all():
            raise ValueError("Regression tuning requires finite numeric targets.")
        if validation_data is None and len(targets) < cv:
            raise ValueError(f"KFold tuning requires at least {cv} entities.")
        cv_splitter = (
            KFold(n_splits=int(cv), shuffle=True, random_state=random_state)
            if validation_data is None
            else None
        )
        if validation_data is not None and (
            validation_targets.dtype.kind not in "iuf"
            or not np.isfinite(validation_targets).all()
        ):
            raise ValueError("Validation regression targets must be finite numeric values.")
        score_name = "r2" if scoring is None else scoring

    if not isinstance(score_name, str) and not callable(score_name):
        raise ValueError("scoring must be a scikit-learn scorer name or scorer callable.")
    scorer = get_scorer(score_name) if isinstance(score_name, str) else score_name
    search_space = _normalize_search_space(estimator, search_space)
    names, dimensions = _dimensions(
        estimator, search_space, Integer, RealDimension, Categorical
    )
    folds = (
        list(cv_splitter.split(np.zeros(len(targets)), targets))
        if validation_data is None
        else None
    )
    scores = []
    fold_scores_by_trial = []
    best_score_so_far = -np.inf
    best_params_so_far = None

    def objective(values):
        nonlocal best_score_so_far, best_params_so_far
        parameters = dict(zip(names, values))
        candidate = clone(estimator).set_params(**parameters)
        encoder, features, counts = candidate._fit_transform_graphs(
            graphs, train_selectors
        )
        if counts != entity_counts:
            raise RuntimeError("Encoder settings changed the number of entity feature rows.")
        fold_scores = []
        if validation_data is not None:
            valid_matrices = candidate._features_per_graph(
                encoder, validation_graphs, validation_selectors
            )
            valid_features = candidate._stack_entity_features(valid_matrices)
            model = candidate._make_estimator()
            model.fit(features, targets)
            fold_scores.append(float(scorer(model, valid_features, validation_targets)))
        else:
            for train_indices, valid_indices in folds:
                model = candidate._make_estimator()
                model.fit(features[train_indices], targets[train_indices])
                fold_scores.append(
                    float(scorer(model, features[valid_indices], targets[valid_indices]))
                )
        score = float(np.mean(fold_scores))
        scores.append(score)
        fold_scores_by_trial.append(fold_scores)
        is_best = score > best_score_so_far
        if is_best:
            best_score_so_far = score
            best_params_so_far = parameters.copy()
        if trial_callback is not None:
            trial_callback(
                {
                    "trial": len(scores),
                    "parameters": parameters.copy(),
                    "fold_scores": list(fold_scores),
                    "score": score,
                    "scoring": score_name if isinstance(score_name, str) else getattr(score_name, "__name__", "callable"),
                    "best_score": float(best_score_so_far),
                    "best_params": best_params_so_far.copy(),
                    "is_best": is_best,
                }
            )
        if progress is not None:
            progress.update(1)
            progress.set_postfix(score=f"{score:.4f}", best=f"{best_score_so_far:.4f}")
        return -score

    initial_count = lhs_count
    progress = _make_progress_bar(verbose, total_trials, "Tuning entity NSPPK")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module=r"sklearn\.gaussian_process"
            )
            for message in (
                r"divide by zero encountered in matmul",
                r"overflow encountered in matmul",
                r"invalid value encountered in matmul",
            ):
                warnings.filterwarnings(
                    "ignore",
                    message=message,
                    category=RuntimeWarning,
                    module=r"sklearn\.gaussian_process\._gpr",
                )
            warnings.filterwarnings(
                "ignore",
                message=r"The objective has been evaluated at point .* before, using random point .*",
                category=UserWarning,
                module=r"skopt\.optimizer\.optimizer",
            )
            result = gp_minimize(
                objective,
                dimensions,
                n_calls=total_trials,
                n_initial_points=initial_count,
                initial_point_generator="lhs",
                random_state=random_state,
                verbose=False,
            )
    finally:
        if progress is not None:
            progress.close()

    best_index = int(np.argmin(result.func_vals))
    best_parameters = dict(zip(names, result.x))
    estimator.set_params(**best_parameters)
    estimator.fit(graphs, y, entity_indices=train_selectors, verbose=verbose)
    result.task = estimator._task
    result.best_params = best_parameters
    result.best_score = float(scores[best_index])
    result.scoring = score_name if isinstance(score_name, str) else getattr(score_name, "__name__", "callable")
    result.cv = int(cv)
    result.n_samples = total_trials
    result.n_total_trials = total_trials
    result.n_lhs_trials = lhs_count
    result.n_additional_trials = additional_count
    result.n_initial_points = initial_count
    result.search_space = search_space
    result.cv_scores = np.asarray(scores, dtype=float)
    result.fold_scores = np.asarray(fold_scores_by_trial, dtype=float)
    result.target_layout = target_layout
    result.validation_data_used = validation_data is not None
    estimator.best_params_ = best_parameters.copy()
    estimator.best_score_ = result.best_score
    estimator.tune_result_ = result
    return result


def _dimensions(estimator, search_space, Integer, RealDimension, Categorical):
    names = list(search_space)
    constructor_parameters = signature(type(estimator).__init__).parameters
    dimensions = []
    for name, specification in search_space.items():
        if isinstance(specification, tuple):
            if len(specification) != 2 or not all(
                isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                for value in specification
            ):
                raise ValueError(
                    f"Tuple bounds for {name!r} must contain exactly two numeric values."
                )
            low, high = specification
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError(f"Bounds for {name!r} must be finite and increasing.")

            current = estimator.get_params(deep=False)[name]
            parameter = constructor_parameters.get(name)
            default = Parameter.empty if parameter is None else parameter.default
            type_reference = default if default is not Parameter.empty and default is not None else current
            if isinstance(type_reference, (bool, np.bool_)):
                raise ValueError(
                    f"Boolean parameter {name!r} must use an explicit categorical list."
                )
            if isinstance(type_reference, (Integral, np.integer)) or (
                type_reference is None
                and all(isinstance(value, Integral) for value in specification)
            ):
                if int(low) != low or int(high) != high:
                    raise ValueError(f"Integer parameter {name!r} requires integer bounds.")
                dimensions.append(Integer(int(low), int(high), name=name))
            elif isinstance(type_reference, (Real, np.floating)) or (
                type_reference is None
                and all(isinstance(value, Real) for value in specification)
            ):
                dimensions.append(RealDimension(float(low), float(high), name=name))
            else:
                raise ValueError(
                    f"Numeric bounds for {name!r} require an integer or real parameter."
                )
        elif isinstance(specification, list) and specification:
            dimensions.append(Categorical(specification, name=name))
        else:
            raise ValueError(
                f"Range for {name!r} must be a two-number tuple or a non-empty list."
            )
    return names, dimensions


def tune_predictor(
    estimator,
    X,
    y,
    *,
    search_space=None,
    scoring=None,
    cv=5,
    n_lhs_trials=20,
    n_additional_trials=20,
    n_samples=None,
    n_initial_points=None,
    random_state=0,
    trial_callback: Callable[[dict], None] | None = None,
    verbose=False,
    validation_data=None,
):
    """Tune only NSPPK encoder settings, then refit the winning wrapper."""
    gp_minimize, Integer, RealDimension, Categorical = _load_skopt()

    if isinstance(cv, (bool, np.bool_)) or not isinstance(cv, Integral) or cv < 2:
        raise ValueError("cv must be an integer of at least 2.")
    lhs_count, additional_count = _resolve_trial_counts(
        n_lhs_trials, n_additional_trials, n_samples, n_initial_points
    )
    total_trials = lhs_count + additional_count
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, Integral)
    ):
        raise ValueError("random_state must be an integer or None.")
    if not isinstance(verbose, (bool, np.bool_)):
        raise ValueError("verbose must be a boolean.")
    if trial_callback is not None and not callable(trial_callback):
        raise ValueError("trial_callback must be callable or None.")

    graphs = list(X)
    if not graphs:
        raise ValueError("X must contain at least one graph.")
    if y is None:
        raise ValueError("y is required.")
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional target array.")
    check_consistent_length(graphs, y)

    validation_graphs = validation_targets = None
    if validation_data is not None:
        if not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2:
            raise ValueError("validation_data must be a (graphs, targets) pair.")
        validation_graphs = list(validation_data[0])
        validation_targets = np.asarray(validation_data[1])
        if not validation_graphs:
            raise ValueError("validation_data must contain at least one graph.")
        if validation_targets.ndim != 1:
            raise ValueError("Validation targets must be a one-dimensional array.")
        check_consistent_length(validation_graphs, validation_targets)

    is_classifier = estimator._task == "classifier"
    if is_classifier:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Classification tuning requires at least two classes.")
        if validation_data is None and counts.min() < cv:
            raise ValueError(
                f"Stratified {cv}-fold tuning requires at least {cv} examples in every class."
            )
        cv_splitter = (
            StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=random_state)
            if validation_data is None
            else None
        )
        if validation_data is not None and np.unique(validation_targets).size < 2:
            raise ValueError("Validation classification targets must contain at least two classes.")
        score_name = "average_precision" if scoring is None else scoring
    else:
        if y.dtype.kind not in "iuf" or not np.isfinite(y).all():
            raise ValueError("Regression tuning requires finite numeric targets.")
        if validation_data is None and len(y) < cv:
            raise ValueError(f"KFold tuning requires at least {cv} examples.")
        cv_splitter = (
            KFold(n_splits=int(cv), shuffle=True, random_state=random_state)
            if validation_data is None
            else None
        )
        if validation_data is not None and (
            validation_targets.dtype.kind not in "iuf"
            or not np.isfinite(validation_targets).all()
        ):
            raise ValueError("Validation regression targets must be finite numeric values.")
        score_name = "r2" if scoring is None else scoring

    if not isinstance(score_name, str) and not callable(score_name):
        raise ValueError("scoring must be a scikit-learn scorer name or scorer callable.")
    scorer = get_scorer(score_name) if isinstance(score_name, str) else score_name
    search_space = _normalize_search_space(estimator, search_space)
    names, dimensions = _dimensions(
        estimator, search_space, Integer, RealDimension, Categorical
    )
    folds = (
        list(cv_splitter.split(np.zeros(len(y)), y))
        if validation_data is None
        else None
    )
    scores = []
    fold_scores_by_trial = []
    best_score_so_far = -np.inf
    best_params_so_far = None

    learned_parameters = (
        "attribute_dim",
        "attribute_alphabet_size",
        "edge_attribute_dim",
        "edge_attribute_alphabet_size",
    )
    learned_preprocessing = any(
        getattr(estimator, name, None) is not None
        for name in learned_parameters
    )
    for name in learned_parameters:
        specification = search_space.get(name)
        if isinstance(specification, tuple) and specification[1] > 0:
            learned_preprocessing = True
        elif isinstance(specification, list) and any(value is not None for value in specification):
            learned_preprocessing = True

    def objective(values):
        nonlocal best_score_so_far, best_params_so_far
        parameters = dict(zip(names, values))
        fold_scores = []

        if validation_data is not None:
            candidate = clone(estimator).set_params(**parameters)
            candidate.fit(graphs, y)
            fold_scores.append(
                float(scorer(candidate, validation_graphs, validation_targets))
            )
        elif not learned_preprocessing:
            candidate = clone(estimator).set_params(**parameters)
            encoder = candidate._make_encoder()
            encoder.fit(graphs)
            features = encoder.transform(graphs)
            for train_indices, valid_indices in folds:
                model = candidate._make_estimator()
                model.fit(features[train_indices], y[train_indices])
                fold_scores.append(
                    float(scorer(model, features[valid_indices], y[valid_indices]))
                )
        else:
            for train_indices, valid_indices in folds:
                candidate = clone(estimator).set_params(**parameters)
                train_graphs = [graphs[index] for index in train_indices]
                valid_graphs = [graphs[index] for index in valid_indices]
                candidate.fit(train_graphs, y[train_indices])
                fold_scores.append(
                    float(scorer(candidate, valid_graphs, y[valid_indices]))
                )

        score = float(np.mean(fold_scores))
        scores.append(score)
        fold_scores_by_trial.append(fold_scores)
        is_best = score > best_score_so_far
        if is_best:
            best_score_so_far = score
            best_params_so_far = parameters.copy()
        if trial_callback is not None:
            trial_callback(
                {
                    "trial": len(scores),
                    "parameters": parameters.copy(),
                    "fold_scores": list(fold_scores),
                    "score": score,
                    "scoring": score_name if isinstance(score_name, str) else getattr(score_name, "__name__", "callable"),
                    "best_score": float(best_score_so_far),
                    "best_params": best_params_so_far.copy(),
                    "is_best": is_best,
                }
            )
        if progress is not None:
            progress.update(1)
            progress.set_postfix(score=f"{score:.4f}", best=f"{best_score_so_far:.4f}")
        return -score

    initial_count = lhs_count
    progress = _make_progress_bar(verbose, total_trials, "Tuning NSPPK")
    try:
        with warnings.catch_warnings():
            # Keep warnings local to the BO surrogate fit.
            warnings.filterwarnings(
                "ignore",
                category=ConvergenceWarning,
                module=r"sklearn\.gaussian_process",
            )
            for message in (
                r"divide by zero encountered in matmul",
                r"overflow encountered in matmul",
                r"invalid value encountered in matmul",
            ):
                warnings.filterwarnings(
                    "ignore",
                    message=message,
                    category=RuntimeWarning,
                    module=r"sklearn\.gaussian_process\._gpr",
                )
            warnings.filterwarnings(
                "ignore",
                message=r"The objective has been evaluated at point .* before, using random point .*",
                category=UserWarning,
                module=r"skopt\.optimizer\.optimizer",
            )
            result = gp_minimize(
                objective,
                dimensions,
                n_calls=total_trials,
                n_initial_points=initial_count,
                initial_point_generator="lhs",
                random_state=random_state,
                verbose=False,
            )
    finally:
        if progress is not None:
            progress.close()

    best_index = int(np.argmin(result.func_vals))
    best_parameters = dict(zip(names, result.x))
    estimator.set_params(**best_parameters)
    estimator.fit(graphs, y, verbose=verbose)

    result.task = estimator._task
    result.best_params = best_parameters
    result.best_score = float(scores[best_index])
    result.scoring = score_name if isinstance(score_name, str) else getattr(score_name, "__name__", "callable")
    result.cv = int(cv)
    result.n_samples = total_trials
    result.n_total_trials = total_trials
    result.n_lhs_trials = lhs_count
    result.n_additional_trials = additional_count
    result.n_initial_points = initial_count
    result.search_space = search_space
    result.cv_scores = np.asarray(scores, dtype=float)
    result.fold_scores = np.asarray(fold_scores_by_trial, dtype=float)
    result.validation_data_used = validation_data is not None
    estimator.best_params_ = best_parameters.copy()
    estimator.best_score_ = result.best_score
    estimator.tune_result_ = result
    return result
