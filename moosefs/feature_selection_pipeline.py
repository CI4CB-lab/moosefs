from itertools import combinations
import random
from typing import Any, Optional

from joblib import Parallel, delayed, parallel_backend
import numpy as np
import pandas as pd

# tqdm is not used; keep imports minimal
from .core import Feature, ParetoAnalysis
from .metrics.stability_metrics import compute_stability_metrics
from .utils import extract_params, get_class_info


class FeatureSelectionPipeline:
    """End-to-end pipeline for ensemble feature selection.

    Orchestrates feature scoring, merging, metric evaluation, and Pareto-based
    selection across repeated runs and selector ensembles.
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        *,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        fs_methods: list,
        merging_strategy: Any,
        num_repeats: int,
        num_features_to_select: Optional[int],
        metrics: list = ["logloss", "f1_score", "accuracy"],
        task: str = "classification",
        min_group_size: int = 2,
        fill: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        bootstrap: bool = False,
        bootstrap_num_samples: int = 20,
        bootstrap_min_freq: float = 0.6,
        bootstrap_use_scores: bool = True,
        refit_on_full_data: bool = False,
        stability_mode: str = "agreement",  # "agreement", "cross_folds", or "both"
    ) -> None:
        """Initialize the pipeline.

        Args:
            data: Combined DataFrame where the last column is treated as the target.
            X: Feature DataFrame (use together with ``y`` instead of ``data``).
            y: Target Series aligned with ``X``.
            fs_methods: Feature selectors (identifiers or instances).
            merging_strategy: Merging strategy (identifier or instance).
            num_repeats: Number of repeats for the pipeline.
            num_features_to_select: Desired number of features to select.
            metrics: Metric functions (identifiers or instances).
            task: 'classification' or 'regression'.
            min_group_size: Minimum number of methods in each ensemble.
            fill: If True, enforce exact size after merging.
            random_state: Seed for reproducibility.
            n_jobs: Parallel jobs (use num_repeats when -1 or None).
            bootstrap: If True, apply bootstrap-based aggregation before merging.
            bootstrap_num_samples: Number of bootstrap draws per repeat when ``bootstrap`` is True.
            bootstrap_min_freq: Minimum selection frequency to keep a feature under bootstrap.
            bootstrap_use_scores: If True, average normalized scores across bootstraps (rank mergers).

        Raises:
            ValueError: If task is invalid or required parameters are missing.

        Note:
            Exactly one of ``data`` or the pair ``(X, y)`` must be provided.
        """

        # parameters validation
        if task not in ["classification", "regression"]:
            raise ValueError("Task must be either 'classification' or 'regression'.")
        self.X, self.y = self._validate_X_y(data=data, X=X, y=y)
        self.target_name = self.y.name
        self.data = pd.concat([self.X, self.y], axis=1)
        self.task = task
        self.num_repeats = num_repeats
        self.num_features_to_select = num_features_to_select
        self.random_state = random_state if random_state is not None else random.randint(0, 1000)
        self.n_jobs = n_jobs
        self.min_group_size = min_group_size
        self.fill = fill
        self.bootstrap = bootstrap
        self.bootstrap_num_samples = int(bootstrap_num_samples)
        self.bootstrap_min_freq = float(bootstrap_min_freq)
        self.bootstrap_use_scores = bool(bootstrap_use_scores)
        self.refit_on_full_data = refit_on_full_data
        self.stability_mode = stability_mode

        # set seed for reproducibility
        self._set_seed(self.random_state)

        # Keep original specs and also instantiate now for introspection
        self._fs_method_specs = list(fs_methods)
        self._metric_specs = list(metrics)
        self._merging_specs = (
            list(merging_strategy) if isinstance(merging_strategy, (list, tuple)) else [merging_strategy]
        )

        # dynamically load classes or instantiate them (initial instances)
        self.fs_methods = [self._load_class(m, instantiate=True) for m in self._fs_method_specs]
        self.metrics = [self._load_class(m, instantiate=True) for m in self._metric_specs]
        self.merging_strategies = [self._load_class(m, instantiate=True) for m in self._merging_specs]
        # Backwards-compatible alias when a single merger is used
        self.merging_strategy = self.merging_strategies[0]
        self._multiple_mergers = len(self.merging_strategies) > 1

        # validate and preparation
        if self.num_features_to_select is None:
            raise ValueError("num_features_to_select must be provided")
        # ensemble selector combinations are generated in run() after instantiation
        self.selector_ensembles = []
        self.ensembles = []
        self.ensemble_lookup = {}

    @staticmethod
    def _set_seed(seed):
        """Seed numpy/python RNGs for reproducibility.

        Note: This sets the global random state. In parallel execution,
        each worker process has its own random state, so this is safe.
        PYTHONHASHSEED must be set before Python starts, so we don't set it here.
        """
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def _validate_X_y(*, data=None, X=None, y=None):
        """Normalize user inputs into a feature DataFrame and target Series."""
        if data is not None:
            if X is not None or y is not None:
                raise ValueError("Provide either `data` or (`X`, `y`), not both.")
            if not isinstance(data, pd.DataFrame):
                raise TypeError("`data` must be a pandas DataFrame.")
            if data.shape[1] < 1:
                raise ValueError("`data` must contain at least one column.")
            X_df = data.iloc[:, :-1]
            y_ser = data.iloc[:, -1]
        else:
            if X is None or y is None:
                raise ValueError("Provide either `data` or both `X` and `y`.")
            if not isinstance(X, pd.DataFrame):
                raise TypeError("`X` must be a pandas DataFrame.")
            if not isinstance(y, pd.Series):
                raise TypeError("`y` must be a pandas Series.")
            if len(X) != len(y):
                raise ValueError("`X` and `y` must have the same number of rows.")
            X_df = X
            y_ser = y

        target_name = y_ser.name if y_ser.name is not None else "target"
        if target_name in X_df.columns:
            raise ValueError(f"Target column name '{target_name}' conflicts with feature columns.")
        return X_df.copy(), y_ser.rename(target_name).copy()

    def _per_repeat_seed(self, idx):
        """Derive a per-repeat seed from the top-level seed."""
        return int(self.random_state) + int(idx)

    def _effective_n_jobs(self):
        """Return parallel job count capped by number of repeats."""
        n = self.n_jobs if self.n_jobs is not None and self.n_jobs != -1 else self.num_repeats
        return min(int(n), int(self.num_repeats))

    def _merging_requires_bootstrap(self, merger):
        """Return True when the given merger asks for bootstrap statistics."""
        needs_bootstrap = getattr(merger, "needs_bootstrap_merging", False)
        num_bootstrap = getattr(merger, "num_bootstrap", 0)
        return bool(needs_bootstrap and num_bootstrap and num_bootstrap > 0)

    def _should_collect_bootstrap(self):
        """Return True if bootstrap stats should be gathered."""
        if any(self._merging_requires_bootstrap(m) for m in self.merging_strategies):
            return True
        return bool(self.bootstrap and self.bootstrap_num_samples > 0)

    def _generate_selector_ensembles(self, min_group_size):
        """Generate all selector-name combinations with minimum size.

        Args:
            min_group_size: Minimum ensemble size.

        Returns:
            List of tuples of selector names.
        """
        fs_method_names = [fs_method.name for fs_method in self.fs_methods]
        if min_group_size > len(fs_method_names):
            raise ValueError(
                f"Minimum ensemble size of {min_group_size} exceeds available methods ({len(fs_method_names)})."
            )
        return [
            combo for r in range(min_group_size, len(fs_method_names) + 1) for combo in combinations(fs_method_names, r)
        ]

    # Public method to run the feature selection pipeline
    def run(self, verbose=True):
        """Execute the pipeline and return best merged features.

        Returns:
            (merged_features, best_repeat_idx, best_ensemble_name).
        """
        self._set_seed(self.random_state)
        self._instantiate_components()
        self._build_ensemble_index()
        self._reset_run_tracking()

        result_dicts = [{} for _ in range(self._num_metrics_total())]
        cv_splits = list(self._cv_splits())
        parallel_results = self._execute_folds(cv_splits, verbose)
        self._collect_fold_results(parallel_results, result_dicts)

        result_dicts = self._inject_cross_fold_stability(result_dicts)
        best_ensemble, best_repeat = self._select_best_ensemble(result_dicts)
        if self.refit_on_full_data:
            merged_full = self._refit_on_full_data(best_ensemble)
            return (merged_full, None, best_ensemble)

        return (self.merged_features[(int(best_repeat), best_ensemble)], int(best_repeat), best_ensemble)

    # ── Run orchestration helpers ──────────────────────────────────────────────
    def _instantiate_components(self):
        """Instantiate selectors, metrics, and mergers fresh for each run."""
        self.fs_methods = [self._load_class(m, instantiate=True) for m in self._fs_method_specs]
        self.metrics = [self._load_class(m, instantiate=True) for m in self._metric_specs]
        self.merging_strategies = [self._load_class(m, instantiate=True) for m in self._merging_specs]
        self.merging_strategy = self.merging_strategies[0]
        self._multiple_mergers = len(self.merging_strategies) > 1

    def _build_ensemble_index(self):
        """Enumerate all selector ensemble × merger combinations."""
        self.selector_ensembles = self._generate_selector_ensembles(self.min_group_size)
        self.ensembles = []
        self.ensemble_lookup = {}
        for merger in self.merging_strategies:
            for selectors in self.selector_ensembles:
                name = (merger.name, selectors) if self._multiple_mergers else selectors
                self.ensembles.append(name)
                self.ensemble_lookup[name] = selectors

    def _reset_run_tracking(self):
        """Clear per-run state containers."""
        self.fs_subsets = {}
        self.merged_features = {}

    def _execute_folds(self, cv_splits, verbose):
        """Run each CV fold, possibly in parallel."""
        n_jobs = self._effective_n_jobs()  # cap jobs by fold count
        # Parallelize folds with joblib while pinning inner threads to 1 to avoid oversubscription.
        with parallel_backend("loky", inner_max_num_threads=1):
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(self._pipeline_run_for_fold)(i, train_idx, test_idx, verbose)
                for i, (train_idx, test_idx) in enumerate(cv_splits)
            )
        parallel_results.sort(key=lambda x: x[0])  # deterministic ordering by fold index
        return parallel_results

    def _collect_fold_results(self, parallel_results, result_dicts):
        """Merge per-fold outputs into unified mappings."""
        for _, partial_fs_subsets, partial_merged_features, partial_result_dicts in parallel_results:
            self.fs_subsets.update(partial_fs_subsets)
            self.merged_features.update(partial_merged_features)
            for dict_idx in range(len(result_dicts)):
                result_dicts[dict_idx].update(partial_result_dicts[dict_idx])

    def _select_best_ensemble(self, result_dicts):
        """Apply Pareto over ensembles and repeats to pick the winner."""
        means_list = self._calculate_means(result_dicts, self.ensembles)

        # Track and warn about failed ensembles
        failed_ensembles = []
        for i, (ensemble, group_means) in enumerate(zip(self.ensembles, means_list)):
            if not all(mean is not None for mean in group_means):
                failed_ensembles.append(ensemble)
                means_list[i] = [-float("inf")] * len(group_means)

        if failed_ensembles:
            print(f"Warning: {len(failed_ensembles)}/{len(self.ensembles)} ensembles failed to produce valid features")
            if len(failed_ensembles) <= 5:
                print(f"  Failed ensembles: {failed_ensembles}")

        best_ensemble = self._compute_pareto(means_list, self.ensembles)

        ensemble_rows = self._extract_repeat_metrics(best_ensemble, *result_dicts)
        ensemble_rows = [
            row if all(metric is not None for metric in row) else [-float("inf")] * len(row) for row in ensemble_rows
        ]
        best_repeat = self._compute_pareto(ensemble_rows, [str(i) for i in range(self.num_repeats)])
        return best_ensemble, best_repeat

    def _refit_on_full_data(self, ensemble_name):
        """Run selectors and merger on full data for the chosen ensemble."""
        selectors = self.ensemble_lookup.get(ensemble_name)
        if selectors is None:
            raise ValueError(f"Unknown ensemble: {ensemble_name}")

        # Run each selector once on full data
        fs_subsets_local = {}
        feature_names = self.X.columns.tolist()
        X_full = self.X
        y_full = self.y
        for fs_method in self.fs_methods:
            method_name = fs_method.name
            scores, indices = fs_method.select_features(X_full, y_full)
            fs_subsets_local[(0, method_name)] = [
                Feature(
                    name,
                    score=scores[i] if scores is not None else None,
                    selected=(i in indices),
                )
                for i, name in enumerate(feature_names)
            ]

        # Choose appropriate merger (single or by name)
        if self._multiple_mergers:
            merger_name = ensemble_name[0]
            merger = next(m for m in self.merging_strategies if m.name == merger_name)
        else:
            merger = self.merging_strategy

        return self._merge_ensemble_features(
            {(0, name): fs_subsets_local[(0, name)] for name in selectors},
            0,
            selectors,
            merger,
            bootstrap_stats=None,
            feature_names=feature_names,
        )

    def _cv_splits(self):
        """Yield train/test indices for K-fold CV (stratified when classification)."""
        if self.task == "classification":
            from sklearn.model_selection import StratifiedKFold

            cv = StratifiedKFold(
                n_splits=self.num_repeats,
                shuffle=True,
                random_state=self.random_state,
            )
            return cv.split(self.X, self.y)
        else:
            from sklearn.model_selection import KFold

            cv = KFold(
                n_splits=self.num_repeats,
                shuffle=True,
                random_state=self.random_state,
            )
            return cv.split(self.X, self.y)

    def _pipeline_run_for_fold(self, fold_idx, train_idx, test_idx, verbose):
        """Execute one CV fold and return partial results tuple."""
        # Set seed at the start of each fold worker for reproducibility
        # This ensures each parallel worker has consistent random state
        fold_seed = self._per_repeat_seed(fold_idx)
        self._set_seed(fold_seed)

        X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
        y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        fs_subsets_local = self._compute_subset(train_data, fold_idx)
        feature_names = train_data.drop(columns=[self.target_name]).columns.tolist()
        bootstrap_stats = (
            self._compute_bootstrap_stats(train_data, fold_idx, feature_names)
            if self._should_collect_bootstrap()
            else None
        )
        merged_features_local = self._compute_merging(
            fs_subsets_local,
            fold_idx,
            verbose,
            bootstrap_stats=bootstrap_stats,
            feature_names=feature_names,
        )
        local_result_dicts = self._compute_metrics(
            fs_subsets_local, merged_features_local, train_data, test_data, fold_idx
        )

        # Return fold index as the first element
        return fold_idx, fs_subsets_local, merged_features_local, local_result_dicts

    def _compute_bootstrap_stats(self, train_data, idx, feature_names):
        """Collect selection counts across bootstrap resamples for each selector."""
        self._set_seed(self._per_repeat_seed(idx))

        num_bootstrap_attr = (
            max(getattr(m, "num_bootstrap", 0) for m in self.merging_strategies) if self.merging_strategies else 0
        )
        num_bootstrap = int(num_bootstrap_attr or (self.bootstrap_num_samples if self.bootstrap else 0))
        if num_bootstrap <= 0:
            return {}

        use_scores = (
            any(getattr(m, "use_scores", self.bootstrap_use_scores) for m in self.merging_strategies)
            or self.bootstrap_use_scores
        )

        X_train = train_data.drop(columns=[self.target_name])
        y_train = train_data[self.target_name]

        n_rows = len(train_data)
        stats = {}
        rng = np.random.default_rng(self._per_repeat_seed(idx))

        for fs_method in self.fs_methods:
            counts = np.zeros(len(feature_names), dtype=np.int32)
            score_sums = np.zeros(len(feature_names), dtype=np.float32)

            for b in range(num_bootstrap):
                sample_idx = rng.integers(0, n_rows, size=n_rows)
                X_boot = X_train.iloc[sample_idx]
                y_boot = y_train.iloc[sample_idx]

                scores, indices = fs_method.select_features(X_boot, y_boot)
                if indices is None:
                    continue

                sel_idx = np.asarray(indices, dtype=int)
                if sel_idx.size == 0:
                    continue

                counts[sel_idx] += 1

                if use_scores and scores is not None:
                    score_arr = np.asarray(scores, dtype=np.float32)
                    min_v = float(score_arr.min())
                    rng_v = float(score_arr.max() - min_v) or 1.0
                    norm_scores = (score_arr - min_v) / rng_v
                    score_sums[sel_idx] += norm_scores[sel_idx]

            stats[fs_method.name] = {
                "counts": counts,
                "score_sums": score_sums,
                "n_runs": num_bootstrap,
            }

        stats["_feature_names"] = feature_names
        return stats

    def _build_bootstrap_subsets(self, selectors, feature_names, bootstrap_stats):
        """Construct per-method Feature lists using bootstrap selection frequencies."""
        if not feature_names or not bootstrap_stats:
            return None

        n_features = len(feature_names)
        freq_accum = np.zeros(n_features, dtype=np.float32)
        score_accum = np.zeros(n_features, dtype=np.float32)
        per_method_freq = {}
        per_method_scores = {}
        num_methods = 0

        for method_name in selectors:
            stats = bootstrap_stats.get(method_name)
            if not stats:
                continue
            n_runs = stats.get("n_runs", 0)
            if n_runs <= 0:
                continue
            counts = np.asarray(stats.get("counts", []), dtype=np.float32)
            if counts.shape[0] != n_features:
                continue
            freq = counts / float(n_runs)
            per_method_freq[method_name] = freq
            freq_accum += freq
            num_methods += 1

            if self.bootstrap_use_scores:
                score_sums = np.asarray(stats.get("score_sums", []), dtype=np.float32)
                avg_scores = np.divide(
                    score_sums,
                    counts,
                    out=np.zeros_like(score_sums),
                    where=counts > 0,
                )
            else:
                avg_scores = np.zeros_like(freq)
            per_method_scores[method_name] = avg_scores
            score_accum += avg_scores

        if num_methods == 0:
            return None

        global_freq = freq_accum / float(num_methods)
        global_scores = score_accum / float(num_methods)
        ranking = sorted(
            range(n_features),
            key=lambda i: (global_freq[i], global_scores[i], -i),
            reverse=True,
        )

        base_survivors = [i for i, f in enumerate(global_freq) if f >= self.bootstrap_min_freq]

        if self.fill and self.num_features_to_select:
            max_features = min(self.num_features_to_select, n_features)
            survivor_idx = [i for i in ranking if i in base_survivors][:max_features]
            if len(survivor_idx) < max_features:
                filler_idx = [i for i in ranking if i not in survivor_idx][: max_features - len(survivor_idx)]
                survivor_idx = survivor_idx + filler_idx
        else:
            survivor_idx = base_survivors

        if not survivor_idx:
            return None

        subsets = []
        for method_name in selectors:
            freq = per_method_freq.get(method_name)
            scores = per_method_scores.get(method_name, np.zeros(n_features, dtype=np.float32))
            if freq is None:
                # Method missing stats; skip it to avoid inconsistent lengths.
                continue
            features = [Feature(name=feature_names[i], score=float(scores[i]), selected=True) for i in survivor_idx]
            subsets.append(features)

        if not subsets:
            return None
        return subsets

    def _compute_subset(self, train_data, idx):
        """Compute selected Feature objects per method for this repeat."""
        self._set_seed(self._per_repeat_seed(idx))
        X_train = train_data.drop(columns=[self.target_name])
        y_train = train_data[self.target_name]
        feature_names = X_train.columns.tolist()

        fs_subsets_local = {}
        for fs_method in self.fs_methods:
            method_name = fs_method.name
            scores, indices = fs_method.select_features(X_train, y_train)
            fs_subsets_local[(idx, method_name)] = [
                Feature(
                    name,
                    score=scores[i] if scores is not None else None,
                    selected=(i in indices),
                )
                for i, name in enumerate(feature_names)
            ]

        return fs_subsets_local

    def _compute_merging(
        self,
        fs_subsets_local,
        idx,
        verbose=True,
        bootstrap_stats=None,
        feature_names=None,
    ):
        """Merge per-ensemble features and return mapping for this repeat."""
        self._set_seed(self._per_repeat_seed(idx))
        merged_features_local = {}
        for merger in self.merging_strategies:
            for selectors in self.selector_ensembles:
                merged = self._merge_ensemble_features(
                    fs_subsets_local,
                    idx,
                    selectors,
                    merger=merger,
                    bootstrap_stats=bootstrap_stats,
                    feature_names=feature_names,
                )
                ensemble_name = (merger.name, selectors) if self._multiple_mergers else selectors
                if merged:
                    merged_features_local[(idx, ensemble_name)] = merged
                elif verbose:
                    print(f"Warning: {ensemble_name} produced no merged features.")
        return merged_features_local

    def _merge_ensemble_features(
        self,
        fs_subsets_local,
        idx,
        selectors,
        merger,
        *,
        bootstrap_stats=None,
        feature_names=None,
    ):
        """Merge features for a specific ensemble of selectors."""
        group_features = [[f for f in fs_subsets_local[(idx, method)] if f.selected] for method in selectors]

        # Strategy-provided bootstrap handling (e.g., FrequencyBootstrapMerger)
        if bootstrap_stats and self._merging_requires_bootstrap(merger):
            feature_names = feature_names or bootstrap_stats.get("_feature_names")
            return merger.merge(
                group_features,
                self.num_features_to_select,
                fill=self.fill,
                group=selectors,
                feature_names=feature_names,
                bootstrap_stats=bootstrap_stats,
            )

        # Pipeline-level bootstrap pre-processing for any merger
        if bootstrap_stats and self.bootstrap:
            preprocessed = self._build_bootstrap_subsets(selectors, feature_names, bootstrap_stats)
            if preprocessed is not None:
                subsets = preprocessed
                is_set_based_attr = getattr(merger, "is_set_based", None)
                is_set_based = bool(is_set_based_attr()) if callable(is_set_based_attr) else bool(is_set_based_attr)
                if is_set_based:
                    return merger.merge(subsets, self.num_features_to_select, fill=self.fill)
                return merger.merge(subsets, self.num_features_to_select)

        # Determine set-based vs rank-based via method call when available
        is_set_based_attr = getattr(merger, "is_set_based", None)
        if callable(is_set_based_attr):
            is_set_based = bool(is_set_based_attr())
        elif isinstance(is_set_based_attr, bool):
            is_set_based = is_set_based_attr
        else:
            is_set_based = True
        if is_set_based:
            return merger.merge(group_features, self.num_features_to_select, fill=self.fill)
        else:
            return merger.merge(group_features, self.num_features_to_select)

    def _compute_performance_metrics(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        """Compute performance metrics using configured metric methods."""
        self._set_seed(self.random_state)
        if not self.metrics:
            return []

        shared_results = {}
        metric_values = []

        for metric in self.metrics:
            aggregator = getattr(metric, "aggregate_from_results", None)
            if not callable(aggregator):
                metric_values.append(metric.compute(X_train, y_train, X_test, y_test))
                continue

            signature_fn = getattr(metric, "model_signature", None)
            cache_key = signature_fn() if callable(signature_fn) else None
            results = shared_results.get(cache_key) if cache_key is not None else None

            if results is None:
                results = metric.train_and_predict(X_train, y_train, X_test, y_test)
                if cache_key is not None:
                    shared_results[cache_key] = results

            metric_values.append(aggregator(y_test, results))

        return metric_values

    def _compute_metrics(
        self,
        fs_subsets_local,
        merged_features_local,
        train_data,
        test_data,
        idx,
    ):
        """Compute performance and stability metrics for each ensemble."""
        self._set_seed(self._per_repeat_seed(idx))
        num_metrics = self._num_metrics_total()
        local_result_dicts = [{} for _ in range(num_metrics)]
        feature_train = train_data.drop(columns=self.target_name)
        feature_test = test_data.drop(columns=self.target_name)
        y_train_full = train_data[self.target_name]
        y_test_full = test_data[self.target_name]
        column_positions = {name: position for position, name in enumerate(feature_train.columns)}

        for ensemble_name in self.ensembles:
            selectors = self.ensemble_lookup.get(ensemble_name, ())
            key = (idx, ensemble_name)
            if key not in merged_features_local:
                continue

            merged_feature_names = merged_features_local[key]
            ordered_features = [feature for feature in merged_feature_names if feature in column_positions]
            ordered_features.sort(key=column_positions.__getitem__)
            if not ordered_features:
                continue

            X_train_subset = feature_train[ordered_features]
            X_test_subset = feature_test[ordered_features]

            metric_vals = self._compute_performance_metrics(
                X_train_subset,
                y_train_full,
                X_test_subset,
                y_test_full,
            )
            for m_idx, val in enumerate(metric_vals):
                local_result_dicts[m_idx][key] = val

            metric_slot = len(metric_vals)
            if self.stability_mode in ("agreement", "both"):
                fs_lists = [[f.name for f in fs_subsets_local[(idx, method)] if f.selected] for method in selectors]
                stability = compute_stability_metrics(fs_lists) if fs_lists else 0
                local_result_dicts[metric_slot][key] = stability
                metric_slot += 1

            # cross-fold stability is filled later after all folds are known

        return local_result_dicts

    @staticmethod
    def _calculate_means(
        result_dicts,
        ensemble_names,
    ):
        """Calculate mean metrics per ensemble across repeats."""
        means_list = []
        for ensemble in ensemble_names:
            ensemble_means = []
            for d in result_dicts:
                vals = [value for (idx, name), value in d.items() if name == ensemble]
                m = np.mean(vals) if len(vals) else np.nan
                ensemble_means.append(None if np.isnan(m) else float(m))
            means_list.append(ensemble_means)
        return means_list

    def _inject_cross_fold_stability(self, result_dicts):
        """Compute stability of merged features across folds for each ensemble.

        Cross-fold stability is a single value per ensemble (computed across all folds),
        but we replicate it for each fold index so it can be used in per-fold Pareto selection.
        """
        if self.stability_mode not in ("cross_folds", "both"):
            return result_dicts

        # Prepare destination dict index
        dest_idx = len(self.metrics)
        if self.stability_mode == "both":
            dest_idx += 1

        stability_dict = result_dicts[dest_idx] if dest_idx < len(result_dicts) else {}
        for ensemble in self.ensembles:
            merged_sets = [
                set(self.merged_features[(fold_idx, ensemble)])
                for fold_idx in range(self.num_repeats)
                if (fold_idx, ensemble) in self.merged_features
            ]
            if len(merged_sets) < 2:
                stability_value = 0.0
            else:
                stability_value = compute_stability_metrics([list(s) for s in merged_sets])

            # Replicate the cross-fold stability value for each fold
            # so it can be accessed with (fold_idx, ensemble) keys
            for fold_idx in range(self.num_repeats):
                stability_dict[(fold_idx, ensemble)] = stability_value

        if dest_idx >= len(result_dicts):
            result_dicts.append(stability_dict)
        else:
            result_dicts[dest_idx] = stability_dict
        return result_dicts

    @staticmethod
    def _compute_pareto(values, names):
        """Return the name of the winner using Pareto analysis.

        Args:
            values: List of metric vectors (one per group).
            names: Corresponding group names.

        Returns:
            Name of the best group according to Pareto dominance.

        Raises:
            ValueError: If all groups have failed (all -inf values).
        """
        if not values:
            raise ValueError("Cannot perform Pareto analysis with no data.")

        # Check if all ensembles have completely failed
        all_failed = all(all(v == -float("inf") for v in vec) for vec in values)
        if all_failed:
            raise ValueError(
                "All ensembles failed to produce valid features. "
                "Check your feature selector and merging strategy configuration."
            )

        pareto = ParetoAnalysis(values, names)
        pareto_results = pareto.get_results()
        return pareto_results[0][0]

    def _extract_repeat_metrics(
        self,
        ensemble,
        *result_dicts,
    ):
        """Return a row per repeat for the given ensemble."""
        result_array = []
        for idx in range(self.num_repeats):
            row = [d.get((idx, ensemble)) for d in result_dicts]
            result_array.append(row)
        return result_array

    def _load_class(self, input, instantiate=False):
        """Resolve identifiers to classes/instances and optionally instantiate.

        Args:
            input: Identifier or instance of a selector/merger/metric.
            instantiate: If True, instantiate using extracted parameters.

        Returns:
            Class or instance.

        Raises:
            ValueError: If ``input`` is invalid.
        """
        if isinstance(input, str):
            cls, params = get_class_info(input)
            if instantiate:
                init_params = extract_params(cls, self, params)
                return cls(**init_params)
            return cls
        elif hasattr(input, "select_features") or hasattr(input, "merge"):
            # Assumes valid instance if it has a 'select_features' or 'merge' method.
            if instantiate:
                # Best-effort: re-instantiate using the class and pipeline params
                cls = input.__class__
                init_params = extract_params(cls, self, [])
                try:
                    return cls(**init_params)
                except Exception:
                    # Fallback to returning the same instance if re-instantiation fails
                    return input
            return input
        else:
            raise ValueError(
                "Input must be a string identifier or a valid instance of a feature selector or merging strategy."
            )

    def _num_metrics_total(self):
        """Count performance metrics plus configured stability signals."""
        extra = 0
        if self.stability_mode in ("agreement", "both"):
            extra += 1
        if self.stability_mode in ("cross_folds", "both"):
            extra += 1
        return len(self.metrics) + extra

    def __str__(self):
        return (
            f"Feature selection pipeline with: merging strategy: {self.merging_strategy}, "
            f"feature selection methods: {self.fs_methods}, "
            f"number of repeats: {self.num_repeats}"
        )

    def __call__(self):
        return self.run()
