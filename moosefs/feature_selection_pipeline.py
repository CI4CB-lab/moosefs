from itertools import combinations
import os
import random
from typing import Any, Optional

from joblib import Parallel, delayed, parallel_backend
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# tqdm is not used; keep imports minimal
from .core import Feature, ParetoAnalysis
from .metrics.stability_metrics import compute_stability_metrics, diversity_agreement
from .utils import extract_params, get_class_info

# for test purpose
agreement_flag = False


class FeatureSelectionPipeline:
    """End-to-end pipeline for ensemble feature selection.

    Orchestrates feature scoring, merging, metric evaluation, and Pareto-based
    selection across repeated runs and method subgroups.
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
            min_group_size: Minimum number of methods in each subgroup.
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
        self._validate_task(task)
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

        # set seed for reproducibility
        self._set_seed(self.random_state)

        # Keep original specs and also instantiate now for introspection
        self._fs_method_specs = list(fs_methods)
        self._metric_specs = list(metrics)
        self._merging_spec = merging_strategy

        # dynamically load classes or instantiate them (initial instances)
        self.fs_methods = [self._load_class(m, instantiate=True) for m in self._fs_method_specs]
        self.metrics = [self._load_class(m, instantiate=True) for m in self._metric_specs]
        self.merging_strategy = self._load_class(self._merging_spec, instantiate=True)

        # validate and preparation
        if self.num_features_to_select is None:
            raise ValueError("num_features_to_select must be provided")
        # subgroup names are generated in run() after instantiation
        self.subgroup_names: list = []

    @staticmethod
    def _validate_task(task: str) -> None:
        """Validate task string.

        Args:
            task: Expected 'classification' or 'regression'.
        """
        if task not in ["classification", "regression"]:
            raise ValueError("Task must be either 'classification' or 'regression'.")

    @staticmethod
    def _set_seed(seed: int, idx: Optional[int] = None) -> None:
        """Seed numpy/python RNGs for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

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

    def _per_repeat_seed(self, idx: int) -> int:
        """Derive a per-repeat seed from the top-level seed."""
        return int(self.random_state) + int(idx)

    def _effective_n_jobs(self) -> int:
        """Return parallel job count capped by number of repeats."""
        n = self.n_jobs if self.n_jobs is not None and self.n_jobs != -1 else self.num_repeats
        return min(int(n), int(self.num_repeats))

    def _merging_requires_bootstrap(self) -> bool:
        """Return True when the chosen merger asks for bootstrap statistics."""
        needs_bootstrap = getattr(self.merging_strategy, "needs_bootstrap_merging", False)
        num_bootstrap = getattr(self.merging_strategy, "num_bootstrap", 0)
        return bool(needs_bootstrap and num_bootstrap and num_bootstrap > 0)

    def _should_collect_bootstrap(self) -> bool:
        """Return True if bootstrap stats should be gathered."""
        if self._merging_requires_bootstrap():
            return True
        return bool(self.bootstrap and self.bootstrap_num_samples > 0)

    def _generate_subgroup_names(self, min_group_size: int) -> list:
        """Generate all selector-name combinations with minimum size.

        Args:
            min_group_size: Minimum subgroup size.

        Returns:
            List of tuples of selector names.
        """
        fs_method_names = [fs_method.name for fs_method in self.fs_methods]
        if min_group_size > len(fs_method_names):
            raise ValueError(
                f"Minimum group size of {min_group_size} exceeds available methods ({len(fs_method_names)})."
            )
        return [
            combo for r in range(min_group_size, len(fs_method_names) + 1) for combo in combinations(fs_method_names, r)
        ]

    # Public method to run the feature selection pipeline
    def run(self, verbose: bool = True) -> tuple:
        """Execute the pipeline and return best merged features.

        Returns:
            (merged_features, best_repeat_idx, best_group_names).
        """
        self._set_seed(self.random_state)

        # Fresh objects for each run to avoid hidden state
        self.fs_methods = [self._load_class(m, instantiate=True) for m in self._fs_method_specs]
        self.metrics = [self._load_class(m, instantiate=True) for m in self._metric_specs]
        self.merging_strategy = self._load_class(self._merging_spec, instantiate=True)

        # Regenerate subgroup names from fresh fs_methods
        self.subgroup_names = self._generate_subgroup_names(self.min_group_size)

        # Reset internal state so that run() always starts fresh
        self.fs_subsets: dict = {}
        self.merged_features: dict = {}

        num_metrics = self._num_metrics_total()
        result_dicts: list = [{} for _ in range(num_metrics)]

        # Ensure we don't allocate more jobs than repeats
        n_jobs = self._effective_n_jobs()

        with parallel_backend("loky", inner_max_num_threads=1):  # Prevents oversubscription
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(self._pipeline_run_for_repeat)(i, verbose) for i in range(self.num_repeats)
            )

        # Sort results by repeat index
        parallel_results.sort(key=lambda x: x[0])  # Now, x[0] is the repeat index

        # Merge results in a fixed order
        self.fs_subsets = {}
        self.merged_features = {}

        for (
            _,
            partial_fs_subsets,
            partial_merged_features,
            partial_result_dicts,
        ) in parallel_results:
            self.fs_subsets.update(partial_fs_subsets)
            self.merged_features.update(partial_merged_features)
            for dict_idx in range(num_metrics):
                result_dicts[dict_idx].update(partial_result_dicts[dict_idx])

        # Compute Pareto analysis as usual
        means_list = self._calculate_means(result_dicts, self.subgroup_names)
        means_list = self._replace_none(means_list)
        # pairs = sorted(zip(self.subgroup_names, means_list), key=lambda p: tuple(p[0]))
        # self.subgroup_names, means_list = map(list, zip(*pairs))
        best_group = self._compute_pareto(means_list, self.subgroup_names)
        best_group_metrics = self._extract_repeat_metrics(best_group, *result_dicts)
        best_group_metrics = self._replace_none(best_group_metrics)
        best_repeat = self._compute_pareto(best_group_metrics, [str(i) for i in range(self.num_repeats)])

        return (self.merged_features[(int(best_repeat), best_group)], int(best_repeat), best_group)

    def _pipeline_run_for_repeat(self, i: int, verbose: bool) -> Any:
        """Execute one repeat and return partial results tuple."""
        self._set_seed(self._per_repeat_seed(i))

        train_data, test_data = self._split_data(test_size=0.20, random_state=self._per_repeat_seed(i))

        fs_subsets_local = self._compute_subset(train_data, i)
        feature_names = train_data.drop(columns=[self.target_name]).columns.tolist()
        bootstrap_stats = (
            self._compute_bootstrap_stats(train_data, i, feature_names) if self._should_collect_bootstrap() else None
        )
        merged_features_local = self._compute_merging(
            fs_subsets_local,
            i,
            verbose,
            bootstrap_stats=bootstrap_stats,
            feature_names=feature_names,
        )
        local_result_dicts = self._compute_metrics(fs_subsets_local, merged_features_local, train_data, test_data, i)

        # Return repeat index as the first element
        return i, fs_subsets_local, merged_features_local, local_result_dicts

    def _replace_none(self, metrics: list) -> list:
        """Replace any group with None with a list of -inf.

        Args:
            metrics: Per-group metric lists.

        Returns:
            Same shape with None replaced by -inf rows.
        """
        return [
            (
                group_metrics
                if all(metric is not None for metric in group_metrics)
                else [-float("inf")] * len(group_metrics)
            )
            for group_metrics in metrics
        ]

    def _split_data(self, test_size: float, random_state: int) -> tuple:
        """Split data into train/test using stratification when classification."""
        stratify = self.y if self.task == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        return train_df, test_df

    def _compute_bootstrap_stats(self, train_data: pd.DataFrame, idx: int, feature_names: list) -> dict:
        """Collect selection counts across bootstrap resamples for each selector."""
        self._set_seed(self._per_repeat_seed(idx))

        num_bootstrap = int(
            getattr(self.merging_strategy, "num_bootstrap", 0) or (self.bootstrap_num_samples if self.bootstrap else 0)
        )
        if num_bootstrap <= 0:
            return {}

        use_scores = bool(getattr(self.merging_strategy, "use_scores", self.bootstrap_use_scores))

        X_train = train_data.drop(columns=[self.target_name])
        y_train = train_data[self.target_name]

        n_rows = len(train_data)
        stats: dict = {}
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

    def _build_bootstrap_subsets(self, group: tuple, feature_names: list, bootstrap_stats: dict) -> Optional[list]:
        """Construct per-method Feature lists using bootstrap selection frequencies."""
        if not feature_names or not bootstrap_stats:
            return None

        n_features = len(feature_names)
        freq_accum = np.zeros(n_features, dtype=np.float32)
        per_method_freq = {}
        per_method_scores = {}
        num_methods = 0

        for method_name in group:
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

        if num_methods == 0:
            return None

        global_freq = freq_accum / float(num_methods)
        survivor_idx = [i for i, f in enumerate(global_freq) if f >= self.bootstrap_min_freq]

        # If nothing meets the threshold but fill=True, keep the top-k by frequency.
        if not survivor_idx and self.fill and self.num_features_to_select:
            top_idx = np.argsort(-global_freq, kind="stable")[: self.num_features_to_select]
            survivor_idx = [int(i) for i in top_idx]

        if not survivor_idx:
            return None

        subsets = []
        for method_name in group:
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

    def _compute_subset(self, train_data: pd.DataFrame, idx: int) -> dict:
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
        fs_subsets_local: dict,
        idx: int,
        verbose: bool = True,
        bootstrap_stats: Optional[dict] = None,
        feature_names: Optional[list] = None,
    ) -> dict:
        """Merge per-group features and return mapping for this repeat."""
        self._set_seed(self._per_repeat_seed(idx))
        merged_features_local = {}
        for group in self.subgroup_names:
            merged = self._merge_group_features(
                fs_subsets_local,
                idx,
                group,
                bootstrap_stats=bootstrap_stats,
                feature_names=feature_names,
            )
            if merged:
                merged_features_local[(idx, group)] = merged
            elif verbose:
                print(f"Warning: {group} produced no merged features.")
        return merged_features_local

    def _merge_group_features(
        self,
        fs_subsets_local: dict,
        idx: int,
        group: tuple,
        *,
        bootstrap_stats: Optional[dict] = None,
        feature_names: Optional[list] = None,
    ) -> list:
        """Merge features for a specific group of methods.

        Args:
            idx: Repeat index.
            group: Tuple of selector names.

        Returns:
            Merged features (type depends on strategy).
        """
        group_features = [[f for f in fs_subsets_local[(idx, method)] if f.selected] for method in group]

        # Strategy-provided bootstrap handling (e.g., FrequencyBootstrapMerger)
        if bootstrap_stats and self._merging_requires_bootstrap():
            feature_names = feature_names or bootstrap_stats.get("_feature_names")
            return self.merging_strategy.merge(
                group_features,
                self.num_features_to_select,
                fill=self.fill,
                group=group,
                feature_names=feature_names,
                bootstrap_stats=bootstrap_stats,
            )

        # Pipeline-level bootstrap pre-processing for any merger
        if bootstrap_stats and self.bootstrap:
            preprocessed = self._build_bootstrap_subsets(group, feature_names, bootstrap_stats)
            if preprocessed is not None:
                subsets = preprocessed
                is_set_based_attr = getattr(self.merging_strategy, "is_set_based", None)
                is_set_based = bool(is_set_based_attr()) if callable(is_set_based_attr) else bool(is_set_based_attr)
                if is_set_based:
                    return self.merging_strategy.merge(subsets, self.num_features_to_select, fill=self.fill)
                return self.merging_strategy.merge(subsets, self.num_features_to_select)

        # Determine set-based vs rank-based via method call when available
        is_set_based_attr = getattr(self.merging_strategy, "is_set_based", None)
        if callable(is_set_based_attr):
            is_set_based = bool(is_set_based_attr())
        elif isinstance(is_set_based_attr, bool):
            is_set_based = is_set_based_attr
        else:
            is_set_based = True  # default behavior as before
        if is_set_based:
            return self.merging_strategy.merge(group_features, self.num_features_to_select, fill=self.fill)
        else:
            return self.merging_strategy.merge(group_features, self.num_features_to_select)

    def _compute_performance_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> list:
        """Compute performance metrics using configured metric methods.

        Returns:
            Averaged metric values per configured metric.
        """
        self._set_seed(self.random_state)
        if not self.metrics:
            return []

        shared_results: dict = {}
        metric_values: list = []

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
        fs_subsets_local: dict,
        merged_features_local: dict,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        idx: int,
    ) -> list:
        """Compute and collect performance and stability metrics for subgroups.

        Args:
            fs_subsets_local: Local selected Feature lists per (repeat, method).
            merged_features_local: Merged features per (repeat, group).
            train_data: Training dataframe.
            test_data: Test dataframe.
            idx: Repeat index.

        Returns:
            List of per-metric dicts keyed by (repeat, group).
        """
        self._set_seed(self._per_repeat_seed(idx))
        num_metrics = self._num_metrics_total()
        local_result_dicts = [{} for _ in range(num_metrics)]
        feature_train = train_data.drop(columns=self.target_name)
        feature_test = test_data.drop(columns=self.target_name)
        y_train_full = train_data[self.target_name]
        y_test_full = test_data[self.target_name]
        column_positions = {name: position for position, name in enumerate(feature_train.columns)}

        for group in self.subgroup_names:
            key = (idx, group)
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

            fs_lists = [[f.name for f in fs_subsets_local[(idx, method)] if f.selected] for method in group]
            stability = compute_stability_metrics(fs_lists) if fs_lists else 0

            if agreement_flag:
                agreement = diversity_agreement(fs_lists, ordered_features, alpha=0.5) if fs_lists else 0
                local_result_dicts[len(metric_vals)][key] = agreement
                local_result_dicts[len(metric_vals) + 1][key] = stability
            else:
                local_result_dicts[len(metric_vals)][key] = stability

        return local_result_dicts

    @staticmethod
    def _calculate_means(
        result_dicts: list,
        group_names: list,
    ) -> list:
        """Calculate mean metrics per subgroup across repeats.

        Args:
            result_dicts: Per-metric dicts keyed by (repeat, group).
            group_names: Subgroup names to summarize.

        Returns:
            List of [means per metric] for each subgroup.
        """
        means_list = []
        for group in group_names:
            group_means = []
            for d in result_dicts:
                vals = [value for (idx, name), value in d.items() if name == group]
                m = np.mean(vals) if len(vals) else np.nan
                group_means.append(None if np.isnan(m) else float(m))
            means_list.append(group_means)
        return means_list

    @staticmethod
    def _compute_pareto(groups: list, names: list) -> Any:
        """Return the name of the winner using Pareto analysis."""
        pareto = ParetoAnalysis(groups, names)
        pareto_results = pareto.get_results()
        return pareto_results[0][0]

    def _extract_repeat_metrics(
        self,
        group: Any,
        *result_dicts: dict,
    ) -> list:
        """Return a row per repeat for the given group.

        Missing values remain as None and are later replaced by -inf.
        """
        result_array: list = []
        for idx in range(self.num_repeats):  # <- full range
            row = [d.get((idx, group)) for d in result_dicts]
            result_array.append(row)
        return result_array

    def _load_class(self, input: Any, instantiate: bool = False) -> Any:
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

    def _num_metrics_total(self) -> int:
        """Return total number of metrics tracked per group.

        Includes performance metrics plus stability and optional agreement.
        """
        return len(self.metrics) + (2 if agreement_flag else 1)

    def __str__(self) -> str:
        return (
            f"Feature selection pipeline with: merging strategy: {self.merging_strategy}, "
            f"feature selection methods: {self.fs_methods}, "
            f"number of repeats: {self.num_repeats}"
        )

    def __call__(self) -> Any:
        return self.run()
