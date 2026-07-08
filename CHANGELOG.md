# Changelog

## [0.4.0] - 2026-07-08

### Fixed

- Rank-based mergers (`BordaMerger`, `ArithmeticMeanMerger`, `L2NormMerger`) now align subsets by feature name; positional alignment meant the merged result could only ever contain the first selector's features
- `UnionOfIntersectionsMerger` fill/trim scores are min-max normalized within each selector (previously across selectors at each list position); single-selector merges with `fill=True` no longer crash
- User-provided selector/merger instances keep their configuration (e.g. `ConsensusMerger(k=3, fill=True)` was silently re-instantiated with defaults)
- Selector kwargs are forwarded to the underlying models: `random_state` and hyperparameters were silently dropped by the RandomForest, XGBoost, and MutualInfo selectors, and Lasso/ElasticNet lost kwargs after the first call
- Metric instances are accepted by the pipeline (e.g. `metrics=[Accuracy()]`)

### Added

- `selector_pool_factor` parameter (default `2.0`): selectors rank a wider candidate pool while mergers still return `num_features_to_select` features, giving larger intersections and more complete rankings
- `include_consistency` parameter (default `False`): the consistency objective is now opt-in instead of always adding a noisy Pareto dimension
- Warnings when Pareto selection uses more than 4 objectives, when `min_group_size` equals the number of selectors, and when the utopia-distance tie-break decides the winner
- Validation of `stability_mode` values

### Changed

- Scale-sensitive models standardize their inputs internally: SVM, Lasso, and ElasticNet-classification selectors, and the LogisticRegression metric model

### Note

- Default results differ from 0.3.x due to `selector_pool_factor=2.0`, internal standardization, and the merger fixes

## [0.3.1] - 2026-02-23

- Re-release of 0.3.0 (no code changes)

## [0.3.0] - 2026-02-23

### Changed

- Metric evaluation uses `HistGradientBoosting` instead of `GradientBoosting` for faster model training

## [0.2.0] - 2026-02-03

### Changed

- Refactored to single-stage Pareto optimization with consistency metric
- Added `stability_mode` parameter with options: "selector_agreement", "fold_stability", "all"
- Pipeline now refits on full data after ensemble selection (better generalization)
- Switched from train/test split to Stratified K-Fold cross-validation
- `num_repeats` parameter now specifies number of CV folds (previously: random train/test repeats)
- Internal terminology: `subgroup_names` renamed to `selector_ensembles`

### Added

- `FrequencyBootstrapMerger` for bootstrap-based feature merging
- Support for multiple merging strategies in single pipeline run (pass list to `merging_strategy`)
- Metrics caching (`fold_model_cache`) to reduce redundant model training within folds
- Cross-fold stability injection for robust feature selection

### Fixed

- Pareto analysis handling of failed ensembles (-inf values)
- Seed management for reproducible cross-fold stability keys

### Removed

- `diversity_agreement()` function from stability_metrics module
- `__version__` export from package root (use `importlib.metadata` instead)
- Pipeline-level bootstrap parameter (bootstrap is now merger-specific via `FrequencyBootstrapMerger`)
- CLI interface (`efs-pipeline` command) and scripts directory - use the Python API directly

### Breaking Changes

- Return value: `pipeline.run()` now returns `(features, ensemble)` instead of `(features, best_repeat, ensemble)`
- Removed: `from moosefs import __version__` no longer works
- Removed: `diversity_agreement()` function no longer available
- Attribute renamed: `pipeline.subgroup_names` → `pipeline.ensembles`

### Benchmark Results (3-repeated 4-fold CV)

Dataset: 300 samples, 500 features, 30 informative

| Metric              | v0.1.0         | v0.2.0         | Change |
|---------------------|----------------|----------------|--------|
| Execution Time (s)  | 35.2 ± 3.6     | 37.5 ± 3.4     | +6.5%  |
| Test F1             | 0.824 ± 0.051  | 0.825 ± 0.036  | +0.2%  |
| Test Accuracy       | 0.824 ± 0.050  | 0.826 ± 0.036  | -      |
| Feature TPR         | 0.350          | 0.353          | -      |
| Stability (Jaccard) | 0.206          | 0.253          | +23%   |

## [0.1.0] - 2024-10-01

### Initial Release

- Initial release of MOOSE-FS
- `FeatureSelectionPipeline` for ensemble feature selection
- Multiple feature selectors: F-statistic, Random Forest, SVM, XGBoost, Mutual Info, MRMR, Lasso, ElasticNet
- Multiple merging strategies: Borda, Union of Intersections, Consensus, L2 Norm, Arithmetic Mean
- Performance metrics for classification and regression tasks
- Stability metrics using Novovicova measure
- Pareto-based multi-objective optimization for ensemble selection
