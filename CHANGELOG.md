# Changelog

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
