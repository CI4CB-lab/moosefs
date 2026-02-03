Usage
=====

Overview
--------

The ``moosefs`` package provides a customizable, ensemble-based feature selection pipeline that balances predictive performance and stability. The core entry is ``FeatureSelectionPipeline``.

Getting Started
---------------

.. code-block:: python

    from moosefs import FeatureSelectionPipeline

    # Either pass a single DataFrame (last column = target) or `X` and `y`.
    # Assume `data` is a pandas DataFrame whose last column "label" stores the target.
    X = data.drop(columns=["label"])
    y = data["label"]
    fs_methods = ["f_statistic_selector", "random_forest_selector", "svm_selector"]
    merging_strategy = "union_of_intersections_merger"

    pipeline = FeatureSelectionPipeline(
        X=X,
        y=y,
        fs_methods=fs_methods,
        merging_strategy=merging_strategy,
        num_repeats=5,
        task="classification",
        num_features_to_select=10,
        stability_mode="fold_stability",  # Options: "selector_agreement", "fold_stability", "all"
    )

    # Run the pipeline
    selected_features, best_ensemble = pipeline.run()
    print(f"Selected {len(selected_features)} features using ensemble: {best_ensemble}")

Stability Modes
---------------

The ``stability_mode`` parameter controls which stability metrics are included in the Pareto optimization:

- ``"selector_agreement"``: Measures how much the selectors within each ensemble agree on feature selection
- ``"fold_stability"``: Measures consistency of selected features across cross-validation folds (default)
- ``"all"``: Includes both stability metrics in the optimization

.. code-block:: python

    # Use both stability metrics for maximum robustness
    pipeline = FeatureSelectionPipeline(
        X=X, y=y,
        fs_methods=fs_methods,
        merging_strategy=merging_strategy,
        num_repeats=5,
        task="classification",
        num_features_to_select=10,
        stability_mode="all",
    )

Multiple Merging Strategies
---------------------------

You can pass multiple merging strategies to evaluate different combination approaches:

.. code-block:: python

    pipeline = FeatureSelectionPipeline(
        X=X, y=y,
        fs_methods=fs_methods,
        merging_strategy=["borda_merger", "union_of_intersections_merger"],
        num_repeats=5,
        task="classification",
        num_features_to_select=10,
    )

Extendability
-------------

Custom selectors can subclass ``FeatureSelector`` and implement ``compute_scores``:

.. code-block:: python

    import numpy as np
    from moosefs.feature_selectors.base_selector import FeatureSelector

    class MySelector(FeatureSelector):
        name = "MySelector"

        def __init__(self, task: str, num_features_to_select: int):
            super().__init__(task, num_features_to_select)

        def compute_scores(self, X, y):
            # return a 1D numpy array of per-feature scores
            return np.random.rand(X.shape[1])
