from typing import Iterable, Optional

import numpy as np

from .base_merger import MergingStrategy


class FrequencyBootstrapMerger(MergingStrategy):
    """Set-based merger that keeps features with high selection frequency.

    For each selector, selection counts are aggregated across bootstrap runs.
    For each group, per-feature frequency is averaged across the group's selectors,
    then the top-k features (and optionally a minimum frequency) are retained.
    """

    name = "FrequencyBootstrap"

    def __init__(
        self,
        num_bootstrap: int = 20,
        min_freq: float = 0.6,
        *,
        use_scores: bool = True,
        fill: bool = False,
    ) -> None:
        super().__init__("set-based")
        self.num_bootstrap = int(num_bootstrap)
        self.min_freq = float(min_freq)
        self.use_scores = use_scores
        self.fill = fill
        # Flag checked by the pipeline to trigger bootstrap collection
        self.needs_bootstrap_merging = True

    # ------------------------------------------------------------------
    def merge(
        self,
        subsets: list,
        num_features_to_select: Optional[int] = None,
        **kwargs,
    ) -> list:
        """Merge based on bootstrap selection frequency.

        Args:
            subsets: Unused but kept for API compatibility.
            num_features_to_select: Number of features to keep.
            **kwargs:
                group: Tuple of selector names in this subgroup.
                feature_names: List of feature names aligned to counts.
                bootstrap_stats: Dict mapping selector name -> counts/score_sums/n_runs.
                fill: If True, backfill to requested size even when below `min_freq`.

        Returns:
            Ordered list of merged feature names.
        """
        group: Iterable = kwargs.get("group")
        feature_names = kwargs.get("feature_names")
        bootstrap_stats = kwargs.get("bootstrap_stats")
        fill = kwargs.get("fill", self.fill)

        if group is None or feature_names is None or bootstrap_stats is None:
            raise ValueError("FrequencyBootstrapMerger requires group, feature_names, and bootstrap_stats.")
        if num_features_to_select is None:
            raise ValueError("`num_features_to_select` must be provided.")

        n_features = len(feature_names)
        if n_features == 0:
            return []

        freq_sum = np.zeros(n_features, dtype=np.float32)
        count_sum = np.zeros(n_features, dtype=np.float32)
        score_sum = np.zeros(n_features, dtype=np.float32)
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
                raise ValueError("Bootstrap counts length does not match feature names length.")
            freq_sum += counts / float(n_runs)
            count_sum += counts

            if self.use_scores:
                score_sum += np.asarray(stats.get("score_sums", []), dtype=np.float32)

            num_methods += 1

        if num_methods == 0:
            return []

        freq = freq_sum / float(num_methods)
        if self.use_scores:
            avg_scores = np.divide(
                score_sum,
                count_sum,
                out=np.zeros_like(score_sum),
                where=count_sum > 0,
            )
        else:
            avg_scores = np.zeros_like(freq)

        candidates = [(name, float(f), float(s)) for name, f, s in zip(feature_names, freq, avg_scores)]
        candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)

        filtered = [name for name, f, _ in candidates if f >= self.min_freq]
        filtered = filtered[:num_features_to_select]

        if not fill:
            return filtered

        if len(filtered) >= num_features_to_select:
            return filtered

        # Backfill with remaining highest-frequency features to reach desired count.
        extras = [name for name, _, _ in candidates if name not in filtered][: num_features_to_select - len(filtered)]
        return filtered + extras
