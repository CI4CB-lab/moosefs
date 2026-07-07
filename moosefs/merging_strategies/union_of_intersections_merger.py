from itertools import combinations
from typing import Optional

from .base_merger import MergingStrategy


class UnionOfIntersectionsMerger(MergingStrategy):
    """Union of intersections across selector subsets."""

    name = "UnionOfIntersections"

    def __init__(self) -> None:
        super().__init__("set-based")

    def merge(
        self,
        subsets: list,
        num_features_to_select: Optional[int] = None,
        fill: bool = False,
        **kwargs,
    ) -> set:
        """Merge by union of pairwise intersections.

        Args:
            subsets: Feature lists (one list per selector).
            num_features_to_select: Required when ``fill=True``.
            fill: If True, trim/pad output to requested size.
            **kwargs: Unused.

        Returns:
            Set of selected feature names.

        Raises:
            ValueError: If inputs are invalid or size is missing when ``fill=True``.
        """
        self._validate_input(subsets)

        if fill and num_features_to_select is None:
            raise ValueError("`num_features_to_select` must be provided when `fill=True`.")

        if len(subsets) == 1:
            if not fill:
                return {f.name for f in subsets[0]}
            top = sorted(subsets[0], key=lambda f: f.score, reverse=True)
            return {f.name for f in top[:num_features_to_select]}

        # Compute core as the union of pairwise intersections
        name_sets = [{f.name for f in subset} for subset in subsets]
        core = set().union(*[a & b for a, b in combinations(name_sets, 2)])

        if not fill:
            return core  # Return raw core without enforcing `num_features_to_select`

        # Global feature scores: sum of per-selector min-max-normalized scores
        feature_names, scores = self._aligned_scores(subsets)
        totals = self._normalize_scores(scores).sum(axis=1)
        feature_score_map = dict(zip(feature_names, totals))

        # Prune or fill to get exactly `num_features_to_select`
        core_list = sorted(core, key=lambda x: feature_score_map[x], reverse=True)
        core_size = len(core_list)

        if core_size >= num_features_to_select:
            return set(core_list[:num_features_to_select])

        # Fill with highest-ranked extra features
        extras = sorted(feature_score_map.keys(), key=lambda x: feature_score_map[x], reverse=True)
        extras = [f for f in extras if f not in core][: num_features_to_select - core_size]

        return set(core_list + extras)
