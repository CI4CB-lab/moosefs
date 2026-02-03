# merging_strategies/__init__.py

from .arithmetic_mean_merger import ArithmeticMeanMerger
from .base_merger import MergingStrategy
from .borda_merger import BordaMerger
from .consensus_merger import ConsensusMerger
from .frequency_bootstrap_merger import FrequencyBootstrapMerger
from .l2_norm_merger import L2NormMerger
from .union_of_intersections_merger import UnionOfIntersectionsMerger

__all__ = [
    "MergingStrategy",
    "BordaMerger",
    "UnionOfIntersectionsMerger",
    "ArithmeticMeanMerger",
    "L2NormMerger",
    "ConsensusMerger",
    "FrequencyBootstrapMerger",
]
