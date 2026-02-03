from moosefs.core.novovicova import StabilityNovovicova


def compute_stability_metrics(features_list: list) -> float:
    """Compute stability SH(S) across selections.

    Args:
        features_list: Selected feature names per selector.

    Returns:
        Stability in [0, 1].
    """
    return StabilityNovovicova(features_list).compute_stability()


def _jaccard(a: set, b: set) -> float:
    """Return Jaccard similarity, handling empty sets as 1.0 if both empty."""
    return len(a & b) / len(a | b) if a | b else 1.0
