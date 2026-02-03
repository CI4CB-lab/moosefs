from moosefs.core.novovicova import StabilityNovovicova
from moosefs.metrics.stability_metrics import compute_stability_metrics


def test_compute_stability_metrics_matches_core():
    selectors = [["a", "b"], ["b", "c"], ["a", "c"]]
    # wrapper should match direct computation
    expected = StabilityNovovicova([set(s) for s in selectors]).compute_stability()
    got = compute_stability_metrics(selectors)
    assert got == expected
