from typing import Optional


class Feature:
    """Container for a single feature.

    Stores the feature name, an optional score, and whether it is selected.

    Args:
        name: Feature identifier (e.g., column name).
        score: Optional importance/score for ranking.
        selected: Whether the feature is selected.
    """

    __slots__ = ("name", "score", "selected")

    def __init__(self, name: str, score: Optional[float] = None, selected: bool = False) -> None:
        self.name: str = name
        self.score: Optional[float] = score
        self.selected: bool = selected

    def set_score(self, score: float) -> None:
        """Set the feature score.

        Args:
            score: Importance/score value.
        """
        self.score = score

    def set_selected(self, selected: bool) -> None:
        """Set the selected flag.

        Args:
            selected: True if selected; otherwise False.
        """
        self.selected = selected

    def __str__(self) -> str:
        """Return a readable string representation."""
        return f"Feature(name={self.name}, score={self.score}, selected={self.selected})"

    def __repr__(self) -> str:
        """Return an unambiguous representation for debugging."""
        return f"Feature('{self.name}', {self.score}, {self.selected})"
