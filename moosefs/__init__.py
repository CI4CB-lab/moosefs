# simplifying basic import
from importlib.metadata import PackageNotFoundError, version

from .feature_selection_pipeline import FeatureSelectionPipeline

try:
    __version__ = version("moose-fs")
except PackageNotFoundError:  # running from a checkout without installation
    __version__ = "0.0.0.dev0"

__all__ = ["FeatureSelectionPipeline", "__version__"]
