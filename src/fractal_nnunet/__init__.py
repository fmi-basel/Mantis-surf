"""
Semantic segmentation with nnUnet.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal-nnunet")
except PackageNotFoundError:
    __version__ = "uninstalled"