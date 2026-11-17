"""
Semantic segmentation with nnUnet.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mantis-surf")
except PackageNotFoundError:
    __version__ = "uninstalled"
