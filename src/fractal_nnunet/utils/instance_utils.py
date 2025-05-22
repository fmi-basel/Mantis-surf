# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#
"""Helper functions for running instance segmentation based on nnUnet output."""

import logging

import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries

from fractal_nnunet.utils.fractal_utils import remove_channel_dimension
from fractal_nnunet.utils.plantseg_utils import dt_watershed, gasp

logger = logging.getLogger(__name__)


def check_if_binary(a: np.ndarray):
    is_binary = np.array_equal(a, a.astype(bool))
    if not is_binary:
        raise ValueError("Input array must be binary (0 or 1).")

    return


def anisotropic_gaussian_blur(
    seg_binary: np.ndarray,
    sigma: int | float,
    pixmeta: list[float],
    convert_to_8bit: bool = True,
) -> tuple[np.ndarray, tuple[int | float]]:
    """Perform gaussian blur of binary 3D image with anisotropic sigma. Sigma anisotropy calculated from pixel spacing.
    Pixmeta is in order z,y,x.
    """
    check_if_binary(seg_binary)

    # Scale sigma to match pixel anisotropy
    spacing = (
        np.array(pixmeta) / pixmeta[1]
    )  # (z, y, x) where x,y is normalized to 1 e.g. (3, 1, 1)
    anisotropic_sigma = tuple(sigma * (spacing[1] / x) for x in spacing)
    # Perform 3D gaussian blur
    # Output image has value range 0-1 (not always max 1)
    if convert_to_8bit:
        # Convert binary segmentation into 8-bit image
        seg_fill_8bit = (seg_binary * 255).astype(np.uint8)
        blurred = gaussian(seg_fill_8bit, sigma=anisotropic_sigma, preserve_range=False)
    else:
        blurred = gaussian(seg_binary, sigma=anisotropic_sigma, preserve_range=True)

    return blurred, anisotropic_sigma


def select_label(seg: np.ndarray, label_str: str) -> np.ndarray:
    """Select given label from segmentation image and return as binary (0,1) mask."""
    seg_sel = seg.copy()
    # Select label that corresponds to current object, set all other objects to 0
    seg_sel[seg_sel != float(label_str)] = 0
    # Binarize object segmentation image
    seg_sel[seg_sel > 0] = 1

    return seg_sel


def assign_semantic_classes_to_gasp(
    gasp_result: np.ndarray,
    semantic_seg: np.ndarray,
) -> dict:
    """Given GASP instance segmentation, assign each instance to a semantic class by determining which class
    has maximal pixel overlap with the GASP label (major fraction).
    Returns dictionary where key is the GASP label, value is the assigned semantic class.
    """
    instances = np.unique(gasp_result)

    # key is GASP label, value is maximally matching semantic class
    label_mapping = {}

    # determine the semantic class each GASP label belongs to using maximum pixel overlap
    for label in instances:
        # select region of semantic segmentation that corresponds to a given gasp label
        selected_sem = semantic_seg[gasp_result == label]

        # find pixel counts for each semantic class present in region
        unique_vals, counts = np.unique(selected_sem, return_counts=True)

        # normalize counts by total, to determine fraction of pixels in each semantic class
        norm_counts = counts / counts.sum()
        # find semantic class maximally represented in region
        max_index = np.argmax(counts)
        max_semantic_class = unique_vals[max_index]

        # raise warning if more than 10% of pixels are from more than one class - this might mean the classification
        # is potentially ambiguous
        if norm_counts[max_index] < 0.9:
            logger.warning(
                f"GASP label {label} has more than 10% of pixels belong to more than one semantic class. "
                f"Detected classes: {unique_vals} with normalized counts: {norm_counts}.')"
            )

        label_mapping[label] = max_semantic_class

    return label_mapping


def relabel_by_class_instances(
    gasp_result: np.ndarray,
    label_mapping: dict,
    select_semantic_class: int,
    starting_label_id: int = 1,
) -> tuple[np.ndarray, int]:
    """Given GASP instance segmentation, select GASP labels that belong to a specific semantic class and
    relabel the instance segmentation in numerically increasing order, starting from 1. Background is set to 0.
    Returns relabeled GASP image and the number of instances assigned.
    """
    # make reassignment dictionary
    # key is GASP label, value is the instance label ID it should be reassigned to
    reassign_labels = {}
    # first instance has label equal to starting_label_id. subsequent instances have sequentially
    # increasing integer values.
    i = starting_label_id
    for label, semantic_id in label_mapping.items():
        if semantic_id == select_semantic_class:
            reassign_labels[label] = i
            i += 1
        else:
            # set background to 0
            reassign_labels[label] = 0

    gasp_result_relabeled = gasp_result.copy()

    # iterate over GASP labels and set region to assigned instance label ID
    for label, semantic_id in reassign_labels.items():
        gasp_result_relabeled[gasp_result_relabeled == label] = semantic_id

    instance_count = i - starting_label_id
    return gasp_result_relabeled, instance_count


def boundaries_to_gasp(
    boundary_seg: np.ndarray,
    gaussian_sigma: int | float,
    pixmeta: list[float],
    watershed_threshold: float,
    watershed_sigma_seeds: float,
    watershed_sigma_weights: float,
    watershed_alpha: float,
    watershed_min_size: int,
    gasp_beta: float,
    semantic_seg: np.ndarray | None = None,
    select_semantic_class: int | None = None,
) -> np.ndarray | None:
    """Generate instance segmentation from boundary segmenation image using distance transform watershed and GASP.
    Input is the boundary segmentation (binary, 0 or 1), e.g. output from nnUnet boundary semantic segmentation.
    If a semantic class segmentation image (semantic_seg) is optionally provided, its boundaries are combined with the
    boundary segmentation for increased robustness.
    Output is GASP output as numpy array.
    """
    # if provided, combine boundary and semantic segmentation into single image
    if semantic_seg is not None:
        if select_semantic_class is None:
            raise ValueError(
                "Label to select from semantic class segmentation must be provided."
            )

        semantic_selected = select_label(semantic_seg, str(select_semantic_class))
        semantic_boundaries = find_boundaries(
            semantic_selected, connectivity=1, mode="thick"
        )
        boundaries = np.logical_or(semantic_boundaries, boundary_seg).astype(np.uint8)
    else:
        boundaries = boundary_seg

    # check that boundaries are binary (0 or 1).
    check_if_binary(boundaries)

    # if boundary image is empty, return None and skip processing this region
    if not np.any(boundaries):
        return None

    # Blur boundaries to make probability-map-like image
    blurred_boundaries, anisotropic_sigma = anisotropic_gaussian_blur(
        boundaries, sigma=gaussian_sigma, pixmeta=pixmeta, convert_to_8bit=True
    )
    superpixels = dt_watershed(
        boundary_pmaps=blurred_boundaries,
        threshold=watershed_threshold,
        sigma_seeds=watershed_sigma_seeds,
        sigma_weights=watershed_sigma_weights,
        min_size=watershed_min_size,
        alpha=watershed_alpha,
        pixel_pitch=tuple(pixmeta),
        apply_nonmax_suppression=True,
    )

    gasp_result = gasp(
        boundary_pmaps=blurred_boundaries, superpixels=superpixels, beta=gasp_beta
    )

    return gasp_result


def run_semantic_to_instance_segmentation(
    boundary_seg: np.ndarray,
    semantic_seg: np.ndarray,
    select_semantic_class: int,
    pixmeta: list[float],
    starting_label_id: int = 1,
    gaussian_sigma: int | float = 1,
    watershed_threshold: float = 0.05,
    watershed_sigma_seeds: float = 1.0,
    watershed_sigma_weights: float = 2.0,
    watershed_alpha: float = 1,
    watershed_min_size: int = 100,
    gasp_beta: float = 0.5,
    combine_boundary_and_semantic_edges: bool = True,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    # masked loading with fractal tasks core requires channel dim, but subsequent processing steps fail; temp fix here.
    # add back channel dimension at end
    # clean up this mess with ngio
    # TODO: this only works for 3D images; task needs to be updated to handle 2D.
    boundary_seg = remove_channel_dimension(boundary_seg)
    semantic_seg = remove_channel_dimension(semantic_seg)

    if combine_boundary_and_semantic_edges:
        semantic_seg_for_gasp = semantic_seg
    else:
        semantic_seg_for_gasp = None

    gasp_result = boundaries_to_gasp(
        boundary_seg=boundary_seg,
        gaussian_sigma=gaussian_sigma,
        pixmeta=pixmeta,
        watershed_threshold=watershed_threshold,
        watershed_sigma_seeds=watershed_sigma_seeds,
        watershed_sigma_weights=watershed_sigma_weights,
        watershed_alpha=watershed_alpha,
        watershed_min_size=watershed_min_size,
        gasp_beta=gasp_beta,
        semantic_seg=semantic_seg_for_gasp,
        select_semantic_class=select_semantic_class,
    )

    if gasp_result is None:
        return None, None

    label_mapping = assign_semantic_classes_to_gasp(gasp_result, semantic_seg)

    instance_seg, instance_count = relabel_by_class_instances(
        gasp_result,
        label_mapping,
        select_semantic_class,
        starting_label_id=starting_label_id,
    )
    return instance_seg, instance_count
