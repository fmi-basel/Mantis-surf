#
# Bundled from PlantSeg
# https://github.com/kreshuklab/plant-seg
# v.1.8.0
#
"""Watershed and GASP functions bundled from Plantseg,
for conversion of semantic to instance segmentation.
"""

import numpy as np
from elf.segmentation import GaspFromAffinities, stacked_watershed
from elf.segmentation.watershed import apply_size_filter, distance_transform_watershed


# From plantseg/functionals/segmentation/utils.py
def shift_affinities(affinities, offsets):
    rolled_affs = []
    for i, _ in enumerate(offsets):
        offset = offsets[i]
        shifts = tuple(int(off / 2) for off in offset)

        padding = [[0, 0] for _ in range(len(shifts))]
        for ax, shf in enumerate(shifts):
            if shf < 0:
                padding[ax][1] = -shf
            elif shf > 0:
                padding[ax][0] = shf

        padded_inverted_affs = np.pad(
            affinities, pad_width=((0, 0),) + tuple(padding), mode="constant"
        )

        crop_slices = tuple(
            slice(padding[ax][0], padded_inverted_affs.shape[ax + 1] - padding[ax][1])
            for ax in range(3)
        )

        padded_inverted_affs = np.roll(padded_inverted_affs[i], shifts, axis=(0, 1, 2))[
            crop_slices
        ]
        rolled_affs.append(padded_inverted_affs)
        del padded_inverted_affs

    rolled_affs = np.stack(rolled_affs)
    return rolled_affs


# from plant-seg/plantseg/functionals/segmentation/segmentation.py
def dt_watershed(
    boundary_pmaps: np.ndarray,
    threshold: float = 0.5,
    sigma_seeds: float = 1.0,
    stacked: bool = False,
    sigma_weights: float = 2.0,
    min_size: int = 100,
    alpha: float = 1.0,
    pixel_pitch: tuple[int | float, ...] | None = None,
    apply_nonmax_suppression: bool = False,
    n_threads: int | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Performs watershed segmentation using distance transforms on boundary probability maps.

    This function applies the distance transform watershed algorithm to the input boundary
    probability maps, either slice-by-slice or in original shape depending on the 'stacked' parameter.
    The watershed method is applied to the boundary probability maps with optional pre-processing
    like thresholding, smoothing, and masking.

    Args:
        boundary_pmaps (np.ndarray): Input array of boundary probability maps, often obtained
            from deep learning models. Each pixel/voxel value represents the probability of
            being part of a boundary.
        threshold (float, optional): Threshold value applied to the boundary probability map
            before computing the distance transform. Values below this threshold are considered
            background. Defaults to 0.5.
        sigma_seeds (float, optional): Standard deviation for Gaussian smoothing applied to
            the seed map (used for initializing the watershed regions). Higher values result
            in more smoothed seeds. Defaults to 1.0.
        stacked (bool, optional): If True, performs watershed segmentation on each 2D slice of
            a 3D volume independently (slice-by-slice). If False, performs watershed segmentation
            in 3D for volumetric data or 2D for 2D input. Defaults to False.
        sigma_weights (float, optional): Standard deviation for Gaussian smoothing applied
            to the weight map. The weight map combines the distance transform and input map
            to guide the watershed. Larger values result in smoother weight maps. Defaults to 2.0.
        min_size (int, optional): Minimum size of the segmented regions to retain. Regions
            smaller than this size are removed. Defaults to 100.
        alpha (float, optional): Blending factor to combine the input boundary probability maps
            and the distance transform when constructing the weight map. A higher alpha
            prioritizes the input maps. Defaults to 1.0.
        pixel_pitch (Optional[tuple[int, ...]], optional): Voxel anisotropy factors (e.g., spacing
            along different axes) to use during the distance transform. If None, the distances are
            computed isotropically. For anisotropic volumes, this should match the voxel spacing.
            Defaults to None.
        apply_nonmax_suppression (bool, optional): If True, applies non-maximum suppression to
            the detected seeds, reducing seed redundancy. This requires the Nifty library.
            Defaults to False.
        n_threads (Optional[int], optional): Number of threads to use for parallel processing in
            2D mode (stacked mode). If None, the default number of threads will be used.
            Defaults to None.
        mask (Optional[np.ndarray], optional): A binary mask that excludes certain regions from
            segmentation. Only regions within the mask will be considered. If None, all regions
            are included. Must have the same shape as 'boundary_pmaps'. Defaults to None.

    Returns:
        np.ndarray: A labeled segmentation map where each region is assigned a unique label.

    """
    # Prepare the keyword arguments for the watershed function
    boundary_pmaps = boundary_pmaps.astype("float32")
    ws_kwargs = {
        "threshold": threshold,
        "sigma_seeds": sigma_seeds,
        "sigma_weights": sigma_weights,
        "min_size": min_size,
        "alpha": alpha,
        "pixel_pitch": pixel_pitch,
        "apply_nonmax_suppression": apply_nonmax_suppression,
        "mask": mask,
    }
    if stacked:
        # Apply watershed slice by slice (for 3D data)
        segmentation, _ = stacked_watershed(
            boundary_pmaps,
            ws_function=distance_transform_watershed,
            n_threads=n_threads,
            **ws_kwargs,
        )
    else:
        # Apply watershed in 3D for 3D data or in 2D for 2D data
        segmentation, _ = distance_transform_watershed(boundary_pmaps, **ws_kwargs)

    return segmentation


# from plant-seg/plantseg/functionals/segmentation/segmentation.py
def gasp(
    boundary_pmaps: np.ndarray,
    superpixels: np.ndarray | None = None,
    gasp_linkage_criteria: str = "average",
    beta: float = 0.5,
    post_minsize: int = 100,
    n_threads: int = 6,
) -> np.ndarray:
    """Perform segmentation using the GASP algorithm with affinity maps.

    Args:
        boundary_pmaps (np.ndarray): Cell boundary prediction.
        superpixels (Optional[np.ndarray]): Superpixel segmentation. If None,
            GASP will be run from the pixels. Default is None.
        gasp_linkage_criteria (str): Linkage criteria for GASP. Default is 'average'.
        beta (float): Beta parameter for GASP. Small values steer towards under-segmentation,
            while high values bias towards over-segmentation. Default is 0.5.
        post_minsize (int): Minimum size of the segments after GASP. Default is 100.
        n_threads (int): Number of threads used for GASP. Default is 6.

    Returns:
        np.ndarray: GASP output segmentation.
    """
    remove_singleton = False
    if superpixels is not None:
        assert (
            boundary_pmaps.shape == superpixels.shape
        ), "Shape mismatch between boundary_pmaps and superpixels."
        if superpixels.ndim == 2:  # Ensure superpixels is 3D if provided
            superpixels = superpixels[None, ...]
            boundary_pmaps = boundary_pmaps[None, ...]
            remove_singleton = True

    # Prepare the arguments for running GASP
    run_GASP_kwargs = {
        "linkage_criteria": gasp_linkage_criteria,
        "add_cannot_link_constraints": False,
        "use_efficient_implementations": False,
    }

    # Interpret boundary_pmaps as affinities and prepare for GASP
    boundary_pmaps = boundary_pmaps.astype("float32")
    affinities = np.stack([boundary_pmaps] * 3, axis=0)

    offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    # Shift is required to correct aligned affinities
    affinities = shift_affinities(affinities, offsets=offsets)

    # invert affinities
    affinities = 1 - affinities

    # Initialize and run GASP
    gasp_instance = GaspFromAffinities(
        offsets,
        superpixel_generator=None
        if superpixels is None
        else (lambda *args, **kwargs: superpixels),
        run_GASP_kwargs=run_GASP_kwargs,
        n_threads=n_threads,
        beta_bias=beta,
    )
    segmentation, _ = gasp_instance(affinities)

    # Apply size filtering if specified
    if post_minsize > 0:
        segmentation, _ = apply_size_filter(
            segmentation.astype("uint32"), boundary_pmaps, post_minsize
        )

    if remove_singleton:
        segmentation = segmentation[0]

    return segmentation
