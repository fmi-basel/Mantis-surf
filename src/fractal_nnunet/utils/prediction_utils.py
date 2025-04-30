# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#
"""Helper functions for prediction with nnUnet."""

from typing import Literal

import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def log_transform_array(
    img: np.ndarray,
) -> np.ndarray:
    """Perform log transform of input image. Return same dtype as input array."""
    img_to_transform = img.copy()
    dtype = img_to_transform.dtype
    dtype_max = np.iinfo(dtype).max  # e.g. 65535 for uint16
    # add 1 to all values below max for dtype to avoid divide by 0 error
    img_to_transform[img_to_transform < dtype_max] += 1
    img_log = np.log(img_to_transform)  # returns float numpy array

    # rescale to input range and dtype
    img_log = dtype_max * img_log / (np.log(dtype_max + 1))
    img_log = img_log.astype(dtype)

    return img_log


def transpose_3dimage_for_nnunet(
    img: np.ndarray,
    img_dtype: np.dtype | None = None,
    add_channel_axis: bool = False,
) -> np.ndarray:
    """Transpose ZYX 3D image to match required nnUnet axis order: CXYZ"""
    if img.ndim == 3:
        img_transposed = img.transpose(
            2, 1, 0
        )  # Transposes the array from (z, y, x) to (x, y, z)
        if add_channel_axis:
            img_transposed = img_transposed[
                np.newaxis, ...
            ]  # Add channel dimension to make it (1, X, Y, Z)
    elif img.ndim == 4 and img.shape[0] == 1:
        # input image already has channel axis
        img_transposed = img.transpose(0, 3, 2, 1)
    else:
        raise ValueError(f"Only 3D images supported. Image shape: {img.shape}")

    return img_transposed.astype(img_dtype)


def output_probability_from_nnunet(
    img: np.ndarray,
    ch_sel: Literal[0, 1],
    convert_to_uint16: bool = True,
) -> np.ndarray:
    """Convert probability map output from nnUnet into ZYX axis ordering."""
    if img.ndim == 4:
        img_transposed = img[
            ch_sel
        ]  # select first or second probability channel output 0 or 1
        img_transposed = img_transposed.transpose(
            2, 1, 0
        )  # Transposes the array from (z, y, x) to (x, y, z)
        if convert_to_uint16:
            # Convert to uint16 by scaling
            img_transposed = (img_transposed * 65535).astype(np.uint16)
    else:
        raise ValueError("Expecting 4D numpy array as probability map output.")

    return img_transposed


def transpose_spacing_for_nnunet(image_properties: dict) -> dict:
    """Transpose ZYX pixel spacing metadata to match required nnUnet axis order: XYZ"""
    spacing = image_properties["spacing"]
    spacing_transposed = [spacing[2], spacing[1], spacing[0]]
    image_properties_transposed = {"spacing": spacing_transposed}
    return image_properties_transposed


def init_predictor(
    model_path: str,
    use_folds: tuple = (0, 1, 2, 3, 4),
) -> nnUNetPredictor:
    """Initialize nnUnet V2 predictor."""
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=torch.cuda.is_available(),
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=False,
    )

    # load train model directory
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_path,
        use_folds=use_folds,  # You can use (0, 1, 2, 3, 4) for ensemble
    )
    return predictor


def predict_nnunet(
    img: np.ndarray,
    predictor: nnUNetPredictor,
    image_properties: dict,
    save_or_return_probabilities: bool = False,
    label_dtype: np.dtype | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Run nnUnet V2 prediction for a single numpy array."""
    img_transposed = transpose_3dimage_for_nnunet(
        img, img_dtype=np.float32, add_channel_axis=True
    )
    image_properties_transposed = transpose_spacing_for_nnunet(image_properties)

    ret = predictor.predict_single_npy_array(
        input_image=img_transposed,
        image_properties=image_properties_transposed,
        save_or_return_probabilities=save_or_return_probabilities,
    )
    if save_or_return_probabilities:
        return (
            transpose_3dimage_for_nnunet(ret[0], img_dtype=label_dtype),
            output_probability_from_nnunet(ret[1], ch_sel=1, convert_to_uint16=True),
        )
    else:
        return transpose_3dimage_for_nnunet(ret, img_dtype=label_dtype)
