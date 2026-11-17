# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#
"""Helper functions Fractal task."""

import logging

import numpy as np
from fractal_tasks_core.channels import (
    ChannelInputModel,
    ChannelNotFoundError,
    OmeroChannel,
    get_channel_from_image_zarr,
)

logger = logging.getLogger(__name__)


class UnetChannelInputModel(ChannelInputModel):
    """Channel input for cellpose with normalization options.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
    """

    def get_omero_channel(self, zarr_url) -> OmeroChannel | None:
        try:
            return get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=self.wavelength_id,
                label=self.label,
            )
        except ChannelNotFoundError as e:
            logger.warning(
                f"Channel with wavelength_id: {self.wavelength_id} "
                f"and label: {self.label} not found, exit from the task.\n"
                f"Original error: {e!s}"
            )
            return None


def remove_channel_dimension(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 4:
        # if first dim is channel, remove it
        img = np.squeeze(
            img, axis=0
        )  # axis=0 removes first dimension only if its size is 1.
    return img
