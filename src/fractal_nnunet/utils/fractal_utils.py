# Copyright 2025 (C) Friedrich Miescher Institute for Biomedical Research
#
# Original authors:
# Nicole Repina <nicole.repina@fmi.ch>
#

import logging

from fractal_tasks_core.channels import (ChannelInputModel,
                                         OmeroChannel,
                                         ChannelNotFoundError,
                                         get_channel_from_image_zarr)
from typing import Union

logger = logging.getLogger(__name__)


class UnetChannelInputModel(ChannelInputModel):
    """
    Channel input for cellpose with normalization options.

    Attributes:
        wavelength_id: Unique ID for the channel wavelength, e.g. `A01_C01`.
            Can only be specified if label is not set.
        label: Name of the channel. Can only be specified if wavelength_id is
            not set.
    """

    def get_omero_channel(self, zarr_url) -> Union[OmeroChannel, None]:
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
                f"Original error: {str(e)}"
            )
            return None

