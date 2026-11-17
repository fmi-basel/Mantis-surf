
<img width="4616" height="2094" alt="Asset 22@600x" src="https://github.com/user-attachments/assets/5e99daf4-2af6-41a6-a686-6ba454eef5fe" />


<br> `Mantis-surf` is a Python package for 3D image segmentation, developed as part of the [Mantis ecosystem](https://github.com/fmi-basel/Mantis) for high-content, multiplexed, and volumetric microscopy image analysis.

`Mantis-surf` provides [Fractal](https://fractal-analytics-platform.github.io/) tasks for semantic segmentation using [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and conversion of semantic segmentation labels into individual object instances. It operates on OME-Zarr datasets and supports region-of-interest (ROI)-based processing.

## Functionality

### 1. nnU-Net semantic segmentation

The `nnunet_segmentation` task applies a user-trained nnUNet v2 model to 3D microscopy images. The task supports masked loading to segment selected image regions defined by ROI bounding boxes.

<img width="1321" height="391" alt="Screenshot 2026-11-17 at 21 46 31" src="https://github.com/user-attachments/assets/59e54b17-863c-4b41-bcf9-10e4c3979047" />

- **Input:** Single-channel 3D image (ZYX) from an OME-Zarr image, with a ROI table defining the regions to segment.
- **Output:** Semantic segmentation label image written to the OME-Zarr image, with an optional corresponding ROI table containing bounding boxes of segmented objects.


### 2. Semantic-to-instance segmentation

The `semantic_to_instance_segmentation` task converts nnU-Net semantic segmentation into individual object instances using distance-transform watershed and GASP agglomeration, following an approach similar to [PanSeg](https://kreshuklab.github.io/panseg/). The task supports ROI-based processing.

<img width="1323" height="389" alt="Screenshot 2026-11-17 at 21 46 44" src="https://github.com/user-attachments/assets/ee5d3ad9-a489-4c15-b4ad-cccc42492ee0" />

- **Input:** Two 3D label images (ZYX) containing semantic class and boundary segmentations, together with an ROI table defining the regions to process.
- **Output:** Instance segmentation label image with unique object IDs written to the OME-Zarr image, with an optional corresponding ROI table containing bounding boxes of segmented objects.
  

## Installation

Mantis-surf requires Python 3.11 or newer. The nnU-Net and PyTorch dependencies may require separate installation depending on the computational environment.

A Conda-based installation workflow is:

```bash
conda create -n mantis-surf python=3.11
conda activate mantis-surf

conda install -c conda-forge python-elf
pip install torch torchvision
pip install nnunetv2
pip install -e ".[dev]"
```

## License

BSD 3-Clause License.
