### Purpose
- Convert semantic segmentation output of nnUnet to instance segmentation.
- Select which semantic class is converted to numerically increasing instance labels.
- Based on distance watershed and GASP algorithms similar to PlantSeg implementation.

### Outputs
- Instance segmentation label image (numerically increasing order starting from 1) and corresponding ROI table.

### Limitations
- Currently only supports 3D segmentation, not 2D.
- Follows Cellpose task structure and dependent on fractal tasks core masking.
