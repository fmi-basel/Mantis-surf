### Purpose
- Run 3D nnUnet V2 semantic segmentation on channel images given a user-trained model.

### Outputs
- Semantic segmentation label image and corresponding ROI table.

### Limitations
- Currently only supports 3D segmentation, not 2D.
- Follows Cellpose task structure and dependent on fractal tasks core masking.
