from mantis_surf.nnunet_segmentation import nnunet_segmentation
from mantis_surf.utils.fractal_utils import UnetChannelInputModel

channel = UnetChannelInputModel(wavelength_id="A04_C01")

# zarr_urls = ['/Users/repinico/Documents/Fractal/fractal_output/r2.zarr/A/01/0_fused']
# zarr_urls = ['/Users/nicolerepina/Documents/Fractal/fractal_output/r2.zarr/A/01/0_fused', '/Users/nicolerepina/Documents/Fractal/fractal_output/r2.zarr/A/01/1_fused']
zarr_urls = [
    "/Volumes/gliberal_scratch/Users/repinico/Yokogawa/20241201_DCFLEX/fractal_output/day4p5-more/r0.zarr/C/05/0_fused"
]

for img in zarr_urls:
    nnunet_segmentation(
        zarr_url=img,
        level=0,
        channel=channel,
        input_ROI_table="rdc2d_expanded_fused_ROI_table",
        output_ROI_table="delete_ROI_table",
        output_label_name="delete",
        model_path="/Volumes/tachyon/groups/scratch/gliberal/Users/repinico/Training/organoid3D_training/nnUnet/04/nnUNet_results/Dataset006_pooled/nnUNetTrainer__nnUNetPlans__3d_fullres",
        folds=(1,),
        log_transform_prior_to_prediction=True,
        use_masks=True,
        overwrite=True,
    )

# for img in zarr_urls:
#     semantic_to_instance_segmentation(
#         zarr_url=img,
#         level=0,
#         boundary_label_name= "nnbound",
#         semantic_label_name = "nnsem",
#         masking_ROI_table = "rdc2d_linked_expanded_fused_ROI_table",
#         output_ROI_table= "nnorg_ROI_table",
#         output_label_name= "nnorg",
#         select_semantic_class= 1,
#         gaussian_sigma= 2.0,
#         watershed_threshold= 0.05,
#         watershed_sigma_seeds = 3.0,
#         watershed_sigma_weights= 2.0,
#         watershed_alpha= 1.0,
#         watershed_min_size= 100,
#         gasp_beta= 0.65,
#         combine_boundary_and_semantic_edges = True,
#         overwrite= True,
#     )
