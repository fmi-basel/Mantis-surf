"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import ParallelTask

AUTHORS = "Nicole Repina"
DOCS_LINK = None
INPUT_MODELS = []

TASK_LIST = [
    ParallelTask(
        name="nnUnet Segmentation",
        executable="nnunet_segmentation.py",
        meta={"cpus_per_task": 1, "mem": 8000, "needs_gpu": True},
        category="Segmentation",
        tags=[
            "Semantic Segmentation",
        ],
        docs_info="file:docs_info/nnunet_segmentation.md",
    ),
    ParallelTask(
        name="nnUnet Semantic to Instance Segmentation ",
        executable="semantic_to_instance_segmentation.py",
        meta={"cpus_per_task": 1, "mem": 8000, "needs_gpu": True},
        category="Segmentation",
        tags=[
            "Instance Segmentation",
        ],
        docs_info="file:docs_info/semantic_to_instance_segmentation.md",
    ),
]
