import json
from pathlib import Path

import mantis_surf

PACKAGE = "mantis_surf"
PACKAGE_DIR = Path(mantis_surf.__file__).parent
MANIFEST_FILE = PACKAGE_DIR / "__FRACTAL_MANIFEST__.json"
with MANIFEST_FILE.open("r") as f:
    MANIFEST = json.load(f)
    TASK_LIST = MANIFEST["task_list"]
