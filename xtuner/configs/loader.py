"""Load Python configs with MMEngine 0.10.5 on Python 3.11.

Python 3.11 may report stdlib modules such as os with origin='frozen'.
MMEngine's path-based check misclassifies them as third-party lazy objects,
so calls such as os.getenv fail. Keep MMEngine's lazy config support while
recognizing the interpreter's own standard-library module list.
"""

import sys
import types

from mmengine.config import config as config_module
from mmengine.config import utils as config_utils


def load_config(filename: str) -> config_module.Config:
    """Read a config while correctly recognizing frozen stdlib imports."""
    original_utils_check = config_utils._is_builtin_module
    original_config_check = config_module._is_builtin_module

    def is_builtin(module_name: str) -> bool:
        if module_name.split('.')[0] in sys.stdlib_module_names:
            return True
        return original_utils_check(module_name)

    config_utils._is_builtin_module = is_builtin
    config_module._is_builtin_module = is_builtin
    try:
        cfg = config_module.Config.fromfile(filename)
        # Eager stdlib modules are execution helpers, not config fields.
        # Runner deep-copies the config and module objects are not picklable.
        for key, value in list(dict.items(cfg._cfg_dict)):
            if isinstance(value, types.ModuleType):
                del cfg._cfg_dict[key]
        return cfg
    finally:
        config_utils._is_builtin_module = original_utils_check
        config_module._is_builtin_module = original_config_check
