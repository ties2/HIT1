"""
-----------------
**legacy** module
-----------------

This module contains the public legacy functionality.

>>> from common.elements.legacy import *

"""


from .loaders import (
    LoadClassMasksFile,
    LoadClassMasksFolder,
    LoadInstanceMasksFile,
    LoadInstanceMasksFolder,
    LoadBinMasksFile,
    CachedResource,
    LoadImageFn
)

from .dataset import (
    Dataset,
    DatasetInfo,
    DatasetDirs,
    get_dataset_info,
    get_datasets_root_dir,
)
