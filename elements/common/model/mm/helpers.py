import importlib
import hashlib
import tqdm
import os
import shutil
import sys
import yaml
import tempfile
import torch
import torch.nn

from tqdm import tqdm
from enum import Enum
from types import ModuleType
from yaml.loader import SafeLoader
from typing import List, Dict, Union, Any
from urllib.request import urlopen, Request

from common.elements.utils import get_tmp_dir
from mmengine.runner import load_checkpoint
from mmengine import Config, ConfigDict

os.environ["MM_CONFIG_FOLDER"] = "/media/public_data/pipeline/mmdet/mmdet-current/configs"


def _get_working_dir():
    return get_tmp_dir(os.path.splitext((os.path.basename(sys.modules[__name__].__file__)))[0])


class MMTask(Enum):
    SEGMENTATION = 0
    OBJECT_DETECTION = 1
    INSTANCE_SEGMENTATION = 2
    POSE_ESTIMATION = 3


class MMLibrary:
    mm_lib: Dict[str, Dict[str, Any]] = {
        "mmdetection": {
            "source": "https://github.com/open-mmlab/mmdetection",
            "module": "mmdet",
            "configs": {"faster_rcnn": {"path": "faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"},
                        "detr": {"path": "detr/detr_r50_8xb2-150e_coco.py"}, # does not work
                        "retinanet": {"path": "retinanet/retinanet_r50_fpn_1x_coco.py"}},
            "mmtask": MMTask.OBJECT_DETECTION
        },
        "mmdetection_instance_segmentation": {
            "source": "https://github.com/open-mmlab/mmdetection",
            "module": "mmdet",
            "configs": {"mask_rcnn_r50": {"path": "mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py"}
                        },
            "mmtask": MMTask.INSTANCE_SEGMENTATION
        },
        "mmpose": {
            "source": "https://github.com/open-mmlab/mmdetection",
            "module": "mmpose",
            "configs": {"faster_rcnn": {"path": "faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py"}
                        },
            "mmtask": MMTask.POSE_ESTIMATION
        },
        "swin_segmentation": {
            "source": "https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation",
            "module": "mmseg",
            "configs": {"swin_base":  {"path": "swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py"},
                        "swin_small": {"path": "swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py"},
                        "swin_tiny":  {"path": "swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py"}
                    },
            "mmtask": MMTask.SEGMENTATION
        },
        "swin_object_detection": {
            "source": "https://github.com/SwinTransformer/Swin-Transformer-Object-Detection",
            "module": "mmdet",
            "configs": {"swin_base":  {"path": "swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"},
                        "swin_small": {"path": "swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"},
                        "swin_tiny":  {"path": "swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py"},
                        "swin_tiny2": {"path": "swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py"}
                        },
            "mmtask": MMTask.OBJECT_DETECTION
        },
        "swin_instance_segmentation": {
            "source": "https://github.com/SwinTransformer/Swin-Transformer-Object-Detection",
            "module": "mmdet",
            "configs": {"swin_base":  {"path": "swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py"},
                        "swin_tiny":  {"path": "swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py"},
                        "swin_tiny2": {"path": "swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py"}
                        },
            "mmtask": MMTask.INSTANCE_SEGMENTATION
        },
        "mmsegmentation": {
            "source": "https://github.com/open-mmlab/mmsegmentation",
            "module": "mmseg",
            "configs": {"deeplabv3": {"path": "deeplabv3/deeplabv3_r101-d16-mg124_512x1024_40k_cityscapes.py"},
                        "unet": {"path": "unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py"}},
            # "configs": {"deeplabv3": {"path": "deeplabv3/deeplabv3_r18-d8_512x1024_80k_cityscapes.py"}},
            "mmtask": MMTask.SEGMENTATION
        },
    }

    def __init__(self):
        """
        Library of MM repos that are registered
        """
        return

    def get_library_keys(self) -> List[str]:
        """Get the available repos"""
        return list(self.mm_lib.keys())

    def get_mm_config_keys(self, lib_key) -> List[str]:
        """Get the available configs of a repo"""
        return list(self.mm_lib[lib_key]["configs"].keys())

    def get_mm_task(self, lib_key) -> MMTask:
        """Get the type of task"""
        return self.mm_lib[lib_key]["mmtask"]

    def get_mm_module(self, lib_key):
        """Get the folder of the MM module"""
        return self.mm_lib[lib_key]["module"]

    def get_mm_configs_folder(self, lib_key):
        """Get the folder of the MM module's configs"""
        if not "MM_CONFIG_FOLDER" in os.environ.keys():
            print("Folder of the configuration files as environment variable MM_CONFIG_FOLDER not specified")
            path = "/media/public_data/mmdet-31-10-23/mmdet/configs"
        else:
            path = os.environ["MM_CONFIG_FOLDER"]
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Config folder {path} does not exist")
        return path

    def get_mm_config_path(self, lib_key, cfg_key):
        """Get the folder of the MM module's config file path"""
        path = os.path.abspath(os.path.join(self.get_mm_configs_folder(lib_key), self.mm_lib[lib_key]["configs"][cfg_key]["path"]))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file {path} does not exist")
        return path


def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

    """
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


class MMHelper:
    mm: ModuleType
    mm_models: ModuleType
    configs_path: str

    def __init__(self, module: str, configs_path: str):
        """"Helper for working for MM repositories or forks"""
        self.mm = importlib.import_module(module)
        self.mm_models = importlib.import_module(".".join([module, "registry"]))
        self.configs_path = configs_path

    def get_mm_module(self) -> ModuleType:
        return self.mm

    def get_config(self, config_path: str) -> Config:
        return Config.fromfile(config_path)

    def build_model(self, config: Config, mm_task: MMTask) -> torch.nn.Module:
        if mm_task == MMTask.OBJECT_DETECTION or mm_task == MMTask.INSTANCE_SEGMENTATION:
            model = self.mm_models.MODELS.build(config["model"])
        elif mm_task == MMTask.SEGMENTATION:
            model = self.mm_models.MODELS.build(config["model"])
        elif mm_task == MMTask.POSE_ESTIMATION:
            model = self.mm_models.MODELS.build(config["model"])
        else:
            raise RuntimeError(f"Unknown task {mm_task}")
        model.cfg = config
        return model

    def get_options_by_name(self, name: str, config: Config) -> Dict[str, Any]:
        def next_key(k, v: Union[ConfigDict, list], root: str = ""):
            if k == name:
                options[root[1:]] = v
            if isinstance(v, ConfigDict):
                for _k, _v in v.items():
                    next_key(_k, _v, f"{root}.{str(_k)}")
            elif isinstance(v, list):
                for _k, _v in enumerate(v):
                    next_key(_k, _v, f"{root}.{str(_k)}")
        options = {}
        next_key("", config._cfg_dict, "")
        return options

    def set_options(self, config: Config, name: str, value: Any):
        options = self.get_options_by_name(name, config)
        options = {k: value for k in options.keys()}
        config.merge_from_dict(options)
        return config

    def get_mean(self, config: Config, option_name: str="img_norm_cfg"):
        return config[option_name]["mean"]

    def get_std(self, config: Config, option_name: str = "img_norm_cfg"):
        return config[option_name]["std"]


def create_mm_model(library: str, config: str, num_classes: int, working_dir=_get_working_dir(),  pre_trained: bool=False, dev: str = "cuda:0", verbose=False) -> torch.nn.Module:
    mm_lib = MMLibrary()
    lib_key = library
    config_key = config

    if verbose:
        lib_keys = mm_lib.get_library_keys()
        print(f"Available libraries are: {lib_keys}, using {lib_key}")

        configs = mm_lib.get_mm_config_keys(lib_key)
        print(f"Available configs for {lib_key} are: {configs}, using {config_key}")

    mm_helper = MMHelper(module=mm_lib.get_mm_module(lib_key=lib_key), configs_path=mm_lib.get_mm_configs_folder(lib_key=lib_key))
    config = mm_helper.get_config(config_path=mm_lib.get_mm_config_path(lib_key=lib_key, cfg_key=config_key))

    config = mm_helper.set_options(config, "num_classes", num_classes)
    model = mm_helper.build_model(config, mm_task=mm_lib.get_mm_task(lib_key=lib_key))

    if pre_trained:
        h, t = os.path.split(config.filename)
        filename = os.path.join(h, "metafile.yml")
        with open(filename) as f:
            data = yaml.load(f, Loader=SafeLoader)
        for model_info in data['Models']:
            if t[:-4] in model_info['Name']:
                ckpt_path = model_info['Weights']

        ckpt = os.path.join(working_dir, t[:-4] + ".pth")
        if not os.path.exists(ckpt):
            download_url_to_file(url=ckpt_path, dst=ckpt)
        load_checkpoint(model, ckpt, map_location=dev)
    return model


def get_mm_mean_std(library: str, config: str, verbose=False):
    mm_lib = MMLibrary()
    lib_key= library
    config_key = config

    if verbose:
        lib_keys = mm_lib.get_library_keys()
        print(f"Available libraries are: {lib_keys}, using {lib_key}")

        configs = mm_lib.get_mm_config_keys(lib_key)
        print(f"Available configs for {lib_key} are: {configs}, using {config_key}")

    mm_helper = MMHelper(module=mm_lib.get_mm_module(lib_key=lib_key), configs_path=mm_lib.get_mm_configs_folder(lib_key=lib_key))
    config = mm_helper.get_config(config_path=mm_lib.get_mm_config_path(lib_key=lib_key, cfg_key=config_key))

    mean, std = mm_helper.get_mean(config), mm_helper.get_std(config)
    return mean, std


def _test_module():
    mm_library = MMLibrary()
    lib_key = "mmdetection"
    config_key = "faster_rcnn"

    mm_helper = MMHelper(module=mm_library.get_mm_module(lib_key=lib_key), configs_path=mm_library.get_mm_configs_folder(lib_key=lib_key))

    config_path = mm_library.get_mm_config_path(lib_key=lib_key, cfg_key=config_key)
    config = mm_helper.get_config(config_path)

    task = mm_library.get_mm_task(lib_key=lib_key)
    config = mm_helper.build_model(config, task)
    print(config)


if __name__ == "__main__":
    # _test_forks()
    _test_module()