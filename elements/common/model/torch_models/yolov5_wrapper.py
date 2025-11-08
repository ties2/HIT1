# Parts of the code are derived from the YoloV5 repository of Ultralytics,  https://github.com/ultralytics/yolov5.
# Note: version 4.0, downloaded January 2021.
# This Yolo repository is released under the GNU General Public License v3.0 licence.

# todo for new version of this repro do:
# - utils/general.py: comment out 2x set_printoptions
# - run yolov5_convert_model.py and copy *.pth to yolo_conf_dir


import os
import sys
import yaml
from typing import List, Any, Tuple
import types

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

# try:
import ultralytics
ultralytics_cur_log_level = ultralytics.utils.LOGGER.level
ultralytics.utils.LOGGER.setLevel("ERROR")

# ----------------------------------------------------------------------------------------------------------------------
# Workaround for 'yolov5' package no longer being compatible with 'ultralytics' package (as of 2023-09-27)
#
# Spoofs now non-existent 'ultralytics.yolo.utils.checks' module with a no-op 'check_requirements' function
# ----------------------------------------------------------------------------------------------------------------------

try:
    import ultralytics.yolo
except:
    def _ultralytics_workaround_check_requirements(*args, **kwargs):
        pass

    ultralytics_checks_module = types.ModuleType('checks', 'Workaround for yolov5 incompatibility with ultralytics')
    ultralytics_checks_module.__setattr__('check_requirements', _ultralytics_workaround_check_requirements)
    sys.modules['ultralytics.yolo.utils.checks'] = ultralytics_checks_module

# ----------------------------------------------------------------------------------------------------------------------
# End of workaround
# ----------------------------------------------------------------------------------------------------------------------

import yolov5
ultralytics.utils.LOGGER.setLevel(ultralytics_cur_log_level)

from yolov5.models.yolo import Model
from yolov5.utils.general import non_max_suppression, check_img_size, intersect_dicts
from yolov5.utils.loss import ComputeLoss

# reset printoptions to default and logger defined in the yolov5 libary (general file)
yolov5.utils.general.VERBOSE = False
yolov5.utils.general.set_logging(verbose=False)
torch.set_printoptions(profile="default")  # as suggested by https://pytorch.org/docs/stable/_modules/torch/_tensor_str.html#set_printoptions
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)  # as specified by https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
# except Exception as e:
#     print(e)

class YoloV5(nn.Module):
    """YoloV5 object detector."""

    def __init__(self, model_name: str, num_classes: int, num_chans=3, device='cuda:0', pretrained=False,
                 yolo_conf_dir: str = '/media/public_data/datasets/models/yolov5/latest'):
        """
        The Yolov5 constructor.

        :param model_name: The name of the Yolo model. ('yolov5s', 'yolov5m', 'yolov5l' or 'yolov5x')
        :param num_classes: The number of classes.
        :param num_chans: The number of channels in the images.
        :param device: The device to run the model on.
        :param pretrained: True if pretrained on COCO dataset should be loaded.
        """
        super().__init__()
        self.yolo_conf_dir = yolo_conf_dir
        with open(os.path.join(yolo_conf_dir, 'hyp.scratch.yaml')) as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)
        self.num_classes = num_classes
        self.class_names = [str(c) for c in range(num_classes)]
        cfg_name = os.path.join(yolo_conf_dir, model_name + '.yaml')
        self.model = Model(cfg_name, ch=num_chans, nc=self.num_classes).to(device)
        if pretrained:
            filename = os.path.join(yolo_conf_dir, f'{model_name}.pt')
            if not os.path.isfile(filename): raise RuntimeError(f'No pretrained file with name: {filename}')
            state_dict = intersect_dicts(torch.load(filename)['model'], self.model.state_dict(), exclude=[])
            self.model.load_state_dict(state_dict, strict=False)
        hyp['cls'] *= self.num_classes / 80.  # scale coco-tuned hyp['cls'] to current dataset
        self.model.nc = self.num_classes
        self.model.hyp = hyp
        self.model.gr = 1.0
        self.model.class_weights = np.ones(self.num_classes, dtype=np.float32) / self.num_classes
        self.model.names = self.class_names
        self.loss_fn = ComputeLoss(self.model)

    def optimal_image_size(self, image_size):
        """Calculate the optimal image size for network."""
        max_stride = int(max(self.model.stride))
        return check_img_size(image_size, max_stride)

    def forward(self, inputs: torch.Tensor) -> Tuple[Any, ...]:
        """
        The inference function.

        :param inputs: Tensor of shape NxCxHxW.
        :return: (raw_detections and) feature map.

        If in training mode, the predictions (feature map, tensor for each scale) are returned as Tuple[0]:
        List[B x 3 x cell_v X cell_h x (xyhw_object_conf + numclss*conf_cls), 3 = nr predictions per cell.
        Predictions can be used as input to the loss function.
        If in eval mode, both raw_detections and predictions are returned as Tuple[0] and Tuple[1].
        Raw_detections are flattened predictions: Tensor [B x (3 x nr_cells_over_all_3_levels) x (xyhw_object_conf + numclss*conf_cls).
        Raw_detections can be used as input to the detections function.
        """
        preds = self.model(inputs)
        if self.model.training:
            return preds,               # feature map
        else:  # eval mode
            return preds[0], preds[1]   # raw detections, feature pyramid map

    def loss(self, preds: Tuple[Any, ...], targets: torch.Tensor) -> torch.Tensor:
        """
        The loss function for training.

        :param preds: The predictions from function forward.
        :param targets: The target bounding boxes.
        :return: The loss.
        """
        return self.loss_fn(preds[len(preds)-1], targets)[0]

    def detections(self, preds: Tuple[Any, ...]) -> List[torch.Tensor]:
        """
        Calculate detections from raw detections.

        :param preds: The predictions from function forward.
        :return: A list with for each image a tensor of bounding boxes [y1, x1, y2, x2, class, score].
        """
        detects = non_max_suppression(preds[0])
        # yolo detections format: list with for each image torch tensor with detections: [x1, y1, x2, y2, conf, cls], sorted on conf
        res = []
        for preds in detects:
            if preds is not None:
                dets = torch.empty_like(preds, device=torch.device('cpu'))
                dets[:, 0] = preds[:, 1]
                dets[:, 1] = preds[:, 0]
                dets[:, 2] = preds[:, 3]
                dets[:, 3] = preds[:, 2]
                dets[:, 4] = preds[:, 5]
                dets[:, 5] = preds[:, 4]
                res.append(dets)
            else:
                res.append(torch.empty([0, 6]))  # add dummy
        return res
