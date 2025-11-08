import cv2
from pytorch_grad_cam import EigenGradCAM

import numpy as np


class NormalizeEigenGradCAM(EigenGradCAM):
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None, normalize_result: bool = False):
        super(EigenGradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)
        self._normalize_result = normalize_result

    def scale_cam_image(self, cam, target_size=None):
        result = []
        for img in cam:
            if self._normalize_result:
                img = img - np.min(img)
                img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)
        return result


def create_eigen_gradcam_pt(model, target_layer, use_cuda=False, reshape_transform=None, normalize_result: bool = False):
    return NormalizeEigenGradCAM(model=model, target_layer=target_layer, use_cuda=use_cuda,
                                 reshape_transform=reshape_transform, normalize_result=normalize_result)
