# Convert yolov5 pretrained models to "standard serialize format"
# In order to avoid   'ModuleNotFoundError: No module named 'models.yolo' in  result = unpickler.load()
import glob

# Note: run this script in directory weights of Yolov5 repro

import torch
import os
import sys

# add yolov5 repository directory (https://github.com/ultralytics/yolov5/) to pythonpath
yolov5_repo_dir = '/home/willem/git/yolov5/'
sys.path.insert(0, yolov5_repo_dir)


def convert(model_fp: str):
    checkpoint = torch.load(model_fp)
    state_dict = checkpoint['model'].float().state_dict()  # to FP32
    with open(model_fp, "wb") as f:
        torch.save({'model': state_dict}, f)


if __name__ == '__main__':
    model_dir = os.path.join('/media', 'public_data', 'datasets', 'models', 'yolov5', 'latest')
    [convert(model_file) for model_file in glob.glob(os.path.join(model_dir, '*.pt'))]
