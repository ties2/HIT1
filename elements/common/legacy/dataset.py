import platform
import os
from enum import Enum

from common.elements.utils import deprecated


class Dataset(Enum):
    POTATO_PLANT = 0
    POTATO_PLANT_TILES = 1
    CIFAR10 = 2
    LEAPMOTION_HAND_GESTURES = 3
    ORBS_TRAINING = 4
    ORBS_VALIDATION = 5
    ORBS_TESTING = 6
    ORBS_HIDDEN_TESTING = 7
    POTATO_PLANT_TRAINING = 8
    POTATO_PLANT_VALIDATION = 9
    POTATO_PLANT_TESTING = 10
    POTATO_PLANT_SMALL = 11
    PLASTICS_PE_PET_PP_PS_TRAINING = 12
    PLASTICS_PE_PET_PP_PS_VALIDATION = 13
    PLASTICS_MULTILAYER_PE_PP_TRAINING = 14
    PLASTICS_PP_PET_PE_TRAINING = 15
    BRAZILIAN_COINS_SMALL = 19
    COVID_19CT = 20
    COVID_19CT_16 = 21
    COINS_MASK = 22
    GRID_CAMERAMAN = 23
    ELEPHANT = 24
    CLUSTER_CUBE = 25
    MNIST_TORCHVISION = 26
    MVTECAD = 27
    ORBS_TRAINING_SMALL = 28
    NUCLEI_TRAINING = 29
    NUCLEI_TRAINING_SMALL = 30
    PENNFUDAN = 31
    PLASTICS_REF_SMALL = 32
    CROP_WEED = 33
    OCID_DEBUG = 42
    OCID_ARID20_MIXED = 43
    PLASTICS_REF_TINY = 46
    PLASTICS_REF_MEDIUM = 47
    PLASTICS_REF_SMALL_PE_PP_PET_PS = 48
    PLASTICS_REF_COMPLETE_2021 = 49
    M4E_TRAINING = 50
    M4E_VALIDATION = 51
    M4E_TESTING = 52
    PLASTICS_REF_MEDIUM_PE_PP_PET_PS = 53
    PLASTICS_REF_SMALL_PE_PET = 54
    PLASTICS_REF_MEDIUM_PE_PET = 55
    IMAGE_MATH = 56
    HULL_RUST_HIGH = 57
    HULL_RUST = 58

    def __str__(self):
        return str(self.name)


class DatasetInfo(Enum):
    DATA = 0  # input + annotations
    INPUT_DATA = 1  # input data only
    ANNOTATION = 2  # annotations only (can be any type, segmentation, boxes, etc.)
    MODEL = 3  # A trained model for this dataset
    IMAGE_DIR = 4  # Image subdir
    MASK_DIR = 5  # Masks subdir
    NUM_CHANNELS = 6  # Number of channels in this dataset
    NUM_CLASSES = 7  # The number of classes in this dataset
    CLASS_NAMES = 9  # The class names in this dataset
    MAX_PIXEL_VALUE = 8  # The theoretical maximum pixel value
    IMAGE_NAMES = 10  # A name of a text file containing the image filenames of this dataset
    SHORT_CLASS_NAMES = 11  # Shortened class names
    PREVIEW = 12  # A preview images for this dataset
    NR_OF_SAMPLES = 13  # The number of samples in this dataset
    MEAN = 14  # The dataset mean
    SDEV = 15  # The dataset standard deviation
    MIN = 16  # The dataset minimum value
    MAX = 17  # The dataset maximum value
    CLASS_INFO = 18  # A list of dict containing information about the classes: names, ids, etc.
    DESCRIPTION = 19  # Human readable description


def get_datasets_root_dir(private=False):
    p = platform.system()
    if not private:
        if p == "Windows":
            return os.path.join("x:", "datasets")
        elif p == "Linux":
            return os.path.join(os.sep, "media", "public_data", "datasets")
        else:
            # raise RuntimeError(f"Unrecognized unsupported platform {platform.platform()}")
            return ""
    else:
        if p == "Linux":
            return os.path.join(os.sep, "media", "private_data", "MasterCVDS", "datasets")
        else:
            raise RuntimeError(f"Unrecognized or unsupported platform {platform.platform()}")


DatasetDirs = {
    Dataset.HULL_RUST: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "toddis_defects"),
        DatasetInfo.CLASS_INFO: [{'id': 0, 'name': 'Background', 'color': (0, 0, 0)},
                                 {'id': 255, 'name': 'Rust', 'color': (255, 255, 255)},
                                 ],
    },
    Dataset.HULL_RUST_HIGH: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "hull_rust"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "hull_rust",
                                             "HiRes", "raw_renamed"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "hull_rust",
                                             "HiRes", "labeled_fixed"),
        DatasetInfo.CLASS_INFO: [{'id': 0, 'name': 'Background', 'color': (0, 0, 0)},
                                 {'id': 255, 'name': 'Rust', 'color': (255, 255, 255)},
                                 ],
    },
    Dataset.POTATO_PLANT: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant", "annotation_a")
    },
    Dataset.POTATO_PLANT_TRAINING: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant", "training"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant", "training")
    },
    Dataset.POTATO_PLANT_VALIDATION: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant",
                                             "validation"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant",
                                             "validation")
    },
    Dataset.POTATO_PLANT_TESTING: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant", "testing"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant", "testing")
    },
    Dataset.POTATO_PLANT_TILES: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant", "tiles"),
        DatasetInfo.PREVIEW: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant", "tiles", "PotatoPlant0_0_0.png")
    },
    Dataset.POTATO_PLANT_SMALL: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "Potatoplant", "potato_plants_small.csv"),
    },
    Dataset.CIFAR10: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "external", "image", "classification", "CIFAR10-torchvision"),
        DatasetInfo.MODEL: os.path.join(get_datasets_root_dir(), "external", "image", "classification", "CIFAR10-torchvision", "model-new.pth"),
    },
    Dataset.LEAPMOTION_HAND_GESTURES: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "external", "image", "classification", "leap_database"),
    },
    Dataset.ORBS_TRAINING: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "orbs", "training", "img"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "orbs", "training", "xml"),
    },
    Dataset.ORBS_TRAINING_SMALL: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "orbs", "training_small"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "orbs", "training_small"),
    },
    Dataset.ORBS_VALIDATION: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "orbs",
                                             "validation", "img"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "orbs",
                                             "validation", "xml"),
    },
    Dataset.ORBS_TESTING: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "orbs", "testing", "img"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "orbs", "testing", "xml"),
    },
    Dataset.ORBS_HIDDEN_TESTING: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "orbs", "hidden_testing", "img"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "orbs", "hidden_testing", "xml"),
    },
    Dataset.PLASTICS_PE_PET_PP_PS_TRAINING: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "CircularPlastics", "plastics_pe_pet_pp_ps", "training"),
    },
    Dataset.PLASTICS_PE_PET_PP_PS_VALIDATION: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "CircularPlastics", "plastics_pe_pet_pp_ps", "training"),
    },
    Dataset.PLASTICS_MULTILAYER_PE_PP_TRAINING: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "CircularPlastics", "plastics_multilayer_pe_pp", "training"),
    },
    Dataset.PLASTICS_PP_PET_PE_TRAINING: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "CircularPlastics", "plastics_pp_pet_pe", "training"),
        DatasetInfo.PREVIEW: os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "CircularPlastics", "plastics_pp_pet_pe", "training", "NHLPlastics_DA_DB_DC_DD_2018-01-11_17-25-31.npz")
    },
    Dataset.BRAZILIAN_COINS_SMALL: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "external", "image", "object_detection", "brazilian_coins_small", "img"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "external", "image", "object_detection", "brazilian_coins_small", "annot"),
    },
    Dataset.COVID_19CT: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "COVID-19CT"),
        DatasetInfo.IMAGE_DIR: 'images',
        DatasetInfo.MASK_DIR: 'masks',
        DatasetInfo.NUM_CHANNELS: 1,
        DatasetInfo.NUM_CLASSES: 3,
        DatasetInfo.MAX_PIXEL_VALUE: pow(2, 15),
        DatasetInfo.CLASS_NAMES: ['background', 'ground-glass', 'consolidation'],
        DatasetInfo.SHORT_CLASS_NAMES: ['back', 'glass', 'cons'],
        DatasetInfo.IMAGE_NAMES: 'full_dataset.txt',
    },
    Dataset.COVID_19CT_16: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "COVID-19CT"),
        DatasetInfo.IMAGE_DIR: 'images',
        DatasetInfo.MASK_DIR: 'masks',
        DatasetInfo.NUM_CHANNELS: 1,
        DatasetInfo.NUM_CLASSES: 3,
        DatasetInfo.MAX_PIXEL_VALUE: pow(2, 15),
        DatasetInfo.CLASS_NAMES: ['background', 'ground-glass', 'consolidation_&_pleural_effusion'],
        DatasetInfo.SHORT_CLASS_NAMES: ['back', 'glass', 'cons'],
        DatasetInfo.IMAGE_NAMES: 'covid_16.txt',
    },
    Dataset.COINS_MASK: {
        DatasetInfo.DATA: [os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "COINS_MASK", "coins_mask.png"),
                           os.path.join(get_datasets_root_dir(), "internal", "image", "segmentation", "COINS_MASK", "coins_mask_small.png")]
    },
    Dataset.GRID_CAMERAMAN: {
        DatasetInfo.DATA:
            os.path.join(get_datasets_root_dir(), "internal", "image", "preprocessing", "fourier_transform")
    },
    Dataset.ELEPHANT: {
        DatasetInfo.DATA:
            os.path.join(get_datasets_root_dir(), "internal", "image", "classification", "grad-cam")
    },
    Dataset.CLUSTER_CUBE: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "unsupervised", "plastics")
    },
    Dataset.MVTECAD: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "external", "image", "unsupervised", "mvtecad"),
        DatasetInfo.CLASS_NAMES: ['bottle', 'cable', 'capsule', 'carpet', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
    },
    Dataset.MNIST_TORCHVISION: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "external", "image", "classification", "MNIST-torchvision")
    },
    Dataset.NUCLEI_TRAINING: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "external", "image", "instance_segmentation", "nuclei", "stage1_train"),
        DatasetInfo.NUM_CLASSES: 2
    },
    Dataset.NUCLEI_TRAINING_SMALL: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "external", "image", "instance_segmentation", "nuclei", "train_small"),
        DatasetInfo.NUM_CLASSES: 2
    },
    Dataset.PENNFUDAN: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "external", "image", "instance_segmentation", "pennfudan", "PNGImages"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "external", "image", "instance_segmentation", "pennfudan", "PedMasks"),
        DatasetInfo.NR_OF_SAMPLES: 96,
        DatasetInfo.NUM_CLASSES: 2
    },
    Dataset.PLASTICS_REF_SMALL: {
        DatasetInfo.DESCRIPTION: "With subsets training, validation, testing set using one sample per set. Each subset contains a different image of the same sample.",
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "small"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "small", "crop"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "small", "annot"),
        DatasetInfo.CLASS_INFO: [{'id': 0, 'name': 'Background', 'color': (0, 0, 0)},
                                 {'id': 2826443, 'name': 'PS', 'color': (0, 255, 0)},
                                 {'id': 2909743, 'name': 'PE', 'color': (0, 0, 255)},
                                 {'id': 2911224, 'name': 'PP', 'color': (255, 0, 0)},
                                 {'id': 2911168, 'name': 'PET', 'color': (0, 255, 255)},
                                 {'id': 2911697, 'name': 'PVC', 'color': (0, 0, 255)},
                                 {'id': 2911129, 'name': 'IML', 'color': (255, 255, 255)},
                                 ],
        DatasetInfo.NUM_CHANNELS: 224,
        DatasetInfo.NUM_CLASSES: 7,
        },
    Dataset.PLASTICS_REF_SMALL_PE_PP_PET_PS: {
        DatasetInfo.DESCRIPTION: "With subsets training, validation, testing set using one sample per set. Each subset contains a different image of the same sample.",
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "small_pe_pp_pet_ps"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "small_pe_pp_pet_ps", "crop"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "small_pe_pp_pet_ps", "annot"),
        DatasetInfo.CLASS_INFO: [{'id': 0, 'name': 'Background', 'color': (0, 0, 0)},
                                 {'id': 2826443, 'name': 'PS', 'color': (0, 255, 0)},
                                 {'id': 2909743, 'name': 'PE', 'color': (0, 0, 255)},
                                 {'id': 2911224, 'name': 'PP', 'color': (255, 0, 0)},
                                 {'id': 2911168, 'name': 'PET', 'color': (0, 255, 255)},
                                 ],
        DatasetInfo.NUM_CHANNELS: 224,
        DatasetInfo.NUM_CLASSES: 7,
    },
    Dataset.PLASTICS_REF_SMALL_PE_PET: {
        DatasetInfo.DESCRIPTION: "With subsets training, validation, testing set using one sample per set. Each subset contains a different image of the same sample.",
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "small_pe_pet"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "small_pe_pet", "crop"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "small_pe_pet", "annot"),
        DatasetInfo.CLASS_INFO: [{'id': 0, 'name': 'Background', 'color': (0, 0, 0)},
                                 {'id': 2909743, 'name': 'PE', 'color': (0, 0, 255)},
                                 {'id': 2911168, 'name': 'PET', 'color': (0, 255, 255)},
                                 ],
        DatasetInfo.NUM_CHANNELS: 224,
        DatasetInfo.NUM_CLASSES: 7,
    },
    Dataset.PLASTICS_REF_TINY: {
        DatasetInfo.DESCRIPTION: "With subsets training, validation, testing set using one sample per set (only one class). Each subset contains a different image of the same sample.",
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "tiny"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "tiny", "crop"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "tiny", "annot"),
        DatasetInfo.CLASS_INFO: [{'id': 0, 'name': 'Background', 'color': (0, 0, 0)},
                                 {'id': 2826443, 'name': 'Plastic', 'color': (0, 255, 0)},
                                 ],
        DatasetInfo.NUM_CHANNELS: 224,
        DatasetInfo.NUM_CLASSES: 2,
        DatasetInfo.MEAN: 0.09343876291680331,
        DatasetInfo.SDEV: 0.07841722265934076
        },
    Dataset.PLASTICS_REF_MEDIUM: {
        DatasetInfo.DESCRIPTION: "Class-compatible with PLASTICS_REF_SMALL. One annotated sample per class that is different from PLASTICS_REF_SMALL. Meant as a testing set for PLASTICS_REF_SMALL.",
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "medium"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "medium", "crop"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "medium", "annot"),
        DatasetInfo.CLASS_INFO: [{'id': 0, 'name': 'Background', 'color': (0, 0, 0)},
                                 {'id': 2826443, 'name': 'PS', 'color': (255, 255, 0)},
                                 {'id': 2909743, 'name': 'PE', 'color': (255, 0, 255)},
                                 {'id': 2911224, 'name': 'PP', 'color': (255, 0, 0)},
                                 {'id': 2911168, 'name': 'PET', 'color': (0, 255, 255)},
                                 {'id': 2911697, 'name': 'PVC', 'color': (0, 0, 255)},
                                 {'id': 2911129, 'name': 'IML', 'color': (255, 255, 255)},
                                 ],
        DatasetInfo.NUM_CHANNELS: 224
    },
    Dataset.PLASTICS_REF_MEDIUM_PE_PP_PET_PS: {
        DatasetInfo.DESCRIPTION: "Class-compatible with PLASTICS_REF_SMALL. One annotated sample per class that is different from PLASTICS_REF_SMALL_PE_PP_PET_PS. Meant as a testing set for PLASTICS_REF_SMALL_PE_PP_PET_PS.",
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "medium_pe_pet"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "medium_pe_pp_pet_ps", "crop"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "medium_pe_pp_pet_ps", "annot"),
        DatasetInfo.CLASS_INFO: [{'id': 0, 'name': 'Background', 'color': (0, 0, 0)},
                                 {'id': 2826443, 'name': 'PS', 'color': (255, 255, 0)},
                                 {'id': 2909743, 'name': 'PE', 'color': (255, 0, 255)},
                                 {'id': 2911224, 'name': 'PP', 'color': (255, 0, 0)},
                                 {'id': 2911168, 'name': 'PET', 'color': (0, 255, 255)},
                                 ],
        DatasetInfo.NUM_CHANNELS: 224
    },
    Dataset.PLASTICS_REF_MEDIUM_PE_PET: {
        DatasetInfo.DESCRIPTION: "Class-compatible with PLASTICS_REF_SMALL. One annotated sample per class that is different from PLASTICS_REF_SMALL_PE_PET. Meant as a testing set for PLASTICS_REF_SMALL_PE_PET.",
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "medium_pe_pet"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "medium_pe_pet", "crop"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "medium_pe_pet", "annot"),
        DatasetInfo.CLASS_INFO: [{'id': 0, 'name': 'Background', 'color': (0, 0, 0)},
                                 {'id': 2909743, 'name': 'PE', 'color': (255, 0, 255)},
                                 {'id': 2911168, 'name': 'PET', 'color': (0, 255, 255)},
                                 ],
        DatasetInfo.NUM_CHANNELS: 224
    },
    Dataset.PLASTICS_REF_COMPLETE_2021: {
        DatasetInfo.DESCRIPTION: "Instance segmentation flake dataset for training production models.",
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "complete"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "complete", "crop"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "instance_segmentation", "plastics", "complete", "annot"),
        DatasetInfo.CLASS_INFO: [{'id': 0, 'name': 'Background', 'color': (0, 0, 0)},
                                 {'id': 1, 'name': 'PE', 'color': (255, 0, 255)},
                                 {'id': 2, 'name': 'PP', 'color': (255, 0, 0)},
                                 {'id': 3, 'name': 'PS', 'color': (255, 255, 0)},
                                 {'id': 4, 'name': 'PET', 'color': (0, 255, 255)},
                                 ],
        DatasetInfo.NUM_CHANNELS: 224
    },
    Dataset.CROP_WEED: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "CropWeed"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "CropWeed", "images"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "CropWeed", "masks")
    },
    Dataset.OCID_DEBUG: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "OCID-dataset", "debug"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "OCID-dataset", "debug", 'rgbd'),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "OCID-dataset", "debug", "label"),
        DatasetInfo.NUM_CLASSES: 19,
        DatasetInfo.SDEV: 128,
        DatasetInfo.MEAN: 255,
    },
    Dataset.OCID_ARID20_MIXED: {
        DatasetInfo.DATA: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "OCID-dataset", "ARID20-mixed"),
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "OCID-dataset", "ARID20-mixed", 'rgbd'),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "external", "image", "segmentation", "OCID-dataset", "ARID20-mixed", "label"),
        DatasetInfo.NUM_CLASSES: 89,
        DatasetInfo.SDEV: 128,
        DatasetInfo.MEAN: 255,
    },
    Dataset.M4E_TRAINING: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "M4E", "training", "img"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "M4E", "training", "xml"),
    },
    Dataset.M4E_VALIDATION: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "M4E",
                                             "validation", "img"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "M4E",
                                             "validation", "xml"),
    },
    Dataset.M4E_TESTING: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "M4E", "testing", "img"),
        DatasetInfo.ANNOTATION: os.path.join(get_datasets_root_dir(), "internal", "image", "object_detection", "M4E", "testing", "xml"),
    },
    Dataset.IMAGE_MATH: {
        DatasetInfo.INPUT_DATA: os.path.join(get_datasets_root_dir(), "internal", "other", "image_math")
    },
}


# @deprecated("This function is deprecated, please use dataset information from common.data.datasets_info instead.")
def get_dataset_info(dataset: Dataset, info: DatasetInfo = DatasetInfo.INPUT_DATA, *sub_folders):
    if dataset not in DatasetDirs.keys():
        raise ValueError(f"The dataset directory for {dataset} could not be found")
    if info not in DatasetDirs[dataset].keys():
        raise ValueError(f"The dataset {dataset} has no info on {info}")
    if len(sub_folders) > 0:
        return os.path.join(*(DatasetDirs[dataset][info], *sub_folders))
    else:
        return DatasetDirs[dataset][info]


if not "CVDS_PIPELINE_NO_DEPRECATED_WARNING" in os.environ.keys() or os.environ["CVDS_PIPELINE_NO_DEPRECATED_WARNING"] != "1":
    get_dataset_info = deprecated("This function is deprecated, please use dataset information from common.data.datasets_info instead.")(get_dataset_info)
