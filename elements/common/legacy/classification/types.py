from enum import Enum


class ClassifierType(Enum):
    """
    Classifier types
    """
    SGD_CLASSIFIER = 1
    DECISION_TREE = 2
    RANDOM_FOREST = 3
    LOGISTIC_REGRESSION = 4
    GAUSSIAN_NAIVE_BAYES = 5
    GRADIENT_BOOSTING_CLASSIFIER = 6
    SUPPORT_VECTOR_CLASSIFIER = 7


class TorchvisionModelType(Enum):
    """
    Torchvision Model Architectures
    Link: https://pytorch.org/vision/0.8/models.html
    """
    ALEXNET = 'alexnet'
    DENSENET121 = 'densenet121'
    DENSENET161 = 'densenet161'
    DENSENET169 = 'densenet169'
    DENSENET201 = 'densenet201'
    GOOGLENET = 'googlenet'
    INCEPTION_V3 = 'inception_v3'
    MNASNET0_5 = 'mnasnet0_5'
    MNASNET0_75 = 'mnasnet0_75'
    MNASNET1_0 = 'mnasnet1_0'
    MNASNET1_3 = 'mnasnet1_3'
    MOBILENET_V2 = 'mobilenet_v2'
    RESNET101 = 'resnet101'
    RESNET152 = 'resnet152'
    RESNET18 = 'resnet18'
    RESNET34 = 'resnet34'
    RESNET50 = 'resnet50'
    RESNEXT50_32X4D = 'resnext50_32x4d'
    RESNEXT101_32X8D = 'resnext101_32x8d'
    SHUFFLENETV2 = 'shufflenetv2'
    SHUFFLENET_V2_X0_5 = 'shufflenet_v2_x0_5'
    SHUFFLENET_V2_X1_0 = 'shufflenet_v2_x1_0'
    SHUFFLENET_V2_X1_5 = 'shufflenet_v2_x1_5'
    SHUFFLENET_V2_X2_0 = 'shufflenet_v2_x2_0'
    SQUEEZENET1_0 = 'squeezenet1_0'
    SQUEEZENET1_1 = 'squeezenet1_1'
    VGG11 = 'vgg11'
    VGG11_BN = 'vgg11_bn'
    VGG13 = 'vgg13'
    VGG13_BN = 'vgg13_bn'
    VGG16 = 'vgg16'
    VGG16_BN = 'vgg16_bn'
    VGG19 = 'vgg19'
    VGG19_BN = 'vgg19_bn'
    WIDE_RESNET50_2 = 'wide_resnet50_2'
    WIDE_RESNET101_2 = 'wide_resnet101_2'
