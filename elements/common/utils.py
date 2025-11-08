import contextlib
import copy
import functools
import inspect
import multiprocessing
import subprocess
import sys
import unittest
import warnings
import logging
import os
from enum import Enum
from subprocess import CalledProcessError
from typing import List, Any, Generator, Tuple, Optional

import torch
import time
import numpy as np

newline = "\n"

GLOBAL_CACHE_PATH = '/tmp/pipeline_cache'  # temporary cached files
GLOBAL_TMP_PATH = '/tmp/pipeline_output'
GLOBAL_DEPLOY_PATH = '/tmp/pipeline_deploy'
GLOBAL_CICD_MODE = False


class Timer(contextlib.ContextDecorator):
    # Usage: @Timer() decorator or 'with Timer('name'):' context manager
    def __init__(self, name: str = 'Timer', logger: logging.Logger = logging.getLogger(__name__)):
        self.name = name
        self.logger = logger

    def __enter__(self):
        if multiprocessing.parent_process() is None and torch.cuda.is_available():
            # synchronize can not be called in a forked subprocess
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, _, __, ___):
        if multiprocessing.parent_process() is None and torch.cuda.is_available():
            # synchronize can not be called in a forked subprocess
            torch.cuda.synchronize()
        self.logger.info(f'{self.name}: {time.perf_counter() - self.start:.6f}s')


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class DevNull:
    """
    Object that can be used to set stdout to, if no output should be given.
    Incorporates possible functions that can be called on stdout.
    """

    def noop(*args, **kwargs): pass

    close = write = flush = writelines = noop


def set_cache_dir(p):
    global GLOBAL_CACHE_PATH
    GLOBAL_CACHE_PATH = os.path.abspath(p)
    get_cache_dir()


def get_cache_dir(sub_dir: str = None):
    global GLOBAL_CACHE_PATH
    p = GLOBAL_CACHE_PATH
    if sub_dir is not None:
        p = os.path.join(p, sub_dir)
    os.makedirs(p, exist_ok=True)
    return p


def get_cicd_tmp(subfolder: str = "output"):
    branchname = os.getenv('CI_COMMIT_REF_SLUG')  # GitLab CICD will set this var to the current branch/tag name
    if branchname is None:
        branchname = "unknown_branch"
    tmp_path = os.path.join(os.sep, "media", "private_data", "cicd", "pipeline", branchname, subfolder)
    return tmp_path


def set_tmp_dir(p):
    global GLOBAL_TMP_PATH
    GLOBAL_TMP_PATH = p
    get_tmp_dir()


def get_tmp_dir(sub_dir: str = None):
    global GLOBAL_TMP_PATH
    p = GLOBAL_TMP_PATH
    if sub_dir is not None:
        p = os.path.join(p, sub_dir)
    try:
        os.makedirs(p, exist_ok=True)
    except Exception as e:
        print(f"Warning could not create temporary folder: {p}: Exception: {e}")
    return p


def set_deploy_dir(p):
    global GLOBAL_DEPLOY_PATH
    GLOBAL_DEPLOY_PATH = p
    get_deploy_dir()


def get_deploy_dir(sub_dir: str = None):
    global GLOBAL_DEPLOY_PATH
    p = GLOBAL_DEPLOY_PATH
    if sub_dir is not None:
        p = os.path.join(p, sub_dir)
    try:
        os.makedirs(p, exist_ok=True)
    except Exception as e:
        print(f"\nWarning: {e}")
    return p


def set_cicd_mode(enabled: bool):
    global GLOBAL_CICD_MODE
    GLOBAL_CICD_MODE = enabled


def get_cicd_mode() -> bool:
    global GLOBAL_CICD_MODE
    return GLOBAL_CICD_MODE


class CICDTestType(Enum):
    SHORT = 1  # no pipelines at all
    FULL_QUICK = 2  # pipelines should run minimal number of epochs and with relaxed performance requirements
    FULL_COMPLETE = 3  # pipelines should run normal number of epochs and with more strict performance requirements


def _get_cicd_test_type() -> CICDTestType:
    if not get_cicd_mode():
        return CICDTestType.FULL_COMPLETE  # always run full code when starting manually
    else:
        ci_source = os.getenv("CI_PIPELINE_SOURCE")
        ci_branch_slug = os.getenv("CI_COMMIT_REF_SLUG")
        ci_test_type = os.getenv("CI_TEST_TYPE")

        if ci_test_type is None:
            if ci_branch_slug == "master":
                return CICDTestType.FULL_COMPLETE  # always run complete for master
            if ci_branch_slug == "dev" and ci_source in ["schedule", "merge_request_event"]:
                return CICDTestType.FULL_COMPLETE  # run complete for dev only for merge requests (which should always be into master) or nightlies
            if ci_source == "merge_request_event":
                return CICDTestType.FULL_QUICK  # other merge requests run quick test of pipelines

            # default
            return CICDTestType.SHORT  # for other events (e.g. push) don't run pipelines by default at all
        else:
            if ci_test_type == "SHORT":
                return CICDTestType.SHORT
            elif ci_test_type == "FULL_QUICK":
                return CICDTestType.FULL_QUICK
            else:
                return CICDTestType.FULL_COMPLETE


def get_cicd_test_type() -> CICDTestType:
    type = _get_cicd_test_type()
    return type


class ChangeWorkingDir(object):
    """
    Context manager for changing the current working directory. Will restore to the original working directory on exit.
    """

    def __init__(self, working_dir: str):
        if not os.path.isdir(working_dir):
            raise ValueError(f"Specified working directory does not exist: {working_dir}")

        self._working_dir = working_dir
        self._original_dir: Optional[str] = None

    def __enter__(self):
        self._original_dir = os.getcwd()
        os.chdir(self._working_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._original_dir)


def get_logger(log_file_path):

    # Create or retrieve the logger
    logger = logging.getLogger('pipeline_logs')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent duplicate logs in the root logger

    # Add a StreamHandler if no handlers exist (to avoid duplicates)
    if not logger.handlers:
        # Define a file handler to write logs to the specified file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Add console handler for simultaneous console logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger




def get_console_logger(name=__name__, level=logging.INFO):
    """
    Configures and returns a logger for console output only, with a specified name and logging level.

    :param name: Name of the logger (default is 'CustomLogger')
    :param level: Logging level (default is logging.INFO)
    :return: Configured logger instance
    """
    # Create or retrieve the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Define formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add StreamHandler only if no handlers exist (to avoid duplicates)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def static_var(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def wait_forever(msg: str = "Waiting forever."):
    wait_chars = ['|', '/', '-', '\\', '-', '*']
    print(f'{msg}... ', end="", flush=True)
    i = 0
    while True:
        print(wait_chars[i], end="", flush=True)
        time.sleep(2)
        print("\b", end="", flush=True)
        i = 0 if i > len(wait_chars) - 2 else i + 1


class PrettyPrint:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_bold(msg: str):
    print(PrettyPrint.BOLD + msg + PrettyPrint.END)


def disable_batchnorm_pt(model):
    bns = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d)]
    for i in range(len(bns)):
        bns[i].momentum = 0.99
        bns[i].track_running_stats = False
    return


def select_device(dev: str = 'cpu'):
    # device = 'cpu' or '0' or '0,1,2,3,n' for the specific GPU device
    if isinstance(dev, torch.device):
        return dev
    if dev != 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = dev
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def reproduce_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_compute_device(CUDA_DEVICES="0", reproduce=False, seed=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES  # select CUDA cards [0 .. 3], multiple like: "2,3"
    if reproduce:
        reproduce_seed(seed)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def write_tensor_as_npy(tensor, file_name):  # convert to NxHxWxC
    if tensor.dim() == 2:
        pass
    elif tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.dim() == 4:
        tensor = tensor.permute(0, 2, 3, 1)
    else:
        raise NotImplementedError(f'Unexpected tensor dimension {tensor.dim()}')
    np.save(file_name, tensor.numpy())


def save_tensor_as_npy(*tensors):
    for t in tensors:
        write_tensor_as_npy(t[0], f'{t[1]}.npy')


def concat_np_arrays(*arrays):
    result = np.array([], arrays[0].dtype)
    for array in arrays:
        result = np.append(result, array)
    return result


def chunks(lst: List[Any], n: int) -> Generator[List[Any], None, None]:
    """
    Cuts a list into n chunks of len(lst).
    Not: the last chunk might be shorter

    :param lst: The input list
    :param n: The size of the chunks
    :return: a generator for list of chunks

    >>> for chunk in chunks([1, 2, 3, 4, 5, 6, 7, 8], 3):
    ...     print(chunk)
    [1, 2, 3]
    [4, 5, 6]
    [7, 8]
    """
    n = min(max(1, n), len(lst))
    return (lst[i:i + n] for i in range(0, len(lst), n))


def get_pt_mean(mult: float = 255) -> Tuple[float, float, float]:
    """
    Get the RGB mean of the input images for the pretrained models from the PyTorch model zoo.

    :param mult: Multiplier should equal to the max pixel value in the image (e.g. 255 for 8 bits)
    :return: The RGB mean.
    """
    return 0.485 * mult, 0.456 * mult, 0.406 * mult


def get_pt_std(mult: float = 255) -> Tuple[float, float, float]:
    """
    Get the RGB standard deviation of the input images for the pretrained models from the PyTorch model zoo.

    :param mult: Multiplier should equal to the max pixel value in the image (e.g. 255 for 8 bits)
    :return: The RGB std.
    """
    return 0.229 * mult, 0.224 * mult, 0.225 * mult


def split_dataset(dataset, split=(0.33, 0.33, 0)) -> List[torch.utils.data.Subset]:
    """
    This method splits a torch dataset in parts.

    :param dataset: The torch dataset
    :param split: The list of split items (if the last item is 0 the remaining part is put there)
    :return: A List of Subsets equal to the number of elements in split
    """
    indices = torch.randperm(len(dataset)).tolist()
    result = []
    first = 0
    for s in split:
        if s != 0:
            last = first + round(s * len(indices)) + 1
            split_set = torch.utils.data.Subset(dataset, indices[first:last])
            first = last
        else:
            split_set = torch.utils.data.Subset(dataset, indices[first:])

        result.append(split_set)
    return result


def split_generic_dataset(dataset, split=(0.33, 0.33, 0)) -> List:
    """
    Split samples from GenericDataset into multiple GenericDatasets.

    :param dataset: GenericDataset object.
    :param split: The list of split items (if the last item is 0 the remaining part is put there).
    :return: A List of GenericDatasets equal to the number of elements in split.
    """
    subsets = split_dataset(dataset=dataset, split=split)

    for i, subset in enumerate(subsets):
        subset.dataset = copy.deepcopy(subset.dataset)
        subset.dataset._samples = [subset.dataset.samples[i] for i in subset.indices]
        subsets[i] = subset.dataset
    return subsets


def create_tensor(x: float, dev="cuda:0") -> torch.Tensor:
    """
    Creates a scaler wrapped in a Tensor object with gradients

    :param x: scalar value
    :param dev: Device
    :return: The Tensor

    >>> create_tensor(42, "cpu")
    tensor([42.], requires_grad=True)
    """
    return torch.tensor([float(x)], requires_grad=True, device=dev)


string_types = (type(b''), type(u''))


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))


class RunShellCmd(unittest.TestCase):
    def __init__(self, test_script, python_path, cicd):
        super().__init__("test_command")
        self._cicd = cicd
        self._test_script = test_script
        self._python_path = python_path

    def test_command(self):
        custom_env = os.environ.copy()
        if "PYTHONPATH" in custom_env.keys():
            custom_env["PYTHONPATH"] = self._python_path + ":" + custom_env["PYTHONPATH"]
        else:
            custom_env["PYTHONPATH"] = self._python_path
        print(custom_env["PYTHONPATH"])

        script = os.path.abspath(self._test_script)
        if self._cicd:
            cmd = [sys.executable, script, '--cicd']
        else:
            cmd = [sys.executable, script]
        print(f"Running: {cmd}")
        try:
            subprocess.run(cmd, capture_output=True, check=True, env=custom_env)
        except CalledProcessError as e:
            raise RuntimeError(f"Subprocess '{cmd}' failed with STDERR: {e.stderr.decode('utf-8')}\n STDOUT: {e.stdout.decode('utf-8')}\n")


def assert_float_precision_ac(accelerator, model, precision=torch.float32):
    """
    Check if model wrapped with the accelerator decorator is loaded with a certain floating point precision.

    :param accelerator: accelerator from accelerator.Accelerator
    :param model: The model with the accelerator wrapper
    :param precision: The precision to check for
    """
    if accelerator.unwrap_model(model).dtype != precision:
        raise ValueError(
            f"Model loaded with incorrect floating point datatype {accelerator.unwrap_model(model).dtype}."
        )
