import os
import shutil
import signal
import getpass
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

import sys
import socket

from tensorboard.program import core_plugin, TensorBoardServerException, argparse, TensorBoardPortInUseError, \
    WerkzeugServer

G_FORCE_TB_PORT = None
# Prevent TensorBoard from launching because the TB API does not support closing TB sessions.

G_PREVENT_TB_LAUNCH = False

# Force the Tensorboard socket to bind to the localhost. Use this if connected through a tunnel.
if getpass.getuser() in ["lucas", "cvds"]:
    G_FORCE_TB_PUBLIC = True
else:
    G_FORCE_TB_PUBLIC = False

# Forces tensorboard to use a specific port


def __fix__with_port_scanning(cls):
    def init(wsgi_app, flags):
        # base_port: what's the first port to which we should try to bind?
        # should_scan: if that fails, shall we try additional ports?
        # max_attempts: how many ports shall we try?
        should_scan = flags.port is None
        base_port = (
            core_plugin.DEFAULT_PORT if flags.port is None else flags.port
        )

        if base_port > 0xFFFF:
            raise TensorBoardServerException(
                "TensorBoard cannot bind to port %d > %d" % (base_port, 0xFFFF)
            )
        max_attempts = 100 if should_scan else 1
        base_port = min(base_port + max_attempts, 0x10000) - max_attempts

        for port in range(base_port, base_port + max_attempts):
            subflags = argparse.Namespace(**vars(flags))
            subflags.port = port

            with (socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s):
                status = s.connect_ex((subflags.host, port))
                if status == 0:  # 0: no error, so connection to port was successful. This means that something is already running there.
                    if not should_scan:
                        raise TensorBoardPortInUseError(f"port {port} already in use")
                else:
                    return cls(wsgi_app=wsgi_app, flags=subflags)

        # All attempts failed to bind.
        raise TensorBoardServerException(
            "TensorBoard could not bind to any port around %s "
            "(tried %d times)" % (base_port, max_attempts)
        )

    return init


def prevent_launch_tb():
    global G_PREVENT_TB_LAUNCH
    G_PREVENT_TB_LAUNCH = True


def force_public_tb():
    global G_FORCE_TB_PUBLIC
    G_FORCE_TB_PUBLIC = True


def force_port_tb(port: int = 6006):
    global G_FORCE_TB_PORT
    G_FORCE_TB_PORT = port


def launch_tb(tb_name: str, verbose=True, public=False):
    """
    Launches a tensorboard

    :param tb_name: The name of the tensorboard folder
    :param verbose: Determines if information is printed to the standard output
    :param localhost: Determines if the Tensorboard is bound to the localhost or the external ethernet interface.
    """
    global G_PREVENT_TB_LAUNCH
    global G_FORCE_TB_PUBLIC
    global G_FORCE_TB_PORT

    if G_PREVENT_TB_LAUNCH:
        if verbose:
            print(
                f"TensorBoard will not be launched. To start it manually execute: 'tensorboard --logdir {os.path.abspath(tb_name)} --host {socket.gethostbyname(socket.gethostname())}'")
        return
    tb = program.TensorBoard()
    argv = [None, '--logdir', tb_name, "--samples_per_plugin=images=1000"]
    if G_FORCE_TB_PUBLIC or public:
        argv += ["--host", socket.gethostbyname(socket.gethostname())]
    else:
        argv += ["--host", "localhost"]
    if G_FORCE_TB_PORT is not None:
        argv += ["--port", str(G_FORCE_TB_PORT)]
    tb.configure(argv=argv)
    url = tb.launch()
    if verbose:
        print(f"Tensorboard {tb_name} is available at: {url}")


def delete_tb(tb_name: str):
    """
    Delete a tensorboard in a safe way (only deletes events.out.tfevents and empty folders)

    :param tb_name: Name of the Tensorboard

    :example:
    >>> import os
    >>> w = SummaryWriter("test")
    >>> w.close()
    >>> delete_tb("test")
    >>> os.path.isdir("test")
    False
    """
    # JL: doesn't work
    # for file in glob.glob(os.path.join(tb_name, "events.out.tfevents*"), recursive=True):
    #     os.remove(file)
    # if os.path.isdir(tb_name):
    #     os.removedirs(tb_name)
    shutil.rmtree(os.path.abspath(tb_name), ignore_errors=True)


def create_tb(tb_name: str, delete_previous: bool = True, start_tb: bool = True) -> SummaryWriter:
    """
    Create a tensorboard SummaryWriter object

    :param tb_name: Name of the tensorboard writer
    :param delete_previous: Should the previous writer with the same name be deleted or should this value be appended?
    :param start_tb: Should a TensorBoard be started with this SummaryWriter.
    :return: The SummaryWriter object

    :example:
    >>> writer = create_tb(tb_name="test_tensorboard", start_tb=False)
    >>> isinstance(writer, SummaryWriter)
    True
    """
    if delete_previous:
        delete_tb(tb_name)
    w = SummaryWriter(tb_name)
    if start_tb:
        launch_tb(tb_name)
    return w


# werkzeug
sys.modules['tensorboard.program'].__setattr__('create_port_scanning_werkzeug_server',
                                               __fix__with_port_scanning(WerkzeugServer))
