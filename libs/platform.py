import sys
import platform
from threading import Lock
from cuda import cudart
from cuda import cuda
from loguru import logger

guard_platform_info = Lock()


def is_platform_aarch64():
    # Check if platform is aarch64 using uname
    if platform.uname()[4] == "aarch64":
        return True
    return False


sys.path.append("/opt/nvidia/deepstream/deepstream/lib")
