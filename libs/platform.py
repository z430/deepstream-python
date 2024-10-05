import sys
import platform
from threading import Lock
from cuda import cudart
from cuda import cuda
from loguru import logger

guard_platform_info = Lock()


class PlatformInfo:
    def __init__(self):
        self.is_aarch64_platform = False
        self.is_aarch64_verified = False

    def is_platform_aarch64(self):
        # Check if platform is aarch64 using uname
        if not self.is_aarch64_verified:
            if platform.uname()[4] == "aarch64":
                self.is_aarch64_platform = True
            self.is_aarch64_verified = True
        return self.is_aarch64_platform


sys.path.append("/opt/nvidia/deepstream/deepstream/lib")
