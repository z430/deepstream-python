import argparse
import os
import signal
import sys
from typing import List, Dict
from ctypes import *

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

from loguru import logger
import pyds

from libs.platform import is_platform_aarch64
from libs.utils import gst_log_handler
from libs.input_handler import build_source_bin


class Pipeline:
    """
    Basic of the gstreamer element
        +---------------------------+
        |         Elements           |
        |                            |
        |  +------+         +-----+  |
        |  | sink |  ---->  | src |  |
        |  +------+         +-----+  |
        +---------------------------+

    """

    def __init__(self, input_sources: List[str], app_config: Dict):
        Gst.init(None)
        Gst.debug_add_log_function(gst_log_handler, None)

        self.input_sources = input_sources
        self.num_sources = len(input_sources)
        self.is_live = False
        self.config = app_config

        # create gstreamer pipeline
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            logger.error("Unable to create Pipeline")
            sys.exit(1)

        # elements that will be added to the pipeline
        self.elements = []
        self.build_input()
        self.build_inference()
        self.build_output()

    def _create_element(self, factory_name, name, print_name, detail="", add=False):
        logger.info(f"Creating {print_name}")
        elm = Gst.ElementFactory.make(factory_name, name)

        if not elm:
            logger.error(f"Unable to create {print_name}")
            if detail:
                logger.error(detail)

        if add:
            self._add_element(elm)

        return elm

    def _add_element(self, element, idx=None):
        if idx:
            self.elements.insert(idx, element)
        else:
            self.elements.append(element)
        self.pipeline.add(element)

    def build_input(self):
        streammux = self._create_element(
            "nvstreammux", "streammux", "Streammux", add=True
        )
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batch-size", self.num_sources)
        streammux.set_property("batched-push-timeout", 4000000)

        self.is_live = build_source_bin(
            self.num_sources, self.input_sources, streammux, self.pipeline
        )

        self._create_element("queue", "queue", "Queue", add=True)

    def build_inference(self):
        pgie = self._create_element(
            "nvinfer", "primary-inference", "Primary Inference", add=True
        )
        pgie.set_property("config-file-path", self.config["pgie_config_path"])
        pgie_batch_size = pgie.get_property("batch-size")
        if pgie_batch_size != self.num_sources:
            logger.warning(
                "WARNING: Overriding infer-config batch-size",
                pgie_batch_size,
                " with number of sources ",
                self.num_sources,
                " \n",
            )
            pgie.set_property("batch-size", self.num_sources)

    def build_output(self):
        """Inherited classes should implement this method,
        builts the output pipeline that you desire after the capsfilter"""
        nvvidconv1 = self._create_element(
            "nvvideoconvert", "convertor1", "nvvideoconvert1", add=True
        )
        self._create_element("capsfilter", "capsfilter1", "capsfilter1", add=True)

        if not is_platform_aarch64():
            # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
            # put the probe in src of nvvidconv1
            mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            nvvidconv1.set_property("nvbuf-memory-type", mem_type)

    def run(self):
        logger.info("Starting pipeline")
        logger.info(self.elements)
