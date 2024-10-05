import configparser
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from functools import partial
from inspect import signature
from pathlib import Path
from typing import List, Optional
import signal

gi.require_version("Gst", "1.0")

import pytz
from loguru import logger
import cv2
import numpy as np
from gi.repository import GLib, Gst

from utils.fps import FPSMonitor


MUXER_BATCH_TIMEOUT_USEC = 33000


class DSPipeline:
    def __init__(
        self,
        video_uris: List[str],
        pgie_config_path: Path,
        output_video_path: Path = Path("outputs"),
        input_width: int = 1920,
        input_height: int = 1080,
    ):
        self.video_uris = video_uris
        self.num_sources = len(self.video_uris)
        self.input_width = input_width
        self.input_height = input_height

        self.pgie_config_path = pgie_config_path
        self.fps_streams = {}

        for i in range(self.num_sources):
            self.fps_streams[f"stream{i}"] = FPSMonitor(i)

        logger.info(f"Playing from URI {[str(uri) for uri in self.video_uris]}")
        Gst.init(None)

        logger.info("Creating Pipeline")
        self.pipeline = Gst.Pipeline()

        if not self.pipeline:
            logger.error("Failed to create Pipeline")

        self.is_live = False
        self.elements = {}

        self._create_elements()
        self._link_elements()
        self._add_probes()

    def __add_element(self, element, idx=None):
        if idx:
            self.elements.insert(idx, element)
        else:
            self.elements.append(element)

        self.pipeline.add(element)

    def __create_element(self, factory_name, name, print_name, detail="", add=True):
        """Creates an element with Gst Element Factory make.

        Return the element if successfully created, otherwise print to stderr and return None.
        """
        logger.info(f"Creating {print_name}")
        elm = Gst.ElementFactory.make(factory_name, name)

        if not elm:
            logger.error(f"Unable to create {print_name}")
            if detail:
                logger.error(detail)

        if add:
            self._add_element(elm)

        return elm

    def __cb_newpad(self, decodebin, decoder_src_pad, data):
        caps = decoder_src_pad.get_current_caps()
        if not caps:
            caps = decoder_src_pad.query_caps()
        gststruct = caps.get_structure(0)
        gstname = gststruct.get_name()
        source_bin = data
        features = caps.get_features(0)

        if gstname.find("video") != -1:
            if features.contains("memory:NVMM"):
                bin_ghost_pad = source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    logger.error(
                        "Failed to link decoder src pad to source bin ghost pad\n"
                    )
            else:
                logger.error(" Error: Decodebin did not pick nvidia decoder plugin.\n")

    def __decodebin_child_added(self, child_proxy, Object, name, user_data):
        if name.find("decodebin") != -1:
            Object.connect("child-added", self.__decodebin_child_added, user_data)

        if "source" in name:
            source_element = child_proxy.get_by_name("source")
            if source_element.find_property("drop-on-latency") is None:
                Object.set_property("drop-on-latency", True)

    def __create_source_bin(self, index, uri):
        bin_name = f"source-bin-{index:02d}"
        logger.info(f"Creating source bin {bin_name}")
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            logger.error(" Unable to create source bin")

        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.
        uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
        uri_decode_bin.set_property("rtsp-reconnect-interval", 10)
        uri_decode_bin.set_property("latency", 0)
        if not uri_decode_bin:
            logger.error(" Unable to create uri decode bin")

        # We set the input uri to the source element
        uri_decode_bin.set_property("uri", uri)
        # Connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has beed created by the decodebin
        uri_decode_bin.connect("pad-added", self.__cb_newpad, nbin)
        uri_decode_bin.connect("child-added", self.__decodebin_child_added, nbin)

        # We need to create a ghost pad for the source bin which will act as a proxy
        # for the video decoder src pad. The ghost pad will not have a target right
        # now. Once the decode bin creates the video decoder and generates the
        # cb_newpad callback, we will set the ghost pad target to the video decoder
        # src pad.
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin

    def __create_input_bin(self):
        for i in range(self.num_sources):
            logger.info("Creating source_bin ", i, " \n ")
            uri_name = self.video_uris[i]
            if uri_name.find("rtsp://") == 0:
                self.is_live = True
            source_bin = self.__create_source_bin(i, uri_name)
            if not source_bin:
                sys.stderr.write("Unable to create source bin \n")
            self.pipeline.add(source_bin)
            padname = "sink_%u" % i
            sinkpad = self.streammux.request_pad_simple(padname)
            if not sinkpad:
                sys.stderr.write("Unable to create sink pad bin \n")
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                sys.stderr.write("Unable to create src pad bin \n")
            srcpad.link(sinkpad)

    def _create_streammux(self):
        streammux = self.__create_element("nvstreammux", "stream-muxer", "Stream mux")
        streammux.set_property("width", self.input_width)
        streammux.set_property("height", self.input_height)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
        streammux.set_property("live-source", self.is_live)

        return streammux

    def _create_elements(self):
        self.streammux = self._create_streammux()
        self.source_bin = self.__create_input_bin()

        self.pgie = self._create_element("nvinfer", "primary-inference", "PGIE")
        self.pgie.set_property("config-file-path", self.pgie_config_path)

        if not is_aarch64():
            # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
            mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            self.nvvidconv1.set_property("nvbuf-memory-type", mem_type)
            self.tiler.set_property("nvbuf-memory-type", mem_type)
            self.nvvidconv2.set_property("nvbuf-memory-type", mem_type)

    @staticmethod
    def _link_sequential(elements: list):
        for i in range(0, len(elements) - 1):
            elements[i].link(elements[i + 1])

    def _link_elements(self):
        self.logger.info(f"Linking elements in the Pipeline: {self}")

        sinkpad = self.streammux.get_request_pad("sink_0")
        if not sinkpad:
            self.logger.error("Unable to get the sink pad of streammux")
        srcpad = self.source_bin.get_static_pad("src")
        if not srcpad:
            self.logger.error("Unable to get source pad of decoder")
        srcpad.link(sinkpad)

        self._link_sequential(self.elements[1:])

    def _probe_fn_wrapper(self, _, info, probe_fn, get_frames=False):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            self.logger.error("Unable to get GstBuffer")
            return

        frames = []
        l_frame_meta = []
        ll_obj_meta = []
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if get_frames:
                frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frames.append(frame)

            l_frame_meta.append(frame_meta)
            l_obj_meta = []

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                l_obj_meta.append(obj_meta)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            ll_obj_meta.append(l_obj_meta)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        if get_frames:
            probe_fn(frames, batch_meta, l_frame_meta, ll_obj_meta)
        else:
            probe_fn(batch_meta, l_frame_meta, ll_obj_meta)

        return Gst.PadProbeReturn.OK

    def _wrap_probe(self, probe_fn):
        get_frames = "frames" in signature(probe_fn).parameters
        return partial(self._probe_fn_wrapper, probe_fn=probe_fn, get_frames=get_frames)

    @staticmethod
    def _get_static_pad(element, pad_name: str = "sink"):
        pad = element.get_static_pad(pad_name)
        if not pad:
            raise AttributeError(f"Unable to get {pad_name} pad of {element.name}")

        return pad

    def _add_probes(self):
        tiler_sinkpad = self._get_static_pad(self.tiler, pad_name="sink")

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End-of-stream\n")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            # loop.quit()
        return True

    def release(self):
        """Release resources and cleanup."""
        pass

    def signal_handler(self, sig, frame, pipeline, loop):
        logger.warning("Interrupt received, sending EOS...")
        pipeline.send_event(Gst.Event.new_eos())  # Send EOS to the pipeline
        GLib.timeout_add(
            5000, lambda: loop.quit()
        )  # Quit the loop after waiting a bit for EOS

    def run(self):
        # create an event loop and feed gstreamer bus mesages to it
        loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, loop)

        print("Starting pipeline \n")
        # start play back and listed to events
        open("pipeline.dot", "w").write(
            Gst.debug_bin_to_dot_data(self.pipeline, Gst.DebugGraphDetails.ALL)
        )
        self.pipeline.set_state(Gst.State.PLAYING)
        signal.signal(
            signal.SIGINT,
            lambda sig, frame: self.signal_handler(sig, frame, queue1, loop),
        )

        try:
            loop.run()
        except Exception as e:
            logger.error(f"Error: {e}")
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)
