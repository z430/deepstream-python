import sys

import gi
import argparse
import signal
import os

from ctypes import *

gi.require_version("Gst", "1.0")
from gi.repository import Gst
from gi.repository import GLib

import pyds
from loguru import logger

from libs.platform import PlatformInfo
from libs.input_handler import build_source_bin
from libs.demux_pipeline import demux_pipeline
from libs.face_blur import _anonymize
from libs.probe import Probe

MUXER_BATCH_TIMEOUT_USEC = 33000
PGIE_CONFIG_PATH = "configs/pgies/yolov5.txt"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

platform_info = PlatformInfo()

ELEMENT_NAMES = {
    "streammux": "nvstreammux",
    "queue": "queue",
    "pgie": "nvinfer",
    "nvvidconv": "nvvideoconvert",
    "capsfilter": "capsfilter",
    "nvdemux": "nvstreamdemux",
}


def gst_log_handler(category, level, dfile, dfctn, dline, source, message, *user_data):
    log_message = message.get()

    # Map the log level from GStreamer to loguru
    if level == Gst.DebugLevel.WARNING:
        logger.warning(f"{dfile}:{dline} {dfctn}() - {log_message}")
    elif level == Gst.DebugLevel.ERROR:
        logger.error(f"{dfile}:{dline} {dfctn}() - {log_message}")
    else:
        logger.info(f"{dfile}:{dline} {dfctn}() - {log_message}")


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.error("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.warning("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("Error: %s: %s\n" % (err, debug))
    return True


def make_element(element_name, i):
    element = Gst.ElementFactory.make(element_name, element_name)
    if not element:
        logger.error("Unable to create {0}".format(element_name))
        sys.exit()
    element.set_property("name", "{0}-{1}".format(element_name, str(i)))
    return element


def signal_handler(sig, frame, element, loop):
    logger.warning("Interrupt received, sending EOS...")
    element.send_event(Gst.Event.new_eos())  # Send EOS to the pipeline
    GLib.timeout_add(
        5000, lambda: loop.quit()
    )  # Quit the loop after waiting a bit for EOS


def create_pipeline(number_sources):
    pipeline = Gst.Pipeline()
    if not pipeline:
        logger.error("Unable to create Pipeline")
        sys.exit()

    streammux = make_element(ELEMENT_NAMES["streammux"], 1)
    pipeline.add(streammux)
    streammux.set_property("width", IMAGE_WIDTH)
    streammux.set_property("height", IMAGE_HEIGHT)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    return pipeline, streammux


def add_elements_to_pipeline(pipeline, number_sources):
    queue1 = make_element(ELEMENT_NAMES["queue"], 1)
    pgie = make_element(ELEMENT_NAMES["pgie"], 1)
    nvvidconv = make_element(ELEMENT_NAMES["nvvidconv"], 1)
    capsfilter = make_element(ELEMENT_NAMES["capsfilter"], 1)
    capsfilter.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )
    nvdemux = make_element(ELEMENT_NAMES["nvdemux"], 1)

    if not all([queue1, pgie, nvvidconv, capsfilter, nvdemux]):
        logger.error("Unable to create one or more elements")
        sys.exit()

    pipeline.add(queue1)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(capsfilter)
    pipeline.add(nvdemux)

    pgie.set_property("config-file-path", PGIE_CONFIG_PATH)
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        logger.warning(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)


def link_elements(pipeline, streammux):
    queue1 = pipeline.get_by_name("queue-1")
    pgie = pipeline.get_by_name("nvinfer-1")
    nvvidconv = pipeline.get_by_name("nvvideoconvert-1")
    capsfilter = pipeline.get_by_name("capsfilter-1")
    nvdemux = pipeline.get_by_name("nvstreamdemux-1")

    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(capsfilter)
    capsfilter.link(nvdemux)


def main(args):
    input_sources = args
    number_sources = len(input_sources)
    probe = Probe(number_sources)

    Gst.init(None)
    Gst.debug_add_log_function(gst_log_handler, None)

    pipeline, streammux = create_pipeline(number_sources)
    is_live = build_source_bin(number_sources, input_sources, streammux, pipeline)
    if is_live:
        logger.info("Atleast one of the sources is live")
        streammux.set_property("live-source", 1)

    add_elements_to_pipeline(pipeline, number_sources)
    link_elements(pipeline, streammux)

    demux_pipeline(
        pipeline, pipeline.get_by_name(f"{ELEMENT_NAMES['nvdemux']}-1"), number_sources
    )
    probe.add_probes(
        pipeline.get_by_name(f"{ELEMENT_NAMES['capsfilter']}-1"), _anonymize
    )

    if not platform_info.is_platform_aarch64():
        # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        pipeline.get_by_name(f"{ELEMENT_NAMES['nvvidconv']}-1").set_property(
            "nvbuf-memory-type", mem_type
        )

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Generate pipeline graph
    open("pipeline.dot", "w").write(
        Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    )

    os.system("dot -Tpng pipeline.dot -o outputs/pipeline.png")

    logger.info("Now playing...")
    for i, source in enumerate(input_sources):
        logger.info(f"{i}: {source}")
    logger.info("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)

    # handler for keyboard interrupt
    signal.signal(
        signal.SIGINT,
        lambda sig, frame: signal_handler(
            sig, frame, pipeline.get_by_name(f"{ELEMENT_NAMES['queue']}-1"), loop
        ),
    )

    try:
        loop.run()
    except Exception as e:
        logger.error(f"Exception: {e}")
    finally:
        logger.info("Exiting app")
        pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input streams",
        nargs="+",
        metavar="URIs",
        default=["a"],
        required=True,
    )

    args = parser.parse_args()
    stream_paths = args.input
    return stream_paths


if __name__ == "__main__":
    stream_paths = parse_args()
    sys.exit(main(stream_paths))
