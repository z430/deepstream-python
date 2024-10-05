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
platform_info = PlatformInfo()

fps_streams = {}


def bus_call(bus, message, loop):
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


def make_element(element_name, i):
    element = Gst.ElementFactory.make(element_name, element_name)
    if not element:
        sys.stderr.write(" Unable to create {0}".format(element_name))
    element.set_property("name", "{0}-{1}".format(element_name, str(i)))
    return element


def signal_handler(sig, frame, pipeline, loop):
    logger.warning("Interrupt received, sending EOS...")
    pipeline.send_event(Gst.Event.new_eos())  # Send EOS to the pipeline
    GLib.timeout_add(
        5000, lambda: loop.quit()
    )  # Quit the loop after waiting a bit for EOS


def main(args):
    input_sources = args
    number_sources = len(input_sources)
    probe = Probe(number_sources)

    Gst.init(None)

    # Create gstreamer elements */
    logger.info("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    streammux = make_element("nvstreammux", 1)
    pipeline.add(streammux)
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

    is_live = build_source_bin(number_sources, input_sources, streammux, pipeline)
    if is_live:
        logger.info("Atleast one of the sources is live")
        streammux.set_property("live-source", 1)

    queue1 = make_element("queue", 1)
    pipeline.add(queue1)

    pgie = make_element("nvinfer", 1)
    pgie.set_property("config-file-path", "configs/pgies/yolov5.txt")
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
    pipeline.add(pgie)

    nvvidconv = make_element("nvvideoconvert", 1)
    pipeline.add(nvvidconv)

    capsfilter = make_element("capsfilter", 1)
    capsfilter.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )
    pipeline.add(capsfilter)

    nvdemux = Gst.ElementFactory.make("nvstreamdemux", "nvdemux")
    nvdemux = make_element("nvstreamdemux", 1)
    pipeline.add(nvdemux)

    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(capsfilter)
    capsfilter.link(nvdemux)

    demux_pipeline(pipeline, nvdemux, number_sources)
    probe.add_probes(capsfilter, _anonymize)

    if not platform_info.is_platform_aarch64():
        # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)

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
        lambda sig, frame: signal_handler(sig, frame, queue1, loop),
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
