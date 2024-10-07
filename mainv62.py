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

from libs.platform import is_platform_aarch64
from libs.input_handler import build_source_bin
from libs.demux_pipeline import demux_pipeline
from libs.face_blur import _anonymize
from libs.probe import Probe

MUXER_BATCH_TIMEOUT_USEC = 33000
PGIE_CONFIG_PATH = "configs/pgies/yolov5.txt"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


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


def cb_newpad(decodebin, decoder_src_pad, data):
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
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") != None:
            Object.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    print("Creating source bin")
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
    uri_decode_bin.set_property("rtsp-reconnect-interval", 10)
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def make_element(element_name, i):
    """
    Creates a Gstreamer element with unique name
    Unique name is created by adding element type and index e.g. `element_name-i`
    Unique name is essential for all the element in pipeline otherwise gstreamer will throw exception.
    :param element_name: The name of the element to create
    :param i: the index of the element in the pipeline
    :return: A Gst.Element object
    """
    element = Gst.ElementFactory.make(element_name, element_name)
    if not element:
        sys.stderr.write(" Unable to create {0}".format(element_name))
    element.set_property("name", "{0}-{1}".format(element_name, str(i)))
    return element


def signal_handler(sig, frame, element, loop):
    logger.warning("Interrupt received, sending EOS...")
    element.send_event(Gst.Event.new_eos())  # Send EOS to the pipeline
    GLib.timeout_add(
        5000, lambda: loop.quit()
    )  # Quit the loop after waiting a bit for EOS


def main(args, requested_pgie=None, config=None, disable_probe=False):
    input_sources = args
    number_sources = len(input_sources)

    # Standard GStreamer initialization
    Gst.init(None)
    # Gst.debug_add_log_function(gst_log_handler, None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = input_sources[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    pipeline.add(queue1)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    print("Creating nvstreamdemux \n ")
    nvstreamdemux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
    if not nvstreamdemux:
        sys.stderr.write(" Unable to create nvstreamdemux \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property("live-source", 1)

    streammux.set_property("width", 960)
    streammux.set_property("height", 540)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", 4000000)
    pgie.set_property("config-file-path", PGIE_CONFIG_PATH)
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print(
            "WARNING: Overriding infer-config batch-size",
            pgie_batch_size,
            " with number of sources ",
            number_sources,
            " \n",
        )
        pgie.set_property("batch-size", number_sources)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(nvstreamdemux)

    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    pipeline.add(nvvidconv1)
    pipeline.add(filter1)

    # linking
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(nvstreamdemux)

    demux_pipeline(pipeline, nvstreamdemux, number_sources)

    if not is_platform_aarch64():
        # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    pipeline.set_state(Gst.State.PLAYING)
    signal.signal(
        signal.SIGINT,
        lambda sig, frame: signal_handler(
            sig, frame, pipeline.get_by_name("queue1"), loop
        ),
    )
    try:
        loop.run()
    except Exception as e:
        print(e)
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
    stream_path = parse_args()
    sys.exit(main(stream_path))
