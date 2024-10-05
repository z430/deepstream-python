import sys

import gi
import argparse
import signal
from functools import partial
from inspect import signature
from typing import List
from ctypes import *

gi.require_version("Gst", "1.0")
from gi.repository import Gst
from gi.repository import GLib

import pyds
from datetime import datetime
import pytz



def is_aarch64():
    return platform.uname()[4] == "aarch64"

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


def cb_newpad(decodebin, decoder_src_pad, data):
    """
    The function is called when a new pad is created by the decodebin.
    The function checks if the new pad is for video and not audio.
    If the new pad is for video, the function checks if the pad caps contain NVMM memory features.
    If the pad caps contain NVMM memory features, the function links the decodebin pad to the source bin
    ghost pad.
    If the pad caps do not contain NVMM memory features, the function prints an error message.
    :param decodebin: The decodebin element that is creating the new pad
    :param decoder_src_pad: The source pad created by the decodebin element
    :param data: This is the data that was passed to the callback function. In this case, it is the
    source_bin
    """
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    """
    If the child added to the decodebin is another decodebin, connect to its child-added signal. If the
    child added is a source, set its drop-on-latency property to True.

    :param child_proxy: The child element that was added to the decodebin
    :param Object: The object that emitted the signal
    :param name: The name of the element that was added
    :param user_data: This is a pointer to the data that you want to pass to the callback function
    """
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") != None:
            Object.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    """
    It creates a GstBin, adds a uridecodebin to it, and connects the uridecodebin's pad-added signal to
    a callback function

    :param index: The index of the source bin
    :param uri: The URI of the video file to be played
    :return: A bin with a uri decode bin and a ghost pad.
    """
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    # uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin-{index}")
    uri_decode_bin.set_property("rtsp-reconnect-interval", 10)
    uri_decode_bin.set_property("latency", 200)
    # uri_decode_bin.set_property("cudadec-memtype", 0)
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

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


def main(args):
    input_sources = args
    number_sources = len(input_sources)
 
    Gst.init(None)

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
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    pipeline.add(queue1)

    # linking
    streammux.link(queue1)
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(input_sources):
        print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events
    open("pipeline.dot", "w").write(
        Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
    )
    pipeline.set_state(Gst.State.PLAYING) 

    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="deepstream_demux_multi_in_multi_out.py",
        description="deepstream-demux-multi-in-multi-out takes multiple URI streams as input"
        "and uses `nvstreamdemux` to split batches and output separate buffer/streams",
    )
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

