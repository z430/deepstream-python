import sys
import gi
from libs.platform import PlatformInfo
import ctypes
import configparser
import time
import os
from loguru import logger

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject, GLib

platform_info = PlatformInfo()


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print("Error: {}, {}".format(err, debug))
        loop.quit()
    return True


def on_rtsp_pad_added(src, pad, depay):
    sink_pad = depay.get_static_pad("sink")
    if not sink_pad.is_linked():
        pad.link(sink_pad)


def on_demux_pad_added(src, pad, h264parser):
    sink_pad = h264parser.get_static_pad("sink")
    if not sink_pad.is_linked():
        pad.link(sink_pad)


def cb_newpad(decodebin, decoder_src_pad, data):
    logger.info("In cb_newpad")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # need to check if the pad created by the decodebin is for video and not audio
    if gstname.find("video") != -1:
        # link the decodebin pad only if decodebin has picked nvidia decoder plugin
        # nvdec_*. we do this by checking if the pad caps contain NVMM memory features
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                logger.error("Failed to link decoder src pad to source bin ghost pad")
        else:
            logger.error("Error: Decodebin did not pick nvidia decoder plugin.")


def decodebin_child_added(child_proxy, Object, name, user_data):
    logger.info("Decodebin child added: {0}".format(name))
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if not platform_info.is_integrated_gpu() and name.find("nvv4l2decoder") != -1:
        # Use CUDA unified memory in the pipeline so frames can be easily accessed on CPU in Python.
        # 0: NVBUF_MEM_CUDA_DEVICE, 1: NVBUF_MEM_CUDA_PINNED, 2: NVBUF_MEM_CUDA_UNIFIED
        # Dont use direct macro here like NVBUF_MEM_CUDA_UNIFIED since nvv4l2decoder uses a
        # different enum internally
        Object.set_property("cudadec-memtype", 2)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") != None:
            Object.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    logger.info("Creating source bin")
    # create a source GstBin to abstract this bin's content from the rest of the pipeline
    bin_name = f"source-bin-{index}"
    logger.info(f"Creating source bin {bin_name}")
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        logger.error("Failed to create source bin")
        return None

    # Source element for reading from the uri
    # We will use decodebin and let it figure out the container format
    # stream and the codec and plug the appropriate demux and decode plugins
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        logger.error("Failed to create uri decode bin")
        return None

    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has been created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # we need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right now.
    # Once the decode bin creates the video decoder and provides the src pad, we will
    # set the ghost pad target to the video decoder src pad.

    Gst.bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        logger.error("Failed to add ghost pad in source bin")
        return None
    return nbin


def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write(
            "usage: %s <uri1> [uri2] ... [uriN] <folder to save frames>\n" % args[0]
        )
        sys.exit(1)

    number_sources = len(args) - 2
    # Standard GStreamer initialization
    Gst.init(None)

    # Create the pipeline
    pipeline = Gst.Pipeline()
    is_live = False
    logger.info("Creating Pipeline")

    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        sys.exit(1)

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux \n")
    pipeline.add(streammux)

    for i in range(number_sources):
        logger.info("Creating source bin")
        uri_name = args[i + 1]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        else:
            is_live = False
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            logger.error(f"Failed to create source bin for {uri_name}")
        pipeline.add(source_bin)
        pad_name = f"sink_{i}"
        sinkpad = streammux.request_pad_simple(pad_name)
        if not sinkpad:
            logger.error(f"Unable to create sink pad bin {i}")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            logger.error(f"Unable to create src pad bin {i}")
        srcpad.link(sinkpad)

    logger.info("Creating PGIE")

    # if input_uri.startswith("rtsp://"):
    #     # RTSP Source
    #     source = Gst.ElementFactory.make("rtspsrc", "rtsp-source")
    #     source.set_property("location", input_uri)
    #     source.set_property("latency", 200)
    #
    #     depay = Gst.ElementFactory.make("rtph264depay", "rtp-depay")
    #     h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    #     decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    # else:
    #     # File Source
    #     source = Gst.ElementFactory.make("filesrc", "file-source")
    #     source.set_property("location", input_uri)
    #
    #     demuxer = Gst.ElementFactory.make("qtdemux", "qt-demuxer")
    #     h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    #     decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")

    # Stream Muxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 40000)

    # Primary Inference
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create pgie\n")
    pgie.set_property("config-file-path", "yolov5.txt")

    # OSD
    nvdsosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # Encoder and Muxer
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
    encoder.set_property("bitrate", 4000000)
    codeparser = Gst.ElementFactory.make("h264parse", "h264-parser2")
    container = Gst.ElementFactory.make("qtmux", "qt-muxer")

    # Sink
    sink = Gst.ElementFactory.make("filesink", "file-sink")
    sink.set_property("location", "output.mp4")
    sink.set_property("sync", False)
    sink.set_property("async", False)

    # Check if all elements are created
    elements = [
        source,
        decoder,
        streammux,
        pgie,
        nvdsosd,
        encoder,
        codeparser,
        container,
        sink,
    ]

    if input_uri.startswith("rtsp://"):
        elements.extend([depay, h264parser])
    else:
        elements.extend([demuxer, h264parser])

    for elem in elements:
        if not elem:
            sys.stderr.write("Unable to create element: {}\n".format(elem))
            sys.exit(1)

    # Add elements to the pipeline
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvdsosd)
    pipeline.add(encoder)
    pipeline.add(codeparser)
    pipeline.add(container)
    pipeline.add(sink)

    if input_uri.startswith("rtsp://"):
        pipeline.add(source)
        pipeline.add(depay)
        pipeline.add(h264parser)
        pipeline.add(decoder)
    else:
        pipeline.add(source)
        pipeline.add(demuxer)
        pipeline.add(h264parser)
        pipeline.add(decoder)

    # Link the elements
    if input_uri.startswith("rtsp://"):
        source.connect("pad-added", on_rtsp_pad_added, depay)
        depay.link(h264parser)
        h264parser.link(decoder)
    else:
        demuxer.connect("pad-added", on_demux_pad_added, h264parser)
        source.link(demuxer)
        h264parser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write("Unable to get sink pad of streammux\n")
        sys.exit(1)

    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write("Unable to get src pad of decoder\n")
        sys.exit(1)

    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(nvdsosd)
    nvdsosd.link(encoder)
    encoder.link(codeparser)
    codeparser.link(container)
    container.link(sink)

    # Create and event loop and feed GStreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    # Cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
