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

MUXER_OUTPUT_WIDTH = 540
MUXER_OUTPUT_HEIGHT = 540  # 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
from libs.platform import PlatformInfo
import platform
from libs.FPS import FPSMonitor

import cv2

target_classes = [1]
PGIE_CLASS_ID_PERSON = 0
PGIE_CLASS_ID_HEAD = 1

pgie_classes_str = ["Person", "Head"]
platform_info = PlatformInfo()

fps_streams = {}


def is_aarch64():
    return platform.uname()[4] == "aarch64"


def timestamp_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta

        # Get the GPU memory frame (surface)
        image = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        timezone = pytz.timezone("Asia/Tokyo")
        timestamp = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")

        # Define the font and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)  # White text color

        # Get the text size
        text_size = cv2.getTextSize(timestamp, font, font_scale, thickness)[0]

        # Set the position for the text (bottom right corner of the image)
        text_x = image.shape[1] - text_size[0] - 10  # 10 pixels from the right
        text_y = image.shape[0] - 10  # 10 pixels from the bottom

        # Create a black rectangle with opacity 0.6
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 0, 0),
            -1,
        )

        # Apply opacity to the rectangle
        alpha = 0.4
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Put the timestamp text on the image
        cv2.putText(
            image, timestamp, (text_x, text_y), font, font_scale, color, thickness
        )

        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def _anonymize_bbox(image, obj_meta, mode="blur"):
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)

    x1 = left
    y1 = top
    x2 = left + width
    y2 = top + height

    if mode == "blur":
        bbox = image[y1:y2, x1:x2]
        image[y1:y2, x1:x2] = cv2.GaussianBlur(bbox, (15, 15), 60)
    elif mode == "pixelate":
        reshape_factor = 18
        min_dim = 16
        bbox = image[y1:y2, x1:x2]
        h, w, _ = bbox.shape
        new_shape = (
            max(min_dim, int(w / reshape_factor)),
            max(min_dim, int(h / reshape_factor)),
        )
        bbox = cv2.resize(bbox, new_shape, interpolation=cv2.INTER_LINEAR)
        image[y1:y2, x1:x2] = cv2.resize(bbox, (w, h), interpolation=cv2.INTER_NEAREST)
    elif mode == "fill":
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=-1)
    else:
        raise ValueError(f"Invalid anonymization mode '{mode}'.")

    return image


def _anonymize(frames, _, l_frame_meta: List, ll_obj_meta: List[List]):
    for frame, frame_meta, l_obj_meta in zip(frames, l_frame_meta, ll_obj_meta):
        for obj_meta in l_obj_meta:
            if target_classes and obj_meta.class_id not in target_classes:
                continue

            frame = _anonymize_bbox(frame, obj_meta)


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
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    print("gstname=", gstname)
    if gstname.find("video") != -1:
        print("features=", features)
        if features.contains("memory:NVMM"):
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
    uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
    uri_decode_bin.set_property("rtsp-reconnect-interval", 10)
    # uri_decode_bin.set_property("latency", 0)
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


def signal_handler(sig, frame, pipeline, loop):
    print("Interrupt received, sending EOS...")
    pipeline.send_event(Gst.Event.new_eos())  # Send EOS to the pipeline
    GLib.timeout_add(
        5000, lambda: loop.quit()
    )  # Quit the loop after waiting a bit for EOS


def _probe_fn_wrapper(_, info, probe_fn, get_frames=False):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
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


def _wrap_probe(probe_fn):
    get_frames = "frames" in signature(probe_fn).parameters
    return partial(_probe_fn_wrapper, probe_fn=probe_fn, get_frames=get_frames)


def _get_static_pad(element, pad_name: str = "sink"):
    pad = element.get_static_pad(pad_name)
    if not pad:
        raise AttributeError(f"Unable to get {pad_name} pad of {element.name}")

    return pad


def _add_probes(element, func):
    _sinkpad = _get_static_pad(element, pad_name="src")
    _sinkpad.add_probe(Gst.PadProbeType.BUFFER, _wrap_probe(func))


def format_location_full_callback(splitmuxsink, fragment_id, first_sample, data):
    # Custom filename format based on fragment number and timestamp
    camera_index, output_location = data
    timezone = pytz.timezone("Asia/Tokyo")
    timestamp = datetime.now(timezone).strftime("%Y%m%dT%H%M%S")
    filename = f"{output_location}/camera_{camera_index+1:02d}-{timestamp}.mp4"
    print(f"Saving file: {filename}")
    return filename


def main(args):
    input_sources = args
    number_sources = len(input_sources)

    for i in range(number_sources):
        fps_streams[f"stream{i}"] = FPSMonitor(i)
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

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    nvvidconv = make_element("nvvideoconvert", 1)
    capsfilter = make_element("capsfilter", 1)
    capsfilter.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    )

    nvdemux = Gst.ElementFactory.make("nvstreamdemux", "nvdemux")
    if not nvdemux:
        sys.stderr.write(" Unable to create nvstreamdemux \n")
    pipeline.add(nvdemux)

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property("live-source", 1)

    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    pgie.set_property("config-file-path", "configs/pgies/yolov5.txt")
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
    pipeline.add(nvvidconv)
    pipeline.add(capsfilter)

    # linking
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(capsfilter)
    capsfilter.link(nvdemux)

    for i in range(number_sources):
        # Create queue
        queue = Gst.ElementFactory.make("queue", f"queue_{i}")
        pipeline.add(queue)

        # Create nvvideoconvert
        videoconvert = Gst.ElementFactory.make("nvvideoconvert", f"videoconvert_{i}")
        pipeline.add(videoconvert)

        filter = Gst.ElementFactory.make("capsfilter", f"capsfilter_{i}")
        filter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        )
        pipeline.add(filter)

        # Create encoder
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"encoder_{i}")
        encoder.set_property("bitrate", 4000000)
        pipeline.add(encoder)

        codeparser = Gst.ElementFactory.make(
            "h264parse",
            f"h264-parser_{i}",
        )
        pipeline.add(codeparser)

        # Create splitmuxsink
        output_location = "outputs"
        splitmuxsink = Gst.ElementFactory.make("splitmuxsink", f"splitmuxsink_{i}")
        if not splitmuxsink:
            sys.stderr.write(" Unable to create splitmuxsink \n")
        splitmuxsink.set_property("muxer", Gst.ElementFactory.make("mp4mux", None))
        N = 120  # 2 minutes
        splitmuxsink.set_property("async-finalize", True)
        splitmuxsink.set_property("max-size-time", N * Gst.SECOND)
        # splitmuxsink.set_property("sink", "filesink")
        # splitmuxsink.set_property("sync", True)
        sink_props = Gst.Structure.new_empty("properties")
        sink_props.set_value("sync", True)
        splitmuxsink.set_property("sink-properties", sink_props)
        splitmuxsink.connect(
            "format-location-full", format_location_full_callback, (i, output_location)
        )
        pipeline.add(splitmuxsink)

        # Link the elements
        queue.link(videoconvert)
        videoconvert.link(filter)
        filter.link(encoder)
        encoder.link(codeparser)
        codeparser.link(splitmuxsink)

        queue_sink_pad = queue.get_static_pad("sink")
        if not queue_sink_pad:
            sys.stderr.write(" Unable to get src pad \n")
        else:
            queue_sink_pad.add_probe(Gst.PadProbeType.BUFFER, timestamp_probe, 0)

        # Link nvstreamdemux to queue
        src_pad = nvdemux.request_pad_simple(f"src_{i}")
        if not src_pad:
            sys.stderr.write(f"Unable to get src pad for stream {i}\n")
        sink_pad = queue.get_static_pad("sink")
        if not sink_pad:
            sys.stderr.write(f"Unable to get sink pad for queue_{i}\n")
        src_pad.link(sink_pad)

        # if not is_aarch64:
        #     mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        #     videoconvert.set_property("nvbuf-memory-type", mem_type)

    if not is_aarch64():
        # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)

    _add_probes(capsfilter, _anonymize)

    # create an event loop and feed gstreamer bus mesages to it
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
    signal.signal(
        signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, queue1, loop)
    )

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
