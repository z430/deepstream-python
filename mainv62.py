import sys
from typing import List
import gi
import argparse
import signal
import os
from datetime import datetime
import pytz
from functools import partial
from inspect import signature

from ctypes import *

gi.require_version("Gst", "1.0")
from gi.repository import Gst
from gi.repository import GLib

import cv2
import pyds
from loguru import logger
from libs.platform import is_platform_aarch64
from libs.FPS import FPSMonitor

MUXER_BATCH_TIMEOUT_USEC = 33000
PGIE_CONFIG_PATH = "configs/pgies/yolov5.txt"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
TARGET_CLASSES = [1]


class Probe:
    def __init__(self, num_sources: int) -> None:
        self.fps_streams = {}
        for i in range(num_sources):
            self.fps_streams[f"stream{i}"] = FPSMonitor(i)

    def probe_fn_wrapper(self, _, info, probe_fn, get_frames=False):
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

        self.fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        if get_frames:
            probe_fn(frames, batch_meta, l_frame_meta, ll_obj_meta)
        else:
            probe_fn(batch_meta, l_frame_meta, ll_obj_meta)

        return Gst.PadProbeReturn.OK

    def wrap_probe(self, probe_fn):
        get_frames = "frames" in signature(probe_fn).parameters
        return partial(self.probe_fn_wrapper, probe_fn=probe_fn, get_frames=get_frames)

    def get_static_pad(self, element, pad_name: str = "sink"):
        pad = element.get_static_pad(pad_name)
        if not pad:
            raise AttributeError(f"Unable to get {pad_name} pad of {element.name}")

        return pad

    def add_probes(self, element, func, pad_name="src"):
        pad = self.get_static_pad(element, pad_name=pad_name)
        pad.add_probe(Gst.PadProbeType.BUFFER, self.wrap_probe(func))


def _anonymize(frames, _, l_frame_meta: List, ll_obj_meta: List[List]):
    for frame, frame_meta, l_obj_meta in zip(frames, l_frame_meta, ll_obj_meta):
        for obj_meta in l_obj_meta:
            if TARGET_CLASSES and obj_meta.class_id not in TARGET_CLASSES:
                continue

            rect_params = obj_meta.rect_params
            top = int(rect_params.top)
            left = int(rect_params.left)
            width = int(rect_params.width)
            height = int(rect_params.height)

            x1 = left
            y1 = top
            x2 = left + width
            y2 = top + height
            bbox = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(bbox, (15, 15), 60)


def get_timestamp() -> str:
    timezone = pytz.timezone("Asia/Tokyo")
    return datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")


def render_timestamp(image, timestamp):
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
    cv2.putText(image, timestamp, (text_x, text_y), font, font_scale, color, thickness)


def timestamp_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.error("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Get the GPU memory frame (surface)
        image = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        timestamp = get_timestamp()
        render_timestamp(image, timestamp)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def format_location_full_callback(splitmuxsink, fragment_id, first_sample, data):
    # Custom filename format based on fragment number and timestamp
    camera_index, output_location = data
    timezone = pytz.timezone("Asia/Tokyo")
    timestamp = datetime.now(timezone).strftime("%Y%m%dT%H%M%S")
    filename = f"{output_location}/camera_{camera_index+1:02d}-{timestamp}.mp4"
    logger.info(f"Saving file: {filename}")
    return filename


def demux_pipeline(pipeline, nvdemux, number_sources):
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
            logger.error(" Unable to create splitmuxsink")
        splitmuxsink.set_property("muxer", Gst.ElementFactory.make("mp4mux", None))
        N = 120  # 2 minutes
        splitmuxsink.set_property("async-finalize", True)
        splitmuxsink.set_property("max-size-time", N * Gst.SECOND)
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
            logger.error(" Unable to get src pad \n")
        else:
            queue_sink_pad.add_probe(Gst.PadProbeType.BUFFER, timestamp_probe, 0)

        # Link nvstreamdemux to queue
        src_pad = nvdemux.get_request_pad(f"src_{i}")
        if not src_pad:
            logger.error(f"Unable to get src pad for stream {i}")
        sink_pad = queue.get_static_pad("sink")
        if not sink_pad:
            logger.error(f"Unable to get sink pad for queue_{i}\n")
        src_pad.link(sink_pad)


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
    probe = Probe(number_sources)

    Gst.init(None)
    # Gst.debug_add_log_function(gst_log_handler, None)

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
    probe.add_probes(filter1, _anonymize)

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
