import sys
import gi
from libs.platform import PlatformInfo
from loguru import logger
import math
import pyds
import time

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject, GLib

platform_info = PlatformInfo()

frame_count = {}
fps_streams = {}
last_update_time = {}


def tiler_src_pad_buffer_probe(pad, info, u_data):
    global frame_count, fps_streams, last_update_time

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.error("Unable to get GstBuffer")
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from the GstBuffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            # Get frame metadata
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        stream_id = frame_meta.pad_index  # Unique stream identifier
        frame_number = frame_meta.frame_num

        # Initialize counters if necessary
        if stream_id not in frame_count:
            frame_count[stream_id] = 0
            fps_streams[stream_id] = 0.0
            last_update_time[stream_id] = time.time()

        frame_count[stream_id] += 1
        current_time = time.time()
        time_diff = current_time - last_update_time[stream_id]

        if time_diff >= 1.0:
            fps = frame_count[stream_id] / time_diff
            fps_streams[stream_id] = fps
            frame_count[stream_id] = 0
            last_update_time[stream_id] = current_time
            logger.info(f"Stream {stream_id} FPS: {fps}")

        # Create display meta to overlay FPS
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]

        # Set display text
        py_nvosd_text_params.display_text = (
            f"Stream {stream_id} FPS: {fps_streams[stream_id]:.2f}"
        )

        # Text position
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Set text parameters
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 20
        py_nvosd_text_params.font_params.font_color.set(
            1.0, 1.0, 1.0, 1.0
        )  # White color

        # Set text background color
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)  # Black background

        # Add display meta to frame meta
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


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


def decodebin_child_added(child_proxy, obj, name, user_data):
    logger.info("Decodebin child added: {0}".format(name))
    if name.find("decodebin") != -1:
        obj.connect("child-added", decodebin_child_added, user_data)

    if not platform_info.is_integrated_gpu() and name.find("nvv4l2decoder") != -1:
        # Use CUDA unified memory in the pipeline so frames can be easily accessed on CPU in Python.
        # 0: NVBUF_MEM_CUDA_DEVICE, 1: NVBUF_MEM_CUDA_PINNED, 2: NVBUF_MEM_CUDA_UNIFIED
        # Dont use direct macro here like NVBUF_MEM_CUDA_UNIFIED since nvv4l2decoder uses a
        # different enum internally
        obj.set_property("cudadec-memtype", 2)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") != None:
            obj.set_property("drop-on-latency", True)


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

    Gst.Bin.add(nbin, uri_decode_bin)
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

    number_sources = len(args) - 1
    logger.info(f"Number of sources: {number_sources}")
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
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # add nvvidconv1 and filter1 to convert frames to RGBA
    # which is easier to work with in Python.
    logger.info("Creating nvvidconv1")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")

    logger.info("Creating filter1")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to create filter1 \n")
    filter1.set_property("caps", caps1)
    logger.info("creating tiler")

    logger.info("Creating nvvidconv")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    logger.info("Creating nvosd")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")

    # create caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
    )

    # make the encoder
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    encoder.set_property("bitrate", 4000000)
    if platform_info.is_integrated_gpu():
        encoder.set_property("preset-level", 1)
        encoder.set_property("insert-sps-pps", 1)

    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("batched-push-timeout", 4000000)

    pgie.set_property("config-file-path", "yolov5.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        logger.error(
            f"WARNING: Overriding infer-config batch-size ({pgie_batch_size}) with number of sources ({number_sources})"
        )
        pgie.set_property("batch-size", number_sources)

    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")

    tiler_rows = max(1, int(math.sqrt(number_sources)))
    tiler_columns = max(1, int(math.ceil((1.0 * number_sources) / tiler_rows)))

    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", 1920)
    tiler.set_property("height", 1080)

    sink = Gst.ElementFactory.make("filesink", "file-sink")
    sink.set_property("location", "output.mp4")
    sink.set_property("sync", False)
    sink.set_property("async", False)

    if not platform_info.is_integrated_gpu():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)
        nvvidconv_postosd.set_property("nvbuf-memory-type", mem_type)

    h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parse:
        sys.stderr.write(" Unable to create h264parse \n")

    qtmux = Gst.ElementFactory.make("qtmux", "qtmux")
    if not qtmux:
        sys.stderr.write(" Unable to create qtmux \n")

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(h264parse)
    pipeline.add(qtmux)
    pipeline.add(sink)

    # Attach probe to tiler's src pad
    tiler_src_pad = tiler.get_static_pad("src")
    if not tiler_src_pad:
        logger.error("Unable to get src pad of tiler")
    else:
        tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_src_pad_buffer_probe, 0)

    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(h264parse)
    h264parse.link(qtmux)
    qtmux.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except Exception as e:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
