from datetime import datetime
import pytz

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cv2
from loguru import logger
import pyds


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
        src_pad = nvdemux.request_pad_simple(f"src_{i}")
        if not src_pad:
            logger.error(f"Unable to get src pad for stream {i}")
        sink_pad = queue.get_static_pad("sink")
        if not sink_pad:
            logger.error(f"Unable to get sink pad for queue_{i}\n")
        src_pad.link(sink_pad)
