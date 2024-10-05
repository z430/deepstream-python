from loguru import logger
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


def cb_newpad(decodebin, decoder_src_pad, data):
    logger.info("In cb_newpad")
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
                logger.error("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            logger.error(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    """
    If the child added to the decodebin is another decodebin, connect to its child-added signal. If the
    child added is a source, set its drop-on-latency property to True.

    :param child_proxy: The child element that was added to the decodebin
    :param Object: The object that emitted the signal
    :param name: The name of the element that was added
    :param user_data: This is a pointer to the data that you want to pass to the callback function
    """
    logger.info(f"Decodebin child added: {name}")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") is None:
            Object.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    logger.info("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        logger.error(" Unable to create source bin")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    # uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    uri_decode_bin = Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
    uri_decode_bin.set_property("rtsp-reconnect-interval", 10)
    # uri_decode_bin.set_property("latency", 0)
    # uri_decode_bin.set_property("cudadec-memtype", 0)
    if not uri_decode_bin:
        logger.error(" Unable to create uri decode bin")
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
        logger.error(" Failed to add ghost pad in source bin")
        return None
    return nbin


def build_source_bin(number_sources, input_sources, streammux, pipeline):
    is_live = False
    for i in range(number_sources):
        logger.info(f"Creating source_bin {input_sources[i]}")
        uri_name = input_sources[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            logger.error("Unable to create source bin")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            logger.error("Unable to create sink pad bin")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            logger.error("Unable to create src pad bin")
        srcpad.link(sinkpad)
    return is_live
