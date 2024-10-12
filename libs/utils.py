import gi
import yaml

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
from loguru import logger


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


def signal_handler(sig, frame, element, loop):
    logger.warning("Interrupt received, sending EOS...")
    element.send_event(Gst.Event.new_eos())  # Send EOS to the pipeline
    GLib.timeout_add(
        5000, lambda: loop.quit()
    )  # Quit the loop after waiting a bit for EOS


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
