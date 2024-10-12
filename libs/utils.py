import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
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
