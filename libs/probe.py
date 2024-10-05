from functools import partial
from inspect import signature

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import pyds


def probe_fn_wrapper(_, info, probe_fn, get_frames=False):
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


def wrap_probe(probe_fn):
    get_frames = "frames" in signature(probe_fn).parameters
    return partial(probe_fn_wrapper, probe_fn=probe_fn, get_frames=get_frames)


def get_static_pad(element, pad_name: str = "sink"):
    pad = element.get_static_pad(pad_name)
    if not pad:
        raise AttributeError(f"Unable to get {pad_name} pad of {element.name}")

    return pad


def add_probes(element, func, pad_name="src"):
    pad = get_static_pad(element, pad_name=pad_name)
    pad.add_probe(Gst.PadProbeType.BUFFER, wrap_probe(func))
