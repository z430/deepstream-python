# FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch
FROM nvcr.io/nvidia/deepstream:6.2-devel

# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video
ENV LOGLEVEL="INFO"
ENV GST_DEBUG=3
ENV GST_DEBUG_FILE=/app/output/GST_DEBUG.log
ENV CUDA_VER=11.8

RUN mkdir /app
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --allow-downgrades --allow-change-held-packages \
    --no-install-recommends build-essential ca-certificates libsm6 libxext6 curl \
    'libsm6' 'libxext6' git build-essential cmake pkg-config unzip yasm git \
    checkinstall libjpeg-dev libpng-dev libtiff-dev libunistring-dev libx265-dev \
    libnuma-dev libavcodec-dev libavformat-dev libswscale-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libxvidcore-dev x264 libx264-dev libfaac-dev \
    libmp3lame-dev libtheora-dev libfaac-dev libmp3lame-dev libvorbis-dev \
    libgtk-3-dev libatlas-base-dev gfortran libtool libc6 libc6-dev wget \
    libnuma-dev libgtk2.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp sudo tmux

# Compile Python bindings
RUN apt install python3-gi python3-dev python3-gst-1.0 python-gi-dev git \
    python3 python3-pip cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev -y \
    && apt-get install -y libgstrtspserver-1.0-0 gstreamer1.0-rtsp libgirepository1.0-dev gobject-introspection gir1.2-gst-rtsp-server-1.0

RUN wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.6/pyds-1.1.6-py3-none-linux_x86_64.whl \
    && pip3 install pyds-1.1.6-py3-none-linux_x86_64.whl

RUN git clone https://github.com/marcoslucianops/DeepStream-Yolo.git \
    && cd DeepStream-Yolo \
    && make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo

RUN pip3 install opencv-python pytz loguru cuda-python
RUN mkdir /home/cogai/ \
    && cd /home/cogai \
    && git clone -b 1.16.2 https://github.com/GStreamer/gst-rtsp-server.git  \
    && apt-get install -y libgstrtspserver-1.0 libgstreamer1.0-dev \
    && cd gst-rtsp-server/examples \
    && gcc test-launch.c -o test-launch $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-rtsp-server-1.0)

# RUN cd /app \
#     && git clone https://github.com/FFmpeg/FFmpeg.git \
#     && cd FFmpeg \
#     && ./configure --enable-shared --disable-lzma \
#     && make -j12 \
#     && make install
RUN /opt/nvidia/deepstream/deepstream/user_additional_install.sh

WORKDIR /workspace