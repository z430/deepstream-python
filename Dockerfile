FROM nvcr.io/nvidia/deepstream:7.0-triton-multiarch

# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video
ENV LOGLEVEL="INFO"
ENV GST_DEBUG=2
ENV GST_DEBUG_FILE=/app/output/GST_DEBUG.log
ENV CUDA_VER=12.2

RUN mkdir /app
WORKDIR /app

RUN apt update
RUN apt install -y python3-gi python3-dev python3-gst-1.0 python3-numpy python3-opencv

# Compile Python bindings
RUN apt install python3-gi python3-dev python3-gst-1.0 python-gi-dev git \
    python3 python3-pip cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev -y \
    && apt-get install -y libgstrtspserver-1.0-0 gstreamer1.0-rtsp libgirepository1.0-dev gobject-introspection gir1.2-gst-rtsp-server-1.0

RUN wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.11/pyds-1.1.11-py3-none-linux_x86_64.whl \
    && pip3 install pyds-1.1.11-py3-none-linux_x86_64.whl

RUN git clone https://github.com/marcoslucianops/DeepStream-Yolo.git \
    && cd DeepStream-Yolo \
    && make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo

