apt install python3-gi python3-dev python3-gst-1.0 python-gi-dev git meson \
    python3 python3-pip python3.10-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev

cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/
git submodule update --init

apt-get install -y apt-transport-https ca-certificates -y
update-ca-certificates

cd 3rdparty/gstreamer/subprojects/gst-python/
meson setup build
cd build
ninja
ninja install

cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings
mkdir build
cd build
cmake ..
make -j$(nproc)
