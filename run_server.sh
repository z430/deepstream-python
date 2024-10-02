# default_video_path=/media/ssd/sspf-test-videos/mutsukawa-angled-01/leave_nozzle_back_1.mp4
default_video_path=/data/long-videos/麻溝台SS_真上_20221018_1600_1700.mp4
default_port=8554

if [ -n "$1" ]; then
    video_path=$1
else
    video_path=$default_video_path
fi

if [ -n "$2" ]; then
    port=$2
else
    port=$default_port
fi

echo "Streaming ${video_path}"

/home/cogai/gst-rtsp-server/examples/test-launch \
    --port ${port} \
    "( filesrc location=${video_path} ! qtdemux ! rtph264pay name=pay0 pt=96 )"
