WORK_WS=/root/catkin_ws
DOCKERIMAGE="canyueduxuan/tensorrt:22.12_ros_opencv"
xhost +
CURRENT_DIR=$(pwd)
docker run -it --rm --runtime=nvidia --gpus all  --net=host -v ${CURRENT_DIR}:${WORK_WS} \
    -v /dev/:/dev/ --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  --name="tensorrt_scripts" ${DOCKERIMAGE} /bin/bash 
