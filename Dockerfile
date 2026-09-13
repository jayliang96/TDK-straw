# ROS2 Humble（Ubuntu 22.04）＋ straw_detector / straw_interfaces 的執行環境。
# 主機是 24.04 也沒關係，Humble 只活在容器裡。
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# package.xml 宣告的相依直接用 apt 裝，省掉 build 階段跑 rosdep update。
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-cv-bridge \
        ros-humble-message-filters \
        ros-humble-realsense2-camera \
        python3-opencv \
        python3-numpy \
        python3-pip \
        python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# CLI 的 --realsense 模式（不經 ROS 直連相機）才用得到；節點本身不需要。
RUN pip3 install --no-cache-dir pyrealsense2

# 用和主機相同的 UID 跑，bind mount 內產生的檔案（output/ 等）才不會變 root 擁有。
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ros && useradd -m -u ${UID} -g ${GID} -s /bin/bash ros \
    && usermod -aG video,plugdev ros
RUN mkdir /ws && chown ros:ros /ws
USER ros
WORKDIR /ws
# 只複製 package 進來 build；執行時再把整個 repo bind mount 到同一路徑，
# --symlink-install 之後改 python 檔不必重 build。
COPY --chown=ros:ros ros2 /ws/src/TDK-straw/ros2
RUN bash -c ". /opt/ros/humble/setup.bash \
    && colcon build --symlink-install"

# 進容器就先 source 好 ROS 與 workspace。
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
