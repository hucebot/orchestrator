FROM ros:humble-ros-base

# Install system and ROS dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-cv-bridge \
    ros-humble-rmw-cyclonedds-cpp \
    git \
    && rm -rf /var/lib/apt/lists/*

# Setup the workspace and copy your custom node
WORKDIR /workspace
COPY src/ /workspace/src/
COPY . /workspace/src/orchestrator/

# Resolve all dependencies and BUILD the workspace
WORKDIR /workspace
RUN apt-get update && \
    rosdep update && \
    rosdep install -y -r -q --from-paths src --ignore-src --rosdistro humble && \
    rm -rf /var/lib/apt/lists/*

# Compile everything (your node + TIAGo descriptions) so ROS indexes the mesh paths
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# 6. Setup Entrypoint
COPY .ci/entrypoint.sh /
RUN chmod +x /entrypoint.sh

# Ensure the workspace is sourced automatically in bash
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/install/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]