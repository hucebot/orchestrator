FROM ros:humble-ros-base

# Install system and ROS dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-cv-bridge \
    ros-humble-rmw-cyclonedds-cpp \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy ONLY the package.xml to preserve the Docker cache
COPY src/orchestrator/package.xml src/orchestrator/

# Resolve all dependencies
RUN apt-get update && \
    rosdep update && \
    rosdep install -y -r -q --from-paths src --ignore-src --rosdistro humble && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the  source code
COPY src/ src/

# Setup Entrypoint
COPY .ci/entrypoint.sh /
RUN chmod +x /entrypoint.sh

# Ensure the workspace is sourced automatically in bash
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/install/setup.bash" >> ~/.bashrc

# Compile everything
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]