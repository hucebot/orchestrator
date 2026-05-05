#!/bin/bash
set -e

# Setup ROS 2 environment
source /opt/ros/$ROS_DISTRO/setup.bash

# Source local workspace if built
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi

exec "$@"