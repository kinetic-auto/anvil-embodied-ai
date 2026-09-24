# Training image — conversion, training, dataset eval, and dataset viz.
#
# The inference image ships lerobot + torch but not the Anvil packages, and
# its torchcodec cannot load without FFmpeg shared libraries, which breaks
# video-backed dataset decoding at data-load time.
#
# Build (the base image must already exist locally):
#   docker build -f docker/training/Dockerfile -t anvil-robot-train .

ARG BASE_IMAGE=anvil-robot:latest
FROM ${BASE_IMAGE}

# FFmpeg shared libraries — torchcodec loads libavcodec/libavformat at import
# time and raises OSError without them.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Deps the inference image does not ship. Everything else (lerobot, numpy,
# pillow, opencv, pyyaml, huggingface-hub, matplotlib) is already there.
# --no-deps on the Anvil packages below avoids pulling a second torch.
RUN pip3 install --break-system-packages --no-cache-dir \
    "mcap~=1.3" \
    "mcap-ros2-support~=0.5.7" \
    "rich>=13.0" \
    "rerun-sdk>=0.26.2" \
    "wandb~=0.24"

COPY packages/anvil_shared /workspace/packages/anvil_shared
COPY packages/anvil_trainer /workspace/packages/anvil_trainer
COPY packages/anvil_eval /workspace/packages/anvil_eval
COPY packages/mcap_converter /workspace/packages/mcap_converter
RUN pip3 install --break-system-packages --no-cache-dir --no-deps \
    /workspace/packages/anvil_shared \
    /workspace/packages/anvil_trainer \
    /workspace/packages/anvil_eval \
    /workspace/packages/mcap_converter

# Training does not use the ROS2 launch entrypoint from the base image.
ENTRYPOINT []
CMD ["bash"]
