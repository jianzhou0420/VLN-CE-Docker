# Dockerfile for VLN-CE Standalone Policy Server
#
# This is a STANDALONE container that serves CMA, Seq2Seq, or Waypoint policies
# via WebSocket for VLN-CE evaluation.
#
# The container:
# 1. Loads a trained navigation policy (CMA, Seq2Seq, or Waypoint)
# 2. Starts a WebSocket server on the specified port
# 3. Receives observations (RGB, depth, instruction) from eval client
# 4. Returns navigation actions
#
# Prerequisites:
#   - Model checkpoints in data/checkpoints/
#   - Vocabulary files in data/datasets/
#   - DDPPO depth encoder in data/ddppo-models/
#
# Build (from this directory):
#   cd docker/policy_server
#   docker build -t vlnce-policy-server .
#
# Run (CMA example):
#   docker run --rm -it --gpus all --network=host \
#     -v $(pwd)/data/checkpoints:/app/data/checkpoints:ro \
#     -v $(pwd)/data/datasets:/app/data/datasets:ro \
#     -v $(pwd)/data/ddppo-models:/app/data/ddppo-models:ro \
#     vlnce-policy-server cma --port 8765

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (includes mamba for faster dependency resolution)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-24.3.0-0-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app

# Copy project files needed for installation (all paths relative to docker/policy_server/)
COPY environment.yaml /app/environment.yaml
COPY pyproject.toml /app/pyproject.toml
COPY third_party/VLN-CE/habitat_extensions /app/habitat_extensions
COPY third_party/VLN-CE/vlnce_baselines /app/vlnce_baselines
COPY vlnce_server /app/vlnce_server
COPY third_party/habitat-lab /app/third_party/habitat-lab

# Create environment from environment.yaml (gym excluded, installed separately)
RUN mamba env create -f environment.yaml && \
    mamba clean -afy

# Install gym==0.21 separately
# Downgrade wheel first (newer wheel rejects gym's invalid "opencv-python>=3." spec)
RUN mamba run -n vlnce pip install "wheel<0.40" && \
    mamba run -n vlnce pip install "gym==0.21"

# Remove conflicting OpenGL packages (use system NVIDIA drivers instead)
# This is critical to avoid rendering bugs with NVIDIA GPUs
RUN mamba remove -n vlnce libgl libglvnd libglx libegl --force -y 2>/dev/null || true

# Set up shell to use mamba
SHELL ["mamba", "run", "-n", "vlnce", "/bin/bash", "-c"]

# Set LD_LIBRARY_PATH for the conda environment
ENV LD_LIBRARY_PATH=/opt/conda/envs/vlnce/lib:$LD_LIBRARY_PATH

# Set environment variables for headless rendering
ENV HABITAT_SIM_LOG=quiet
ENV MAGNUM_LOG=quiet
ENV GLOG_minloglevel=2

# Create data directories for volume mounts
RUN mkdir -p /app/data/checkpoints /app/data/datasets /app/data/ddppo-models

# Copy and make entrypoint executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default entrypoint runs the policy server launcher
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
