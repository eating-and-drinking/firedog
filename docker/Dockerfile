# =========================================================
# 四足机器狗 Agent 系统 Dockerfile
# 基础镜像：ROS 2 Humble + CUDA 12（可按实际环境调整）
# =========================================================

FROM ros:humble-ros-base-jammy AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖（分层缓存）
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# 应用代码
COPY . .

# 预下载 Silero VAD（离线场景）
RUN python3 -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=False)" || true

EXPOSE 8000

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["python3", "scripts/launch.py", "--mode", "full"]
