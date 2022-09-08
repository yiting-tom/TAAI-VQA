FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    make \
    gcc \
    g++ \
    ninja-build \
    wget \
    git \
    python3.8-venv \
    python3.8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
ENV CUDA_HOME="/usr/local/cuda-11.1"