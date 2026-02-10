FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install -y python3
RUN apt-get install -y curl
RUN apt-get install -y git
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"
RUN apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN mkdir /root/roma
WORKDIR /root/roma
COPY . /root/roma/
RUN uv sync
