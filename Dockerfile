# Get the base Ubuntu image from Docker Hub
FROM ubuntu:20.04

# Not interactive
ENV DEBIAN_FRONTEND=noninteractive 

# Copy the current folder which contains C++ source code to the Docker image
# under /usr/src
COPY . /usr/src/ao
WORKDIR /usr/src/ao

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basic packages
RUN apt-get update \
    && apt-get -y install build-essential tar curl zip unzip cmake git \
    python3 nasm pkg-config

# Setup the environment with `vcpkg`
RUN git clone https://github.com/Microsoft/vcpkg.git
RUN ./vcpkg/bootstrap-vcpkg.sh \
    && ./vcpkg/vcpkg integrate install \
    && ./vcpkg/vcpkg integrate bash \
    && echo 'export PATH=$PATH:${pwd}/vcpkg' >>~/.bashrc

# TODO Install latest version of cmake

# Build the source code
# RUN ./build.sh

CMD ["bash"]

LABEL Name=ao Version=0.1.0