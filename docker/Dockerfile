FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

LABEL maintainer="nishimura"

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        libgtk2.0-dev \
	libgl1-mesa-glx \
	git \
	curl \ 
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#     python get-pip.py && \
#     rm get-pip.py

RUN pip --no-cache-dir install \
        opencv-python \
        Pillow \
        matplotlib \
        pandas \
        scipy \
        sklearn \
	pulp \
	scikit-image \
	tqdm \
    	visdom 

ARG UID
ARG USER
ARG PASSWORD
RUN groupadd -g 1000 ${USER}_group
RUN useradd -m --uid=${UID} --gid=${USER}_group --groups=sudo ${USER}
RUN echo ${USER}:${PASSWORD} | chpasswd
RUN echo 'root:root' | chpasswd
USER ${USER}
