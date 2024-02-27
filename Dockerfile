FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update -y
RUN apt-get -y upgrade &&\
    apt-get -y install build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \ 
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    wget \
    liblzma-dev \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get purge -y imagemagick imagemagick-6-common
    
RUN cd /usr/src \
    && wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz \
    && tar -zxvf Python-3.11.4.tgz \
    && cd Python-3.11.4 \
    && ./configure --enable-optimizations\
    && make -j4\
    && make install

RUN apt-get -y install  cmake \
    g++ \
    unzip \
    nano \
    git \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /home/akshay/Downloads
RUN git clone https://github.com/opencv/opencv.git\
    && cd opencv && mkdir build && cd build\
    && cmake  .. \
    && make -j8 \
    && make install

RUN apt-get -y upgrade && apt-get install -y --fix-missing

RUN pip3 install --upgrade pip
RUN pip3 install torch==2.1.0\
                 torchvision==0.16.0\
                 torchaudio==2.1.0\
                 numpy \
                 scipy \
                 tqdm \
                 matplotlib \
                 ninja \
                 h5py \
                 opencv-python\
                 torchinfo\
                 torch-tb-profiler

COPY src /home/akshay/FCALIBNet/src
WORKDIR /home/akshay/FCALIBNet

RUN pip3 install src/external/UNet3D