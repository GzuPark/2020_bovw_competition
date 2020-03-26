FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV LANG C.UTF-8

ARG PYTHON_VERSION
ARG CONDA_ENV_NAME

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    sudo \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    wget \
    unzip
RUN rm -rf /var/lib/apt/lists/*
# ssh install and setting

# Install conda
RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh

RUN conda update -y conda && \
    conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION

ENV PATH /usr/local/envs/$CONDA_ENV_NAME/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

# Install pip packages
COPY requirements.txt /tmp/requirements.txt
RUN source activate ${CONDA_ENV_NAME} && pip install --no-cache-dir -r /tmp/requirements.txt

# Enable jupyter lab
RUN source activate ${CONDA_ENV_NAME} && \
    conda install -c conda-forge jupyterlab && \
    jupyter serverextension enable --py jupyterlab --sys-prefix
