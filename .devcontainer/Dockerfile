# Download the official PyTorch image with CUDA 11.8 and cuDNN 8 from Docker Hub
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set to interactive mode so when installing packages with apt-get, the process doesn’t prompt for any user input 
ARG DEBIAN_FRONTEND=noninteractive

#Set timezone
ENV TZ=Europe/Amsterdam

# Install locales and set the locale to en_US.UTF-8
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales && \
    locale-gen en_US.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the locale environment variables
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Install necessary packages 
RUN apt-get update && \
    apt-get install -y \
    git \
    tmux \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set default command to bash opens a bash terminal automatically 
CMD ["/bin/bash"]