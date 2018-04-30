
# NVIDIA DISPLAY DRIVERS
# http://www.linuxandubuntu.com/home/how-to-install-latest-nvidia-drivers-in-linux

# Install the latest nvidia drivers

sudo apt-get purge nvidia*
sudo add-apt-repository ppa:graphics-drivers
And update
sudo apt-get update

sudo apt-get install nvidia-390
lsmod | grep nvidia
lsmod | grep nouveau

sudo apt-mark hold nvidia-390

# Most recent driver: https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa
# Restart  after you install nvidia drivers


# INSTALL CUDA DRIVERS
# Then start the cuda drivers and skip the display drivers install (use .sh)

export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64\
                     	${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Verify cuda drivers
# http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions

# deviceQuery

# INSTALL CUDNN DRIVERS
# Sign up for nvidia cudnn
# https://developer.nvidia.com/rdp/cudnn-download

cd folder/extracted/contents
sudo cp -P include/cudnn.h /usr/include
sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*

sudo cp -P include/cudnn.h /usr/local/cuda-9.0/include/
sudo cp -P lib64/libcudnn* /usr/local/cuda-9.0/lib64/
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*



# INSTALL SUPPORT LIB AND UPDATE
sudo apt-get install libcupti-dev

sudo apt-get update

# If update has an issue due the following:
sudo rm /etc/apt/apt.conf.d/50appstream
# Source: https://askubuntu.com/questions/771547/ubuntu-16-04-apt-get-update-doesnt-work-with-local-repository


# INSTALL DOCKER
install docker
wget -qO- https://get.docker.com/ | sh


NVIDIA-DOCKER
Why we need nvidia-docker
https://github.com/NVIDIA/nvidia-docker/wiki/Motivation

https://github.com/NVIDIA/nvidia-docker/wiki

DOESN’T WORK!
3.    Installing the keras-mxnet nvidia-docker environment
$ docker pull bentaylordata/mxnet-keras











