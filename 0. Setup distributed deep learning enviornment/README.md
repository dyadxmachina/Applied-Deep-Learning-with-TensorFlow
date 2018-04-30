## Setting up your Distributed Deep Learning Environment for Tensorflow 1.8+

This guide will show you how to setup your computer for distributed deep learning using Nvidia CUDA 9, cuDNN 7, and TensorFlow (gpu) 1.7. It will also include TensorRT support from Nvidia. I assume you are running on Ubuntu 16.04 or greater but it should work on older versions of Ubuntu. I also assume you have at least 1 Nvidia compatible GPU on your device. 

- [Part 1 - Nvidia Install][1]
- [Part 2 - Anaconda and Env][2]

> \*This was successfully ran on a 15’’ Alienware laptop with a single NVIDIA GeForce GTX 1070 OC with 8GB GDDR5.

#### Install the latest Nvidia drivers
##### Nvidia Display Drivers
[ Getting Started Guide][3]
##### Setup correct repo
	sudo apt-get purge nvidia\*
	sudo add-apt-repository ppa:graphics-drivers
	sudo apt-get update
##### Install latest driver
	sudo apt-get install nvidia-390
	lsmod | grep nvidia
##### Verify and hold stable version
	lsmod | grep nouveau
	sudo apt-mark hold nvidia-390
##### Most recent driver:[ https://launchpad.net/\~graphics-drivers/+archive/ubuntu/ppa][4]
> ##### Restart  after you install Nvidia drivers
`sudo shutdown -r`
#### Install Cuda Drivers
##### Then start the cuda drivers and skip the display drivers install (use .sh)

	export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
	
	export LD\_LIBRARY\_PATH=/usr/local/cuda-9.0/lib64

##### Verify cuda drivers
###### [Cuda Installation Guide][5]

#### Install cuDNN drivers
##### [Sign up for nvidia cudnn here][6]
![][image-1]

	cd folder/extracted/contents
	sudo cp -P include/cudnn.h /usr/include
	sudo cp -P lib64/libcudnn\* /usr/lib/x86\_64-linux-gnu/
	sudo chmod a+r /usr/lib/x86\_64-linux-gnu/libcudnn\*
	
	sudo cp -P include/cudnn.h /usr/local/cuda-9.0/include/
	sudo cp -P lib64/libcudnn\* /usr/local/cuda-9.0/lib64/
	sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn\*

#### Install command line tools and add to library path
	sudo apt-get install cuda-command-line-tools
	export LD\_LIBRARY\_PATH=${LD\_LIBRARY\_PATH:+${LD\_LIBRARY\_PATH}:}/usr/local/cuda/extras/CUPTI/lib64


#### Install support libraries and update
	sudo apt-get install libcupti-dev
	sudo apt-get update

###### If update has an issue due the following:
	sudo rm /etc/apt/apt.conf.d/50appstream
###### Source: [https://askubuntu.com/questions/771547/ubuntu-16-04-apt-get-update-doesnt-work-with-local-repository][7]

#### Installing Anaconda
> [Download Anaconda 5.1 Python 2.7 Here ][8]
![][image-2]
#### Create Custom Env using Conda
	conda create -n tensorflow python=2.7
	# activate your env
	source activate tensorflow
#### Install Requirements
Requirements include TensorFlow GPU version, Keras, google-cloud, and supporting libs.

	pip install --upgrade pip
	pip install -r requirements.txt

[1]:	https://github.com/christianramsey/Tensorflow-for-Distributed-Deep-Learning/blob/master/0.%20installation%20and%20setup/1.%20Nvidia%20Setup.md "Install Nvidia Drivers and Libs"
[2]:	https://github.com/christianramsey/Tensorflow-for-Distributed-Deep-Learning/blob/master/0.%20installation%20and%20setup/2.%20Anaconda%20Setup.md
[3]:	http://www.linuxandubuntu.com/home/how-to-install-latest-nvidia-drivers-in-linux "How to Install Latest Nvidia Drivers in Linux"
[4]:	%20https://launchpad.net/%5C~graphics-drivers/+archive/ubuntu/ppa "Most recent nvidia graphics driver"
[5]:	http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions "Cuda Installation Guide"
[6]:	https://developer.nvidia.com/rdp/cudnn-download "Sign up for nvidia cudnn here"
[7]:	https://askubuntu.com/questions/771547/ubuntu-16-04-apt-get-update-doesnt-work-with-local-repository "If apt get update doesnt work with local repo"
[8]:	http://www.linuxandubuntu.com/home/how-to-install-latest-nvidia-drivers-in-linux "How to Install Latest Nvidia Drivers in Linux"

[image-1]:	https://lh3.googleusercontent.com/3Bg9brSrCyMC8ta6sOfKZF733nGIKL6kJio9BI6bFBGBvPoLsp4uly1VzIL2umzsoMvNHg=s170 "Sign up for Nvidia cuDNN drivers"
[image-2]:	https://lh3.googleusercontent.com/3Bg9brSrCyMC8ta6sOfKZF733nGIKL6kJio9BI6bFBGBvPoLsp4uly1VzIL2umzsoMvNHg=s170 "Sign up for Nvidia cuDNN drivers"