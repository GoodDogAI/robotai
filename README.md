This is the main brains behind the robotai system.

# Setup
Make sure that you have a python environment setup. If you are on a regular PC, I like to install miniconda, and setup a "robotai" environment for it.

`conda create -n robotai python=3.9`

But if you are on a Jetson Xavier, I prefer to just use the native system python.

Be sure that you checkout all the git submodules

`git submodule update --init`

Then, install the basic system dependencies, this will work on Ubuntu generally
This command will also install the python environment dependencies via a requirements.txt file

`./setup_env.sh`

## Jetson Xavier Setup

### Provision all the peripherals
https://www.gooddog.ai/bumble/xavier-provisioning

### Compiling librealsense

You'll need to make sure to get the latest librealsense. I build from sources,
which won't work fully without the custom kernel modules, but it's good enough for now.

```
git clone https://github.com/IntelRealSense/librealsense/
git checkout v2.51.1 
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=false
make -j4
sudo make install
```

## X86/64 PC Setup

### TensorRT
You will need TensorRT 8.4, which is only available from NVIDIA.

Once you get it, you'll want to follow the installation instructions here (.tar): 
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar

In summary, you'll need to install the tensorrt*.whl file into your python environment. 

And you'll need to adjust your ld conf so that the shared libraries can be found.


```
cd ~/Downloads
tar -xvzf TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
export TRT_LIBPATH=`pwd`/TensorRT-8.4.2.4

pip install TensorRT-8.4.2.4/python/tensorrt-8.4.2.4-cp39-none-linux_x86_64.whl

sudo nano /etc/ld.so.conf.d/tensorrt.conf
(Then paste in /home/jake/TensorRT-8.4.2.4/lib)
```

# Building
Make a new build Directory
`mkdir build`
`cd build`

Configure cmake
`cmake ..`

Build the code
`cmake --build .`

Install the python libraries into their proper places
`cmake --install . --prefix /home/jake/robotai`


