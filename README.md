This is the main brains behind the robotai system.

# Setup
Make sure that you have a python environment setup,  I like to install miniconda, and
setup a "robotai" environment for it.
`conda create -n robotai python=3.9`

Then, install the basic system dependencies, this will work on Ubuntu generally
This command will also install the python environment dependencies via a requirements.txt file
`./setup_env.sh`

# TensorRT
You will need TensorRT 8.4, which is only available from NVIDIA.

```
cd ~/Downloads
tar -xvzf TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
export TRT_LIBPATH=`pwd`/TensorRT-8.4.2.4

pip install TensorRT-8.4.2.4/python/tensorrt-8.4.2.4-cp39-none-linux_x86_64.whl
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


