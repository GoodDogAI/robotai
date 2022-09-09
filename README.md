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

nano ~/.bashrc
export LD_LIBRARY_PATH=/home/jake/TensorRT-8.4.2.4/lib:$LD_LIBRARY_PATH
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



## Notes on Colorspaces

This project involves a lot of different image sources, along with video transcoding, and color space conversion.

- IntelRealsense, outputs YUYV format as 1280W x 720H resolution and 15FPS
    - TODO: Figure out what the range is on the Y and UV values here, 0-255 or 40-216?
- camerad takes that, and seperates out the 1280Wx720H Y plane, and then creates a UV plane which has 1280Wx360H
  This is known as NV12M format
- encoderd takes the NV12M and passes it directly into the NVIDIA encoder which gets saved into H265 format
    - TODO: This is encoded with a default colorspace, does that need to be changed?
- braind takes the NV12M and passes it to the vision onnx models, which do an internal conversion step
    - TODO: Once you know what the range is on the camerad YUV stuff, you can make sure that you have the proper color converter in ONNX
- The host training process gets the H265 packets, and decodes them using NVDEC, into NV12 format
- Then, it takes NV12 into YUV420, and then RGB, but it uses color_space=nvc.ColorSpace.BT_601 and color_range=nvc.ColorRange.MPEG
    - TODO: What actual color_space and color_range to use here?
- Those RGB frames can go directly into ONNX models for vision intermediates, or vision rewards
