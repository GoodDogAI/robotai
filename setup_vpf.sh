# This script compiles the nvidia python video sdk, using the instructions
# https://github.com/NVIDIA/VideoProcessingFramework/wiki/Building-from-source

export PATH_TO_SDK=~/Video_Codec_SDK_11.1.5

# On ubuntu 22.04 and higher, it should be possible to use system FFMPEG, but for now, you will have to use
# a custom build one, this is because of the missing libavcodec/bsf.h header file in older versions
# So, you can just delete this export if you are on a higher ubuntu
export PATH_TO_FFMPEG=~/ffmpeg/build_x64_release_shared

# Export path to CUDA compiler (you may need this sometimes if you install drivers from Nvidia site):
export CUDACXX=/usr/local/cuda/bin/nvcc

cd vpf

export INSTALL_PREFIX=$(pwd)/install
mkdir -p install
mkdir -p build

cd build

# If you want to generate Pytorch extension, set up corresponding CMake value GENERATE_PYTORCH_EXTENSION\

cmake .. \
  -DFFMPEG_DIR:PATH="$PATH_TO_FFMPEG" \
  -DVIDEO_CODEC_SDK_DIR:PATH="$PATH_TO_SDK" \
  -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
  -DGENERATE_PYTORCH_EXTENSION:BOOL="1" \
  -DCMAKE_INSTALL_PREFIX:PATH="$INSTALL_PREFIX"

make && make install

# Copy the actual so files into the main python path
cp $INSTALL_PREFIX/bin/*.so ../../src