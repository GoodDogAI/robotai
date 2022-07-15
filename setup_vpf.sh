# This script compiles the nvidia python video sdk, using the instructions
# https://github.com/NVIDIA/VideoProcessingFramework/wiki/Building-from-source

export PATH_TO_SDK=~/Video_Codec_SDK_11.1.5

# Export path to CUDA compiler (you may need this sometimes if you install drivers from Nvidia site):
export CUDACXX=/usr/local/cuda/bin/nvcc

cd vpf

export INSTALL_PREFIX=$(pwd)/install
mkdir -p install
mkdir -p build

cd build

# If you want to generate Pytorch extension, set up corresponding CMake value GENERATE_PYTORCH_EXTENSION
cmake .. \
  -DFFMPEG_DIR:PATH="$PATH_TO_FFMPEG" \
  -DVIDEO_CODEC_SDK_DIR:PATH="$PATH_TO_SDK" \
  -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
  -DGENERATE_PYTORCH_EXTENSION:BOOL="1" \
  -DCMAKE_INSTALL_PREFIX:PATH="$INSTALL_PREFIX"

make && make install