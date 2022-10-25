sudo apt-get install -y --no-install-recommends \
    clang \
    capnproto \
    libcapnp-dev \
    libzmq3-dev \
    libudev-dev \
    cython3 \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-pkgconfig \
    libbluetooth-dev \
    libavcodec-dev \
    libavutil-dev \
    libavformat-dev \
    libswresample-dev \
    libavfilter-dev \
    libswscale-dev

pip3 install -r requirements.txt

# For pytorch, with cuda 11.6 support, but only on x86 hosts,
# on Xavier platforms, you'll have to install pytorch manually
if [[ $(dpkg --print-architecture) = amd64 ]]; then
  pip3 install --extra-index-url https://download.pytorch.org/whl/cu116 torch torchvision torchaudio torchdata

  conda install -c conda-forge opencv
fi
