sudo apt-get install -y --no-install-recommends \
    clang \
    capnproto \
    libcapnp-dev \
    libzmq3-dev \
    cython3 \
    python3-dev \
    python3-numpy \
    python3-pkgconfig \
    libavcodec-dev \
    libavutil-dev \
    libavformat-dev \
    libswresample-dev \
    libavfilter-dev \
    libswscale-dev

pip3 install -r requirements.txt

# For pytorch, with cuda 11.6 support
pip3 install --extra-index-url https://download.pytorch.org/whl/cu116 torch torchvision torchaudio