#pragma once

#include <string>

#include "visionbuf.h"


class NVEncoder{
    public:
        NVEncoder(std::string encoderdev, int in_width, int in_height, int out_width, int out_height, int bitrate, int fps);
        ~NVEncoder();


        int fd;
        int in_width, in_height, out_width, out_height, bitrate, fps;

    private: 
        std::vector<NVVisionBuf> buf_out;
        std::vector<NVVisionBuf> buf_in;
};

