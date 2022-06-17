#pragma once

#include <string>
#include <fstream>

#include "visionbuf.h"


class NVEncoder{
    public:
        NVEncoder(std::string encoderdev, int in_width, int in_height, int out_width, int out_height, int bitrate, int fps);
        ~NVEncoder();

        int encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra);

        int fd;
        int in_width, in_height, out_width, out_height, bitrate, fps;

    private: 
        std::vector<NVVisionBuf> buf_out;
        std::vector<NVVisionBuf> buf_in;

        // Temp output stream
        std::ofstream outfile;
};

