#pragma once

#include <string>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <map>
#include <thread>
#include <future>
#include "nvvisionbuf.h"




class NVEncoder{
    public:
        NVEncoder(std::string encoderdev, int in_width, int in_height, int out_width, int out_height, int bitrate, int fps);
        ~NVEncoder();

        NVEncoder(const NVEncoder&) = delete;
        NVEncoder& operator=(const NVEncoder&) = delete;

        struct NVResult {
            uint8_t *data;
            size_t len;

            NVResult(NVEncoder &e, uint8_t *d, size_t l, uint32_t i) : enc(e), data(d), len(l), index(i) {}
            // Requeues the resulting buffer to V4L2 once the user is done with it
            ~NVResult();

            private:
                NVEncoder &enc;
                const uint32_t index;
        };

        std::future<std::unique_ptr<NVResult>> encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra);

        int fd;
        int in_width, in_height, out_width, out_height, bitrate, fps;
        int frame_write_index, frame_read_index;

    private: 
        void do_dequeue_capture();
        std::thread dequeue_capture_thread;

        void do_dequeue_output();
        std::thread dequeue_output_thread;

        std::vector<NVVisionBuf> buf_out;
        std::vector<NVVisionBuf> buf_in;
    
        std::map<int, std::promise<std::unique_ptr<NVResult>>> encoder_promises;
};

