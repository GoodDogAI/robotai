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
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionbuf.h"


class NVEncoder{
    public:
        NVEncoder(std::string encoderdev, int32_t in_width, int32_t in_height, int32_t out_width, int32_t out_height, int32_t maxqp, int32_t maxbitrate, int32_t fps);
        ~NVEncoder();

        NVEncoder(const NVEncoder&) = delete;
        NVEncoder& operator=(const NVEncoder&) = delete;

        struct NVResult {
            uint8_t *data;
            size_t len;
            int32_t flags;
            VisionIpcBufExtra extra;

            NVResult(NVEncoder &e, uint8_t *d, size_t l, int32_t f, uint32_t i, const VisionIpcBufExtra &ex) : enc(e), data(d), len(l), flags(f), index(i), extra(ex) {}
            // Requeues the resulting buffer to V4L2 once the user is done with it
            ~NVResult();

            private:
                NVEncoder &enc;
                const uint32_t index;
        };

        std::future<std::unique_ptr<NVResult>> encode_frame(VisionBuf* buf, const VisionIpcBufExtra &extra);

        const int32_t in_width, in_height, out_width, out_height, maxqp, maxbitrate, fps;

    private: 
        void do_dequeue_capture();
        std::thread dequeue_capture_thread;

        void do_dequeue_output();
        std::thread dequeue_output_thread;

        std::vector<NVVisionBuf> buf_out;
        std::vector<NVVisionBuf> buf_in;
        
        int fd;
        int frame_write_index, frame_read_index;
    
        std::map<int, std::promise<std::unique_ptr<NVResult>>> encoder_promises;
        std::map<int, VisionIpcBufExtra> encoder_vision_buf_extras;
};

