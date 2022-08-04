#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <thread>
#include <chrono>

#include "config.h"
#include "nvencoder.h"
#include "cereal/visionipc/visionbuf.h"


TEST_CASE( "Encoder renders a single frame", "[encoder]" ) {
    NVEncoder encoder { ENCODER_DEV, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT, ENCODER_BITRATE, CAMERA_FPS };
    VisionBuf buf;

    VisionIpcBufExtra extra {
                        1, // frame_id
                        0,
                        0,
        };

    auto size = CAMERA_WIDTH * CAMERA_HEIGHT * 3 / 2;
    auto stride = CAMERA_WIDTH;
    auto uv_offset = CAMERA_WIDTH * CAMERA_HEIGHT;
    buf.allocate(size);
    buf.init_yuv(CAMERA_WIDTH, CAMERA_HEIGHT, stride, uv_offset);

    auto future = encoder.encode_frame(&buf, &extra);
    auto result = future.get();

    INFO("Got result from encoder len " << result->len);
    REQUIRE(result->len > 0);
    REQUIRE(result->flags & V4L2_BUF_FLAG_KEYFRAME);
}

TEST_CASE( "Encoder sends keyframes once per second at least", "[encoder]" ) {
    int test_fps = 5;

    NVEncoder encoder { ENCODER_DEV, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT, ENCODER_BITRATE, test_fps };
    VisionBuf buf;

    auto size = CAMERA_WIDTH * CAMERA_HEIGHT * 3 / 2;
    auto stride = CAMERA_WIDTH;
    auto uv_offset = CAMERA_WIDTH * CAMERA_HEIGHT;
    buf.allocate(size);
    buf.init_yuv(CAMERA_WIDTH, CAMERA_HEIGHT, stride, uv_offset);

    for(uint32_t i = 0; i < test_fps * 3; i++){
        VisionIpcBufExtra extra {
                        i + 1, // frame_id
                        0,
                        0,
        };

        auto future = encoder.encode_frame(&buf, &extra);
        auto result = future.get();
        
        INFO("frame " << i << " " << result->flags);

        if (i % test_fps == 0) {
            CHECK(result->flags & V4L2_BUF_FLAG_KEYFRAME);
        }  
    }
}