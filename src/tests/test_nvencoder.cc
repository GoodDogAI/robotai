#include <catch2/catch_test_macros.hpp>

#include <thread>
#include <chrono>

#include "config.h"
#include "nvencoder.h"
#include "cereal/visionipc/visionbuf.h"


TEST_CASE( "Encoder renders a single frame", "[encoder]" ) {
    NVEncoder encoder { ENCODER_DEV, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT, ENCODER_BITRATE, CAMERA_FPS };
    VisionBuf buf;

    VisionIpcBufExtra extra {
                        1,
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

    // while(1) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // }
}