// encoderd takes in visionframes, and encodes them to H265 format using the hardware nvidia encoder
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <linux/videodev2.h>
#include <libv4l2.h>
#include <fcntl.h>

#include "v4l2_nv_extensions.h"
#include "util.h"

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"

#include "config.h"
#include "nvencoder.h"

#define BUF_IN_COUNT 6
#define BUF_OUT_COUNT 6

ExitHandler do_exit;


int main(int argc, char *argv[])
{
    VisionIpcClient vipc_client = VisionIpcClient("camerad", VISION_STREAM_HEAD_COLOR, false);
    NVEncoder encoder(ENCODER_DEV, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT, 2000000, CAMERA_FPS);

    // Connect to the visionipc server
    while (!do_exit) {
        if (!vipc_client.connect(false)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        else {
            std::cout << "Connected to visionipc" << std::endl;
            break;
        }
    }

    // Perform the encoding
    while (!do_exit) {
        VisionIpcBufExtra extra;
        VisionBuf* buf = vipc_client.recv(&extra);
        if (buf == nullptr)
            continue;

        std::cout << "Received frame " << extra.frame_id << std::endl;

        encoder.encode_frame(buf, &extra);
    }

    return EXIT_SUCCESS;
}