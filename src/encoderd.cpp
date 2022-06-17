// encoderd takes in visionframes, and encodes them to H265 format using the hardware nvidia encoder
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cassert>

#include "util.h"

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"

ExitHandler do_exit;

int main(int argc, char *argv[])
{
    VisionIpcClient vipc_client = VisionIpcClient("camerad", VISION_STREAM_HEAD_COLOR, false);

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

    while (!do_exit) {
        VisionIpcBufExtra extra;
        VisionBuf* buf = vipc_client.recv(&extra);
        if (buf == nullptr)
            continue;

        std::cout << "Received frame " << extra.frame_id << std::endl;
    }

    return EXIT_SUCCESS;
}