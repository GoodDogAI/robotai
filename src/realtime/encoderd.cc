// encoderd takes in visionframes, and encodes them to H265 format using the hardware nvidia encoder
#include <chrono>
#include <iostream>
#include <fstream>
#include <deque>
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

using namespace std::chrono_literals;

ExitHandler do_exit;

const char *service_name = "headEncodeData";

int main(int argc, char *argv[])
{
    VisionIpcClient vipc_client { "camerad", VISION_STREAM_HEAD_COLOR, false };
    NVEncoder encoder { ENCODER_DEV, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT, ENCODER_BITRATE, CAMERA_FPS };
    std::deque<std::future<std::unique_ptr<NVEncoder::NVResult>>> encoder_futures {};
    PubMaster pm{ {service_name} };
    int32_t num_frames{ 0 };

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

        if (num_frames % 50 == 0) {
            std::cout << "Received frame " << extra.frame_id << std::endl;
        }

        // Add the future to be retrieved later
        auto future = encoder.encode_frame(buf, &extra);
        encoder_futures.push_back(std::move(future));

        // Process and dump out any finished futures 
        while (!encoder_futures.empty() && encoder_futures.front().wait_for(0s) == std::future_status::ready) {
            auto result = encoder_futures.front().get();

            MessageBuilder msg;
            auto event = msg.initEvent(true);
            auto edat = event.initHeadEncodeData();
            auto edata = edat.initIdx(); 
            edata.setEncodeId(num_frames);
            edata.setFlags(result->flags);

            edat.setData(kj::heapArray<capnp::byte>(result->data, result->len));

            auto words = new kj::Array<capnp::word>(capnp::messageToFlatArray(msg));
            auto bytes = words->asBytes();
            pm.send(service_name, bytes.begin(), bytes.size());
          
            encoder_futures.pop_front();
        }

        num_frames++;
    }

    return EXIT_SUCCESS;
}