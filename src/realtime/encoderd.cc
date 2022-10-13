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

#include <argparse/argparse.hpp>
#include <fmt/core.h>

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
    argparse::ArgumentParser args("braind");

    args.add_argument("camera_stream")
        .default_value(VISION_STREAM_HEAD_COLOR)
        .help("Which camera stream to encode")
        .action([](const std::string& value) {
            static const std::map<std::string, VisionStreamType> choices = { {"head_color", VISION_STREAM_HEAD_COLOR},
                                                                             {"head_depth", VISION_STREAM_HEAD_DEPTH} };

            auto it = choices.find(value);                                                                             
            if (it == choices.end()) {                                                                                 
              throw std::runtime_error("Invalid camera stream");                                                       
            }
      
            return it->second;
        });

    args.add_argument("--maxqp")
        .default_value(-1)
        .scan<'d', int32_t>()
        .help("Set encoder to enforce Maximum QP value (meaning a minimum quality factor) for each frame.");

    args.add_argument("--maxbitrate")
        .default_value(-1)
        .scan<'d', int32_t>()
        .help("Set encoder to enforce Maximum Bitrate value (in bits per second) for each frame.");

    try
    {
        args.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        fmt::print(stderr, "Error parsing arguments {}\n", err.what());
        return EXIT_FAILURE;
    }

    VisionIpcClient vipc_client { "camerad", args.get<VisionStreamType>("camera_stream"), false };
    NVEncoder encoder { ENCODER_DEV, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT, args.get<int32_t>("maxqp"), args.get<int32_t>("maxbitrate"), CAMERA_FPS };
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

    // Receive all stale frames from visionipc
    while (true) {
        VisionIpcBufExtra extra;

        // half a frame timeout, so if there are no pending frames, we can exit
        VisionBuf *buf = vipc_client.recv(&extra, (1000 / CAMERA_FPS) / 2);
        if (buf == nullptr) {
            break;
        }
    }

    // Perform the encoding
    while (!do_exit) {
        VisionIpcBufExtra extra;
        VisionBuf* buf = vipc_client.recv(&extra);
        if (buf == nullptr)
            continue;

        // Add the future to be retrieved later
        auto future = encoder.encode_frame(buf, extra);
        encoder_futures.push_back(std::move(future));

        // Process and dump out any finished futures 
        while (!encoder_futures.empty() && encoder_futures.front().wait_for(0s) == std::future_status::ready) {
            auto result = encoder_futures.front().get();

            MessageBuilder msg;
            auto event = msg.initEvent(true);
            auto edat = event.initHeadEncodeData();
            auto edata = edat.initIdx(); 
            edata.setFrameId(result->extra.frame_id);
            edata.setTimestampSof(result->extra.timestamp_sof);
            edata.setEncodeId(num_frames);
            edata.setType(cereal::EncodeIndex::Type::FULL_H_E_V_C);
            edata.setFlags(result->flags);

            edat.setData(kj::heapArray<capnp::byte>(result->data, result->len));

            auto words = capnp::messageToFlatArray(msg);
            auto bytes = words.asBytes();
            pm.send(service_name, bytes.begin(), bytes.size());
          
            encoder_futures.pop_front();
        }

        num_frames++;
    }

    return EXIT_SUCCESS;
}