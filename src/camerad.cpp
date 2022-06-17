// camerad talks to the Intel RealSense camera and runs a visionipc server to send the frames to other services

#include <chrono>
#include <iostream>
#include <algorithm>
#include <cassert>

#include <librealsense2/rs.hpp>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_server.h"

#include "config.h"

#define CAMERA_BUFFER_COUNT 30

int main(int argc, char *argv[])
{
    VisionIpcServer vipc_server("camerad");
    vipc_server.create_buffers(VISION_STREAM_HEAD_COLOR, CAMERA_BUFFER_COUNT, false, CAMERA_WIDTH, CAMERA_HEIGHT);
    vipc_server.start_listener();

    rs2::context ctx;
    rs2::device_list devices_list = ctx.query_devices();
    size_t device_count = devices_list.size();
    if (!device_count)
    {
        std::cout << "No device detected. Is it plugged in?\n";
        return EXIT_SUCCESS;
    }

    std::cout << "Found " << device_count << " devices\n";

    rs2::device device = devices_list.front();

    std::cout << "Device Name: " << device.get_info(RS2_CAMERA_INFO_NAME) << std::endl;

    auto depth_sens = device.first<rs2::depth_sensor>();
    auto color_sens = device.first<rs2::color_sensor>();

    // Find a matching profile
    auto profiles = color_sens.get_stream_profiles();
    auto stream_profile = *std::find_if(profiles.begin(), profiles.end(), [](rs2::stream_profile &profile)
                                        {
        rs2::video_stream_profile sp = profile.as<rs2::video_stream_profile>();
            return sp.width() == CAMERA_WIDTH && sp.height() == CAMERA_HEIGHT && sp.format() == RS2_FORMAT_YUYV && sp.fps() == CAMERA_FPS; });

    if (!stream_profile)
    {
        std::cerr << "No matching camera configuration found\n";
        return EXIT_FAILURE;
    }

    color_sens.open(stream_profile);

    rs2::frame_queue queue(1);
    color_sens.start(queue);

    uint32_t frame_id = 0;
    auto start = std::chrono::steady_clock::now();

    while (1)
    {
        rs2::video_frame color_frame = queue.wait_for_frame();

        if (color_frame.get_frame_number() != frame_id + 1)
        {
            std::cerr << "Frame number mismatch" << std::endl;
            std::cerr << "Got " << color_frame.get_frame_number() << " expected " << frame_id + 1 << std::endl;
            break;
        }
        else
        {
            frame_id = color_frame.get_frame_number();
        }

        auto cur_yuv_buf = vipc_server.get_buffer(VISION_STREAM_HEAD_COLOR);

        // TODO To get proper frame metadata, it will be necessary to patch the kernel and bypass the UVC stuff
        // For now, it will be best to just go with UVC, and maybe by the time I finish coding, they will have
        // discontinued realsense anyways...
        //auto start_of_frame = color_frame.get_frame_metadata(RS2_FRAME_METADATA_FRAME_TIMESTAMP);
        //std::cout << "Frame " << frame_id << " at " << color_frame.get_frame_timestamp_domain() << std::endl;

        VisionIpcBufExtra extra = {
                        frame_id,
                        0,
                        0,
        };
        cur_yuv_buf->set_frame_id(frame_id);

        vipc_server.send(cur_yuv_buf, &extra);

        // uint8_t *yuyv_data = (uint8_t *)color_frame.get_data();

        // int32_t color_frame_width = color_frame.get_width();
        // int32_t color_frame_height = color_frame.get_height();
        // int32_t color_frame_stride = color_frame.get_stride_in_bytes();

        if (frame_id % 100 == 0)
        {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start) / 100;
            std::cout << "100 Frames "
                      << " took " << duration.count() << "ms" << std::endl;

            start = std::chrono::steady_clock::now();
        }
    }

    return EXIT_SUCCESS;
}