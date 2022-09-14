// camerad talks to the Intel RealSense camera and runs a visionipc server to send the frames to other services

#include <chrono>
#include <iostream>
#include <algorithm>
#include <cassert>

#include <fmt/core.h>
#include <fmt/chrono.h>

#include <librealsense2/rs.hpp>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_server.h"

#include "config.h"

#define CAMERA_BUFFER_COUNT 30

int main(int argc, char *argv[])
{
    VisionIpcServer vipc_server{ "camerad" };
    vipc_server.create_buffers(VISION_STREAM_HEAD_COLOR, CAMERA_BUFFER_COUNT, false, CAMERA_WIDTH, CAMERA_HEIGHT);
    vipc_server.start_listener();

    rs2::context ctx;
    rs2::device_list devices_list{ ctx.query_devices() };
    size_t device_count{ devices_list.size() };
    if (!device_count)
    {
        fmt::print(stderr, "No device detected. Is it plugged in?\n");
        return EXIT_SUCCESS;
    }

    fmt::print("Found {} devices\n", device_count);

    rs2::device device = devices_list.front();

    fmt::print("Device Name: {}\n", device.get_info(RS2_CAMERA_INFO_NAME) );

    auto depth_sens{ device.first<rs2::depth_sensor>() };
    auto color_sens{ device.first<rs2::color_sensor>() };

    // Find a matching profile
    auto profiles{ color_sens.get_stream_profiles() };
    auto stream_profile = *std::find_if(profiles.begin(), profiles.end(), [](rs2::stream_profile &profile)
                                        {
        rs2::video_stream_profile sp = profile.as<rs2::video_stream_profile>();
            return sp.width() == CAMERA_WIDTH && sp.height() == CAMERA_HEIGHT && sp.format() == RS2_FORMAT_YUYV && sp.fps() == CAMERA_FPS; });

    if (!stream_profile)
    {
        fmt::print(stderr, "No matching camera configuration found\n");
        return EXIT_FAILURE;
    }

    color_sens.open(stream_profile);

    rs2::frame_queue queue{ 1 };
    color_sens.start(queue);

    uint32_t frame_id{ 0 };
    auto start {std::chrono::steady_clock::now()};
    rs2_metadata_type last_start_of_frame {};
    constexpr rs2_metadata_type expected_frame_time = 1'000'000 / CAMERA_FPS; // usec


    while (true)
    {
        rs2::video_frame color_frame = queue.wait_for_frame();

        if (color_frame.get_frame_number() != frame_id + 1 && frame_id != 0)
        {
            fmt::print(stderr, "Frame number mismatch\n");
            fmt::print(stderr, "Got {} expected {}\n", color_frame.get_frame_number(), frame_id + 1);
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
        rs2_metadata_type start_of_frame = color_frame.get_frame_metadata(RS2_FRAME_METADATA_FRAME_TIMESTAMP);
        //std::cout << "Frame " << frame_id << " at " << color_frame.get_frame_timestamp_domain() << " " << start_of_frame << std::endl;

        //std::cout << "Exposure " << color_frame.supports_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE) << std::endl;
        
        // Check for any weird camera jitter, and if so, we would like to terminate for now, after some initialization period
        if (frame_id > CAMERA_FPS && std::abs((start_of_frame - last_start_of_frame) - expected_frame_time) > expected_frame_time * 0.05) {
            fmt::print(stderr, "Got unexpected frame jitter of {} vs expected {} usec\n", start_of_frame - last_start_of_frame, expected_frame_time);
            throw std::runtime_error("Unexpectedly high jitter");
        }

        VisionIpcBufExtra extra {
                        frame_id,
                        static_cast<uint64_t>(start_of_frame),
                        0,
        };
        cur_yuv_buf->set_frame_id(frame_id);

        uint8_t *yuyv_data = (uint8_t *)color_frame.get_data();

        int32_t color_frame_width = color_frame.get_width();
        int32_t color_frame_height = color_frame.get_height();
        int32_t color_frame_stride = color_frame.get_stride_in_bytes();

        for(uint32_t row = 0; row < color_frame_height / 2; row++) {
            for (uint32_t col = 0; col < color_frame_width / 2; col++) {
                cur_yuv_buf->y[(row * 2) * cur_yuv_buf->stride + col * 2] = yuyv_data[(row * 2) * color_frame_stride + col * 4];
                cur_yuv_buf->y[(row * 2) * cur_yuv_buf->stride + col * 2 + 1] = yuyv_data[(row * 2) * color_frame_stride + col * 4 + 2];
                cur_yuv_buf->y[(row * 2 + 1) * cur_yuv_buf->stride + col * 2] = yuyv_data[(row * 2 + 1) * color_frame_stride + col * 4];
                cur_yuv_buf->y[(row * 2 + 1) * cur_yuv_buf->stride + col * 2 + 1] = yuyv_data[(row * 2 + 1) * color_frame_stride + col * 4 + 2];

                cur_yuv_buf->uv[row * cur_yuv_buf->stride + col * 2] = (yuyv_data[(row * 2) * color_frame_stride + col * 4 + 1] + yuyv_data[(row * 2 + 1) * color_frame_stride + col * 4 + 1]) / 2;
                cur_yuv_buf->uv[row * cur_yuv_buf->stride + col * 2 + 1] = (yuyv_data[(row * 2) * color_frame_stride + col * 4 + 3] + yuyv_data[(row * 2 + 1) * color_frame_stride + col * 4 + 3]) / 2;
            }
        }

        // Collect min/max statistics on the Y and UV data so we can understand if it's full-range or [16, 235] range
        // auto minmax_frame_y = std::minmax_element(cur_yuv_buf->y, cur_yuv_buf->y + cur_yuv_buf->width * cur_yuv_buf->height);
        // auto minmax_frame_uv = std::minmax_element(cur_yuv_buf->uv, cur_yuv_buf->uv + cur_yuv_buf->width * cur_yuv_buf->height / 2);

        // if (frame_id == 1) {
        //     minmax_global_y.first = *minmax_frame_y.first;
        //     minmax_global_y.second = *minmax_frame_y.second;
        //     minmax_global_uv.first = *minmax_frame_uv.first;
        //     minmax_global_uv.second = *minmax_frame_uv.second;
        // } else {
        //     minmax_global_y.first = std::min(*minmax_frame_y.first, minmax_global_y.first);
        //     minmax_global_y.second = std::max(*minmax_frame_y.second, minmax_global_y.second);
        //     minmax_global_uv.first = std::min(*minmax_frame_uv.first, minmax_global_uv.first);
        //     minmax_global_uv.second = std::max(*minmax_frame_uv.second, minmax_global_uv.second);
        // }

        // fmt::print("Y min {} max {} UV min {} max {}\n", minmax_global_y.first, minmax_global_y.second, minmax_global_uv.first, minmax_global_uv.second);

        vipc_server.send(cur_yuv_buf, &extra);
   
        if (frame_id % 100 == 0)
        {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start) / 100;
            fmt::print("100 frames took {}\n", duration);

            start = std::chrono::steady_clock::now();
        }

        last_start_of_frame = start_of_frame;
    }

    return EXIT_SUCCESS;
}