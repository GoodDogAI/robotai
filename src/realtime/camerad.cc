// camerad talks to the Intel RealSense camera and runs a visionipc server to send the frames to other services
#include <chrono>
#include <iostream>
#include <algorithm>
#include <thread>
#include <cassert>

#include <fmt/core.h>
#include <fmt/chrono.h>
#include <fmt/ranges.h>

#include <librealsense2/rs.hpp>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_server.h"

#include "util.h"
#include "config.h"

#define CAMERA_BUFFER_COUNT 30
#define SENSOR_TYPE_REALSENSE_D455 0x01

ExitHandler do_exit;

void color_sensor_thread(VisionIpcServer &vipc_server, PubMaster &pm, rs2::color_sensor &color_sens) {
    rs2::frame_queue queue{ 1 };
    color_sens.start(queue);

    uint32_t frame_id{ 0 };
    rs2_metadata_type last_start_of_frame {};
    constexpr rs2_metadata_type expected_frame_time = 1'000'000 / CAMERA_FPS; // usec

    while (!do_exit)
    {
        rs2::video_frame color_frame = queue.wait_for_frame();

        if (color_frame.get_frame_number() != frame_id + 1 && frame_id != 0)
        {
            fmt::print(stderr, "Frame number mismatch\n");
            fmt::print(stderr, "Got {} expected {}\n", color_frame.get_frame_number(), frame_id + 1);
            do_exit = true;
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
        // std::cout << "Frame " << frame_id << " at " << color_frame.get_frame_timestamp_domain() << " " << start_of_frame << std::endl;

        // for (int i = RS2_FRAME_METADATA_FRAME_COUNTER; i < RS2_FRAME_METADATA_COUNT; i++)
        // {
        //     if (color_frame.supports_frame_metadata((rs2_frame_metadata_value)i))
        //     {
        //         std::cout << "Metadata " << i << " " << color_frame.get_frame_metadata((rs2_frame_metadata_value)i) << std::endl;
        //     }
        // }

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

        MessageBuilder msg;
        auto event = msg.initEvent(true);
        auto cdat = event.initHeadCameraState();
    
        cdat.setFrameId(frame_id);
        cdat.setTimestampSof(start_of_frame);

        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
        pm.send("headCameraState", bytes.begin(), bytes.size());

        last_start_of_frame = start_of_frame;
    }

    color_sens.stop();
    color_sens.close();
}

void motion_sensor_thread(PubMaster &pm, rs2::motion_sensor &motion_sensor) {
    rs2::frame_queue queue{ 4 }; // Allow a small queue unlike video frames
    motion_sensor.start(queue);

    while (!do_exit)
    {
        rs2::motion_frame motion_frame = queue.wait_for_frame();
        auto vec = motion_frame.get_motion_data();

        MessageBuilder msg;
        auto event = msg.initEvent(true);

        if (motion_frame.get_profile().stream_type() == RS2_STREAM_GYRO) {
            auto gyro = event.initGyroscope();
            gyro.setVersion(1);
            gyro.setTimestamp(motion_frame.get_timestamp());
            gyro.setSensor(SENSOR_REALSENSE_D455);
            gyro.setType(SENSOR_TYPE_GYRO);

            auto gyrodata = gyro.initGyro();
            gyrodata.setV({vec.x, vec.y, vec.z});
            gyrodata.setStatus(true);

            auto words = capnp::messageToFlatArray(msg);
            auto bytes = words.asBytes();
            pm.send("gyroscope", bytes.begin(), bytes.size());
        } else if (motion_frame.get_profile().stream_type() == RS2_STREAM_ACCEL) {
            auto accel = event.initAccelerometer();
            accel.setVersion(1);
            accel.setTimestamp(motion_frame.get_timestamp());
            accel.setSensor(SENSOR_REALSENSE_D455);
            accel.setType(SENSOR_TYPE_ACCELEROMETER);

            auto acceldata = accel.initAcceleration();
            acceldata.setV({vec.x, vec.y, vec.z});
            acceldata.setStatus(true);

            auto words = capnp::messageToFlatArray(msg);
            auto bytes = words.asBytes();
            pm.send("accelerometer", bytes.begin(), bytes.size());
        }
    }

    motion_sensor.stop();
    motion_sensor.close();
}

void depth_sensor_thread(VisionIpcServer &vipc_server, PubMaster &pm, rs2::depth_sensor &depth_sens) {
    rs2::frame_queue queue{ 2 }; // Depth latency is less important, so allow some buffer
    depth_sens.start(queue);

    uint32_t frame_id{ 0 };
    rs2_metadata_type last_start_of_frame {};

    // Generate the lookup table for depth to YUV color mapping
    std::vector<uint8_t> depth_to_yuv_lut(std::numeric_limits<uint16_t>::max());
    std::generate(depth_to_yuv_lut.begin(), depth_to_yuv_lut.end(), [i = 0] () mutable {
        uint16_t min = 200, max = 60000;
        float val = ((float)i - min) / (max - min);
        val = std::clamp(val, 0.0f, 1.0f);



        i++;
        return val * 255;
    });
 
    // For some reason, the frame id's on the depth sensor always restart a short time after starting the queue
    // So, this workaround drains those frames off before starting to read the real frame id's
    while (!do_exit) {
        rs2::depth_frame depth_frame = queue.wait_for_frame();

        std::cout << "Draining Depth frame id " << frame_id << std::endl;
     
        if (depth_frame.get_frame_number() != frame_id + 1 && frame_id != 0)
        {
            frame_id = 0;
            break;
        }
        else
        {
            frame_id = depth_frame.get_frame_number();
        }
    }

    // Now start the real loop
    while (!do_exit)
    {
        rs2::depth_frame depth_frame = queue.wait_for_frame();
        
        if (depth_frame.get_frame_number() != frame_id + 1 && frame_id != 0)
        {
            fmt::print(stderr, "Depth Frame number mismatch\n");
            fmt::print(stderr, "Got {} expected {}\n", depth_frame.get_frame_number(), frame_id + 1);
            
            if (std::abs(static_cast<int64_t>(depth_frame.get_frame_number()) - (frame_id + 1)) > 1)
            {
                fmt::print(stderr, "Too many depth frame mismatches\n");
                do_exit = true;
                break;
            }

            frame_id = depth_frame.get_frame_number();
        }
        else
        {
            frame_id = depth_frame.get_frame_number();
        }

        auto cur_yuv_buf = vipc_server.get_buffer(VISION_STREAM_HEAD_DEPTH);
        rs2_metadata_type start_of_frame = depth_frame.get_frame_metadata(RS2_FRAME_METADATA_FRAME_TIMESTAMP);

        VisionIpcBufExtra extra {
                        frame_id,
                        static_cast<uint64_t>(start_of_frame),
                        0,
        };
        cur_yuv_buf->set_frame_id(frame_id);

        uint16_t *z16_data = (uint16_t *)depth_frame.get_data();

        int32_t depth_frame_width = depth_frame.get_width();
        int32_t depth_frame_height = depth_frame.get_height();


        // This squishes the z16 depth data down into the YUV buffer.
        // The MSB gets put into the Y, since that has the most resolution during video compression
        // The LSB we want to figure out how to stuff that into the UV if we can later.
        for(uint32_t row = 0; row < depth_frame_height / 2; row++) {
            for (uint32_t col = 0; col < depth_frame_width / 2; col++) {
                cur_yuv_buf->y[(row * 2) * cur_yuv_buf->stride + col * 2] = depth_to_yuv_lut[z16_data[(row * 2) * depth_frame_width + col * 2]];
                cur_yuv_buf->y[(row * 2) * cur_yuv_buf->stride + col * 2 + 1] = depth_to_yuv_lut[z16_data[(row * 2) * depth_frame_width + col * 2 + 1]];
                cur_yuv_buf->y[(row * 2 + 1) * cur_yuv_buf->stride + col * 2] = depth_to_yuv_lut[z16_data[(row * 2 + 1) * depth_frame_width + col * 2]];
                cur_yuv_buf->y[(row * 2 + 1) * cur_yuv_buf->stride + col * 2 + 1] = depth_to_yuv_lut[z16_data[(row * 2 + 1) * depth_frame_width + col * 2 + 1]];

                // cur_yuv_buf->uv[row * cur_yuv_buf->stride + col * 2] = std::min({z16_data[(row * 2) * cur_yuv_buf->stride + col * 2],
                //                                                                 z16_data[(row * 2) * cur_yuv_buf->stride + col * 2 + 1],
                //                                                                 z16_data[(row * 2 + 1) * cur_yuv_buf->stride + col * 2],
                //                                                                 z16_data[(row * 2 + 1) * cur_yuv_buf->stride + col * 2 + 1]}) & 0x00FF;
                // const uint16_t z16 = std::min({z16_data[(row * 2) * cur_yuv_buf->stride + col * 2],
                //                                z16_data[(row * 2) * cur_yuv_buf->stride + col * 2 + 1],
                //                                z16_data[(row * 2 + 1) * cur_yuv_buf->stride + col * 2],
                //                                z16_data[(row * 2 + 1) * cur_yuv_buf->stride + col * 2 + 1]});

                // cur_yuv_buf->uv[row * cur_yuv_buf->stride + col * 2] = z16 & 0x00FF;
                // cur_yuv_buf->uv[row * cur_yuv_buf->stride + col * 2 + 1] = (z16 & 0xFF00) >> 8;

                cur_yuv_buf->uv[row * cur_yuv_buf->stride + col * 2] = 127;
                cur_yuv_buf->uv[row * cur_yuv_buf->stride + col * 2 + 1] = 127;
            }
        }

        vipc_server.send(cur_yuv_buf, &extra);

        MessageBuilder msg;
        auto event = msg.initEvent(true);
        auto cdat = event.initDepthCameraState();
    
        cdat.setFrameId(frame_id);
        cdat.setTimestampSof(start_of_frame);

        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
        pm.send("depthCameraState", bytes.begin(), bytes.size());

        last_start_of_frame = start_of_frame;
    }

    depth_sens.stop();
    depth_sens.close();
}

int main(int argc, char *argv[])
{
    PubMaster pm{ {"headCameraState", "depthCameraState", "gyroscope", "accelerometer"} };
    VisionIpcServer vipc_server{ "camerad" };
    vipc_server.create_buffers(VISION_STREAM_HEAD_COLOR, CAMERA_BUFFER_COUNT, false, CAMERA_WIDTH, CAMERA_HEIGHT);
    vipc_server.create_buffers(VISION_STREAM_HEAD_DEPTH, CAMERA_BUFFER_COUNT, false, CAMERA_WIDTH, CAMERA_HEIGHT);
    vipc_server.start_listener();

    rs2::context ctx;
    rs2::device_list devices_list{ ctx.query_devices() };
    size_t device_count{ devices_list.size() };
    if (!device_count)
    {
        fmt::print(stderr, "No device detected. Is it plugged in?\n");
        return EXIT_FAILURE;
    }

    fmt::print("Found {} devices\n", device_count);

    rs2::device device = devices_list.front();

    fmt::print("Device Name: {}\n", device.get_info(RS2_CAMERA_INFO_NAME) );

    auto depth_sens{ device.first<rs2::depth_sensor>() };
    auto color_sens{ device.first<rs2::color_sensor>() };
    auto motion_sens{ device.first<rs2::motion_sensor>() };

    // Find a matching color profile
    auto color_profiles{ color_sens.get_stream_profiles() };

    auto color_stream_profile = *std::find_if(color_profiles.begin(), color_profiles.end(), [](rs2::stream_profile &profile)
                                        {
        rs2::video_stream_profile sp = profile.as<rs2::video_stream_profile>();
            return sp.width() == CAMERA_WIDTH && sp.height() == CAMERA_HEIGHT && sp.format() == RS2_FORMAT_YUYV && sp.fps() == CAMERA_FPS; });

    if (!color_stream_profile)
    {
        fmt::print(stderr, "No matching camera configuration found\n");
        return EXIT_FAILURE;
    }

    color_sens.open(color_stream_profile);

    // Find and setup the gyro and accelerometer streams
    auto motion_profiles{ motion_sens.get_stream_profiles() };

    std::cout << "Camera Motion profiles:" << std::endl;
    for (const auto &profile : motion_profiles)
    {
        std::cout << profile.stream_name() << " " << profile.stream_type() << " " << profile.fps() << " " << profile.format() << std::endl;
    }

    auto gyro_stream_profile = *std::find_if(motion_profiles.begin(), motion_profiles.end(), [](rs2::stream_profile &profile)
                                        {
        rs2::motion_stream_profile sp = profile.as<rs2::motion_stream_profile>();
            return sp.stream_type() == RS2_STREAM_GYRO && sp.format() == RS2_FORMAT_MOTION_XYZ32F && sp.fps() == CAMERA_GYRO_FPS; });

    if (!gyro_stream_profile)
    {
        fmt::print(stderr, "No matching gyro configuration found\n");
        return EXIT_FAILURE;
    }

    auto accel_stream_profile = *std::find_if(motion_profiles.begin(), motion_profiles.end(), [](rs2::stream_profile &profile)
                                        {
        rs2::motion_stream_profile sp = profile.as<rs2::motion_stream_profile>();
            return sp.stream_type() == RS2_STREAM_ACCEL && sp.format() == RS2_FORMAT_MOTION_XYZ32F && sp.fps() == CAMERA_ACCEL_FPS; });

    if (!accel_stream_profile)
    {
        fmt::print(stderr, "No matching accel configuration found\n");
        return EXIT_FAILURE;
    }

    motion_sens.open({ gyro_stream_profile, accel_stream_profile });

    // Find and setup the depth stream
    auto depth_profiles{ depth_sens.get_stream_profiles() };

    // std::cout << "Camera Depth profiles:" << std::endl;
    // for (const auto &profile : depth_profiles)
    // {
    //     auto sp = profile.as<rs2::video_stream_profile>();
    //     std::cout << profile.stream_name() << " " << profile.stream_type() << " " << profile.fps() << " " << profile.format() << " " << sp.width() << " " << sp.height() << std::endl;
    // }

    auto depth_stream_profile = *std::find_if(depth_profiles.begin(), depth_profiles.end(), [](rs2::stream_profile &profile)
                                        {
        rs2::video_stream_profile sp = profile.as<rs2::video_stream_profile>();
            return sp.width() == CAMERA_WIDTH && sp.height() == CAMERA_HEIGHT && sp.format() == RS2_FORMAT_Z16 && sp.fps() == CAMERA_FPS; });

    if (!depth_stream_profile)
    {
        fmt::print(stderr, "No matching depth configuration found\n");
        return EXIT_FAILURE;
    }

    depth_sens.open(depth_stream_profile);

    // Start all the stream threads
    std::thread color_thread{ color_sensor_thread, std::ref(vipc_server), std::ref(pm), std::ref(color_sens) };
    std::thread motion_thread{ motion_sensor_thread, std::ref(pm), std::ref(motion_sens) };
    std::thread depth_thread{ depth_sensor_thread, std::ref(vipc_server), std::ref(pm), std::ref(depth_sens) };

    // Wait for any of the threads to finish
    color_thread.join();
    motion_thread.join();
    depth_thread.join();

    return EXIT_SUCCESS;
}