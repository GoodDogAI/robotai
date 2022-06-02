// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <iostream>             // for cout

// Hello RealSense example demonstrates the basics of connecting to a RealSense device
// and taking advantage of depth data
int main(int argc, char * argv[]) try
{
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

    std::cout << "Device Name: " << device.get_info(RS2_CAMERA_INFO_NAME ) << std::endl;

    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    rs2::pipeline pipe;
    pipe.start(cfg);

    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();

        std::cout << "Frameset size: " << frames.size() << std::endl;
    }



    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}