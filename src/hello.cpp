#include "NvVideoEncoder.h"

#include <librealsense2/rs.hpp> 
#include <iostream>             
#include <linux/videodev2.h>

// Encodes an Intel Real Sense stream using the NVIDIA Hardware encoder on Jetson Platform
int main(int argc, char * argv[]) try
{
    int ret;
    int num_output_buffers = 10;

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

    // Create a NVidia encoder
    NvVideoEncoder *enc = NvVideoEncoder::createVideoEncoder("enc0");

    if (!enc)
    {
        std::cout << "Failed to create NvVideoEncoder\n";
        return EXIT_FAILURE;
    }

    // Set the capture plane format
    ret =
        enc->setCapturePlaneFormat(V4L2_PIX_FMT_H265, 640, 480, 2 * 1024 * 1024);

    if (ret < 0)
    {
        std::cout << "Failed to set capture plane format\n";
        return EXIT_FAILURE;
    }

    // Set the output plane format
    ret =
        enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, 640, 480);

    if (ret < 0)
    {
        std::cout << "Failed to set output plane format\n";
        return EXIT_FAILURE;
    }

    // Set the bitrate
    ret = enc->setBitrate(4 * 1024 * 1024);

    if (ret < 0)
    {
        std::cout << "Failed to set bitrate\n";
        return EXIT_FAILURE;
    }

    /* Set encoder profile for HEVC format */
    ret = enc->setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN);

    if (ret < 0)
    {
        std::cout << "Failed to set encoder profile\n";
        return EXIT_FAILURE;
    }

    ret = enc->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_VBR);

    if (ret < 0)
    {
        std::cout << "Failed to set rate control mode\n";
        return EXIT_FAILURE;
    }

    ret = enc->setPeakBitrate(6 * 1024 * 1024);

    if (ret < 0)
    {
        std::cout << "Failed to set peak bitrate\n";
        return EXIT_FAILURE;
    }

    /* Query, Export and Map the output plane buffers so that we can read
       raw data into the buffers */
    ret = enc->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);

    if (ret < 0)
    {
        std::cout << "Failed to setup output plane\n";
        return EXIT_FAILURE;
    }

    /* Query, Export and Map the capture plane buffers so that we can write
       encoded bitstream data into the buffers */
    ret = enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, num_output_buffers, true, false);

    if (ret < 0)
    {
        std::cout << "Failed to setup capture plane\n";
        return EXIT_FAILURE;
    }
    
    /* Subscibe for End Of Stream event */
    ret = enc->subscribeEvent(V4L2_EVENT_EOS,0,0);

    if (ret < 0)
    {
        std::cout << "Failed to subscribe to EOS event\n";
        return EXIT_FAILURE;
    }

    /* set encoder output plane STREAMON */
    ret = enc->output_plane.setStreamStatus(true);

    if (ret < 0)
    {
        std::cout << "Failed to set output plane stream status\n";
        return EXIT_FAILURE;
    }

    /* set encoder capture plane STREAMON */
    ret = enc->capture_plane.setStreamStatus(true);

    if (ret < 0)
    {
        std::cout << "Failed to set capture plane stream status\n";
        return EXIT_FAILURE;
    }

    /* Enqueue all the empty capture plane buffers. */
    for (uint32_t i = 0; i < enc->capture_plane.getNumBuffers(); i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        ret = enc->capture_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            std::cout << "Error while queueing buffer at capture plane\n";
            return EXIT_FAILURE;
        }
    }

    // Enable the Realsense and start the pipeline
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_YUYV, 30);

    rs2::pipeline pipe;
    pipe.start(cfg);

    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();

        std::cout << color_frame.get_width() << "x" << color_frame.get_height() << std::endl;
        std::cout << color_frame.get_data_size() << std::endl;

        // TODO Convert from YUYV format to expected YUV420 format for encoding

        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        // TODO Change once you want to encode more than one frame
        v4l2_buf.index = 0;
        v4l2_buf.m.planes = planes;

        std::cout << "Queued buffer at output plane\n";

        ret = enc->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            std::cerr << "Error while queueing buffer at output plane" << std::endl;
            return EXIT_FAILURE;
        }

        break;
    }

    while (1)
        {
            struct v4l2_buffer v4l2_capture_buf;
            struct v4l2_plane capture_planes[MAX_PLANES];
            NvBuffer *capplane_buffer = NULL;
            bool capture_dq_continue = true;

            memset(&v4l2_capture_buf, 0, sizeof(v4l2_capture_buf));
            memset(capture_planes, 0, sizeof(capture_planes));
            v4l2_capture_buf.m.planes = capture_planes;
            v4l2_capture_buf.length = 1;

            /* Dequeue from output plane, fill the frame and enqueue it back again.
               NOTE: This could be moved out to a different thread as an optimization. */
            ret = enc->capture_plane.dqBuffer(v4l2_capture_buf, &capplane_buffer, NULL, 10);
            if (ret < 0)
            {
                if (errno == EAGAIN)
                    break;
                std::cerr << "ERROR while DQing buffer at capture plane" << std::endl;
            }

            std::cout << "DQed buffer at capture plane\n";
            /* Invoke encoder capture-plane deque buffer callback */
            // capture_dq_continue = encoder_capture_plane_dq_callback(&v4l2_capture_buf, capplane_buffer, NULL,
            //         &ctx);
            // if (!capture_dq_continue)
            // {
            //     cout << "Capture plane dequeued 0 size buffer " << endl;
            //     ctx.got_eos = true;
            //     return 0;
            // }
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