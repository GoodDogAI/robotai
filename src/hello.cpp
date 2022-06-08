#include "NvVideoEncoder.h"

#include <assert.h>
#include <librealsense2/rs.hpp> 
#include <iostream>             
#include <fstream>
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
        enc->setCapturePlaneFormat(V4L2_PIX_FMT_H265, 1280, 720, 2 * 1024 * 1024);

    if (ret < 0)
    {
        std::cout << "Failed to set capture plane format\n";
        return EXIT_FAILURE;
    }

    // Set the output plane format
    ret =
        enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, 1280, 720);

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
    //ret = enc->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);
    ret = enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);

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
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_YUYV, 15);

    rs2::pipeline pipe;
    pipe.start(cfg);

    for (uint32_t i = 0; i < enc->output_plane.getNumBuffers(); i++)
    {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();

        uint8_t *yuyv_data = (uint8_t *)color_frame.get_data();

        std::cout << color_frame.get_width() << "x" << color_frame.get_height() << std::endl;
        std::cout << color_frame.get_data_size() << std::endl;

        // TODO Convert from YUYV format to expected YUV420 format for encoding

        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *buffer = enc->output_plane.getNthBuffer(i);

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        // assert(buffer->n_planes == 3);
        // assert(buffer->planes[0].fmt.stride * buffer->planes[0].fmt.height == 640 * 480);
        // assert(buffer->planes[1].fmt.stride * buffer->planes[1].fmt.height == 640 * 480 / 4);
        // assert(buffer->planes[2].fmt.stride * buffer->planes[2].fmt.height == 640 * 480 / 4);

        buffer->planes[0].bytesused = buffer->planes[0].fmt.stride * buffer->planes[0].fmt.height;
        buffer->planes[1].bytesused = buffer->planes[1].fmt.stride * buffer->planes[1].fmt.height;
        buffer->planes[2].bytesused = buffer->planes[2].fmt.stride * buffer->planes[2].fmt.height;
    
        // Copy the YUYV rows into the Y plane

        for(uint32_t row = 0; row < color_frame.get_height(); row++) {
            for (uint32_t col = 0; col < color_frame.get_width(); col++) {
                buffer->planes[0].data[row * buffer->planes[0].fmt.stride + col] = yuyv_data[row * color_frame.get_stride_in_bytes() + col * 2];
            }
        }

        for(uint32_t row = 0; row < color_frame.get_height() / 2; row++) {
            for (uint32_t col = 0; col < color_frame.get_width() / 2; col++) {
                buffer->planes[1].data[row * buffer->planes[1].fmt.stride + col] = yuyv_data[row * 2 * color_frame.get_stride_in_bytes() + col * 4 + 1];
                buffer->planes[2].data[row * buffer->planes[2].fmt.stride + col] = yuyv_data[row * 2 * color_frame.get_stride_in_bytes() + col * 4 + 3];
            }
        }

        // No chroma data for testing
        // memset(buffer->planes[1].data, 0, buffer->planes[1].bytesused);
        // memset(buffer->planes[2].data, 0, buffer->planes[2].bytesused);

        std::cout << "Queued buffer at output plane\n";

        ret = enc->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            std::cerr << "Error while queueing buffer at output plane" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Open an output file for writing the encoded data
    std::ofstream outfile("output.h264", std::ios::out | std::ios::binary);


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

            std::cout << "DQed capture buffer " << capplane_buffer->planes[0].bytesused << " bytes used " << capplane_buffer->n_planes << " planes " << std::endl;

            // Write buffer to output file
            outfile.write((char *)capplane_buffer->planes[0].data, capplane_buffer->planes[0].bytesused);

            // Queue the buffer back
            ret = enc->capture_plane.qBuffer(v4l2_capture_buf, NULL);
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