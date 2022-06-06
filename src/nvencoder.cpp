#include "util.h"

#include <assert.h>
#include <librealsense2/rs.hpp> 
#include <iostream>             
#include <fstream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <libv4l2.h>
#include "v4l2_nv_extensions.h"

#define ENCODER_DEV "/dev/nvhost-msenc"
#define ENCODER_COMP_NAME "NVENC"

const int MAIN_BITRATE = 10000000;

// NVENC documentation: https://docs.nvidia.com/jetson/l4t-multimedia/group__V4L2Enc.html


// Encodes an Intel Real Sense stream using the NVIDIA Hardware encoder on Jetson Platform
int main(int argc, char * argv[]) try
{
    int ret;

    int num_output_buffers = 10;
    int out_width = 1920, out_height = 1080;
    int in_width = 1920, in_height = 1080;

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

   
    // Create the v4l encoder manually
    int fd = v4l2_open(ENCODER_DEV, O_RDWR);

    if (fd == -1)
    {
        std::cerr << "Could not open device '" << ENCODER_DEV << "'" << std::endl;
        return EXIT_FAILURE;
    }

    struct v4l2_capability caps;
    checked_v4l2_ioctl(fd, VIDIOC_QUERYCAP, &caps);

    if (!(caps.capabilities & V4L2_CAP_VIDEO_M2M_MPLANE))
    {
        std::cerr << "Device does not support V4L2_CAP_VIDEO_M2M_MPLANE";
        return EXIT_FAILURE;
    }

    struct v4l2_format fmt_out = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        .fmt = {
        .pix_mp = {
            // downscales are free with v4l
            .width = (unsigned int)out_width,
            .height = (unsigned int)out_height,
            .pixelformat = V4L2_PIX_FMT_H265,
            .field = V4L2_FIELD_ANY,
            .colorspace = V4L2_COLORSPACE_DEFAULT,
        }
        }
    };

    // TODO Configure planes themselves
    fmt_out.fmt.pix_mp.num_planes = 1;

    checked_v4l2_ioctl(fd, VIDIOC_S_FMT, &fmt_out);
    

    //TODO Is it necessary to configure the framerate?


    struct v4l2_format fmt_in = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .fmt = {
      .pix_mp = {
        .width = (unsigned int)in_width,
        .height = (unsigned int)in_height,
        .pixelformat = V4L2_PIX_FMT_YUV420M,
        .field = V4L2_FIELD_ANY,
        //.colorspace = V4L2_COLORSPACE_470_SYSTEM_BG,
      }
        }
    };
    fmt_in.fmt.pix_mp.num_planes = 3;

    checked_v4l2_ioctl(fd, VIDIOC_S_FMT, &fmt_in);

    std::cout << "in buffer size " << fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage << " out buffer size " << fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage << std::endl;

    // shared ctrls
    {
        struct v4l2_control ctrls[] = {
            { .id = V4L2_CID_MPEG_VIDEO_HEADER_MODE, .value = V4L2_MPEG_VIDEO_HEADER_MODE_JOINED_WITH_1ST_FRAME},
            { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = MAIN_BITRATE},
            { .id = V4L2_CID_MPEG_VIDEO_BITRATE_MODE, .value = V4L2_MPEG_VIDEO_BITRATE_MODE_VBR},
        };
        for (auto ctrl : ctrls) {
            checked_v4l2_ioctl(fd, VIDIOC_S_CTRL, &ctrl);
        }
    }

    //H265 controls
    {
        struct v4l2_control ctrls[] = {
        { .id = V4L2_CID_MPEG_VIDEO_H265_PROFILE, .value = V4L2_MPEG_VIDEO_H265_PROFILE_MAIN},
        //   { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5},
        //   { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, .value = 29},
        //   { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, .value = 0},
        };
        for (auto ctrl : ctrls) {
            checked_v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrl);
        }
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