#include <assert.h>
#include <librealsense2/rs.hpp> 
#include <iostream>             
#include <fstream>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <libv4l2.h>
#include "v4l2_nv_extensions.h"

#include "util.h"
#include "visionbuf.h"

#define ENCODER_DEV "/dev/nvhost-msenc"
#define ENCODER_COMP_NAME "NVENC"

#define BUF_IN_COUNT 6
#define BUF_OUT_COUNT 6

const int MAIN_BITRATE = 10000000;

// NVENC documentation: https://docs.nvidia.com/jetson/l4t-multimedia/group__V4L2Enc.html

// From Openpilot encoderd
// static void dequeue_buffer(int fd, v4l2_buf_type buf_type, unsigned int *index=NULL, unsigned int *bytesused=NULL, unsigned int *flags=NULL, struct timeval *timestamp=NULL) {
//   v4l2_plane plane = {0};
//   v4l2_buffer v4l_buf = {
//     .type = buf_type,
//     .memory = V4L2_MEMORY_USERPTR,
//     .m = { .planes = &plane, },
//     .length = 1,
//   };
//   checked_ioctl(fd, VIDIOC_DQBUF, &v4l_buf);

//   if (index) *index = v4l_buf.index;
//   if (bytesused) *bytesused = v4l_buf.m.planes[0].bytesused;
//   if (flags) *flags = v4l_buf.flags;
//   if (timestamp) *timestamp = v4l_buf.timestamp;
//   assert(v4l_buf.m.planes[0].data_offset == 0);
// }

static void queue_buffer(int fd, v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, struct timeval timestamp={0}, unsigned int bytesused=0) {
  v4l2_buffer v4l_buf = {0};

  v4l_buf.type = buf_type;
  v4l_buf.index = index;
  v4l_buf.memory = V4L2_MEMORY_USERPTR;
  v4l_buf.length = buf->n_planes;

  v4l2_plane planes[MAX_PLANES];
  memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
  v4l_buf.m.planes = planes;


//   = {
//     .type = buf_type,
//     .index = index,
//     .memory = V4L2_MEMORY_USERPTR,
//     .length = 1,
//     .bytesused = 0,
//     .flags = V4L2_BUF_FLAG_TIMESTAMP_COPY,
//     .timestamp = timestamp
//   };

  checked_v4l2_ioctl(fd, VIDIOC_QBUF, &v4l_buf);
}

static void request_buffers(int fd, v4l2_buf_type buf_type, uint32_t count) {
  struct v4l2_requestbuffers reqbuf = {
    .count = count,
    .type = buf_type,
    .memory = V4L2_MEMORY_USERPTR,
  };
  checked_v4l2_ioctl(fd, VIDIOC_REQBUFS, &reqbuf);
}

// Encodes an Intel Real Sense stream using the NVIDIA Hardware encoder on Jetson Platform
int main(int argc, char * argv[]) try
{
    int ret;
    int out_width = 1920, out_height = 1080;
    int in_width = 1920, in_height = 1080;

    std::vector<VisionBuf> buf_out;
    std::vector<VisionBuf> buf_in;

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
            { .id = V4L2_CID_MPEG_VIDEOENC_NUM_BFRAMES, .value = 0},
        };
        for (auto ctrl : ctrls) {
            checked_v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrl);
        }
    }

    // Allocate the buffers
    request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, BUF_OUT_COUNT);
    request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, BUF_IN_COUNT);

    // start encoder
    v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    checked_v4l2_ioctl(fd, VIDIOC_STREAMON, &buf_type);
    buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    checked_v4l2_ioctl(fd, VIDIOC_STREAMON, &buf_type);

    // queue up output buffers
    for (unsigned int i = 0; i < BUF_OUT_COUNT; i++) {
        VisionBuf buf = VisionBuf(fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage, i);
        buf.allocate();
        buf_out.push_back(buf);
        
        queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, &buf_out[i]);
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