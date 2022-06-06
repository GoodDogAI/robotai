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
// Many ideas and code From Openpilot encoderd


static void dequeue_buffer(int fd, v4l2_buf_type buf_type, unsigned int *index=NULL, unsigned int *bytesused=NULL, unsigned int *flags=NULL, struct timeval *timestamp=NULL) {
    v4l2_plane plane = {0};
    v4l2_buffer v4l_buf = {0};

    v4l_buf.type = buf_type;
    v4l_buf.memory = V4L2_MEMORY_USERPTR;
    v4l_buf.m.planes = &plane;
    v4l_buf.length = 1;

    checked_v4l2_ioctl(fd, VIDIOC_DQBUF, &v4l_buf);

    if (index) *index = v4l_buf.index;
    if (bytesused) *bytesused = v4l_buf.m.planes[0].bytesused;
    if (flags) *flags = v4l_buf.flags;
    if (timestamp) *timestamp = v4l_buf.timestamp;
    assert(v4l_buf.m.planes[0].data_offset == 0);
}

static void queue_buffer(int fd, v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, struct timeval timestamp={0}, unsigned int bytesused=0) {
    v4l2_buffer v4l_buf = {0};

    v4l_buf.type = buf_type;
    v4l_buf.index = index;
    v4l_buf.memory = V4L2_MEMORY_USERPTR;
    v4l_buf.length = buf->n_planes;

    v4l2_plane planes[MAX_PLANES];
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
    v4l_buf.m.planes = planes;

    for (int i = 0; i < buf->n_planes; i++) {
        v4l_buf.m.planes[i].m.userptr = (unsigned long)buf->planes[i].data;
        v4l_buf.m.planes[i].length = buf->planes[i].fmt.sizeimage;
        v4l_buf.m.planes[i].bytesused = bytesused;
    }
    
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
    int out_width = 1280, out_height = 720;
    int in_width = 1280, in_height = 720;

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

    // setup input buffers
    BufferPlaneFormat yuv_format[3];
    yuv_format[0].width = in_width;
    yuv_format[0].height = in_height;
    yuv_format[0].bytesperpixel = 1;
    yuv_format[0].stride = in_width;
    yuv_format[0].sizeimage = in_width * in_height;

    yuv_format[1].width = in_width / 2; 
    yuv_format[1].height = in_height / 2;
    yuv_format[1].bytesperpixel = 1;
    yuv_format[1].stride = in_width / 2;
    yuv_format[1].sizeimage = in_width * in_height / 4;

    yuv_format[2].width = in_width / 2;
    yuv_format[2].height = in_height / 2;
    yuv_format[2].bytesperpixel = 1;
    yuv_format[2].stride = in_width / 2;
    yuv_format[2].sizeimage = in_width * in_height / 4;

    for (uint32_t i = 0; i < BUF_IN_COUNT; i++) {
        VisionBuf buf = VisionBuf(3, yuv_format, i);
        buf.allocate();
        buf_in.push_back(buf);
    }

    // Enable the Realsense and start the pipeline
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, in_width, in_height, RS2_FORMAT_YUYV, 15);

    rs2::pipeline pipe;
    pipe.start(cfg);

    for (uint32_t i = 0; i < BUF_IN_COUNT; i++) {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();

        uint8_t *yuyv_data = (uint8_t *)color_frame.get_data();

        std::cout << color_frame.get_width() << "x" << color_frame.get_height() << std::endl;
        std::cout << color_frame.get_data_size() << std::endl;

        // Grab an input buffer
        VisionBuf buf = buf_in[i];

        queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, i, &buf);
    }

    // Grab the capture buffers with H265 data
    for (uint32_t i = 0; i < BUF_OUT_COUNT; i++) {
        uint32_t index, bytesused;
        
        dequeue_buffer(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, &index, &bytesused);

        std::cout << "dequeued buffer " << index << " bytesused " << bytesused << std::endl;
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