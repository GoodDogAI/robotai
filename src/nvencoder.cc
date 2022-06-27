#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <libv4l2.h>


#include "v4l2_nv_extensions.h"

#include "cereal/visionipc/visionbuf.h"
#include "nvencoder.h"
#include "nvvisionbuf.h"
#include "util.h"

#define BUF_OUT_COUNT 6
#define BUF_IN_COUNT 6


// NVENC documentation: https://docs.nvidia.com/jetson/l4t-multimedia/group__V4L2Enc.html
// Many ideas and code From Openpilot encoderd
static int dequeue_buffer(int fd, v4l2_buf_type buf_type, unsigned int *index = NULL, unsigned int *bytesused = NULL, unsigned int *flags = NULL, struct timeval *timestamp = NULL)
{
    int ret;
    v4l2_plane plane = {0};
    v4l2_buffer v4l_buf = {0};

    v4l_buf.type = buf_type;
    v4l_buf.memory = V4L2_MEMORY_MMAP;
    v4l_buf.m.planes = &plane;
    v4l_buf.length = 1;

    ret = v4l2_ioctl(fd, VIDIOC_DQBUF, &v4l_buf);

    if (ret < 0)
    {
        if (errno == EAGAIN)
            return 0;

        std::cerr << "Failed to dequeue buffer (" << errno << ") : " << strerror(errno) << std::endl;
        exit(1);
    }

    if (index)
        *index = v4l_buf.index;
    if (bytesused)
        *bytesused = v4l_buf.m.planes[0].bytesused;
    if (flags)
        *flags = v4l_buf.flags;
    if (timestamp)
        *timestamp = v4l_buf.timestamp;
    assert(v4l_buf.m.planes[0].data_offset == 0);
    
    return 1;
}

static void queue_buffer(int fd, v4l2_buf_type buf_type, unsigned int index, NVVisionBuf *buf)
{
    v4l2_buffer v4l_buf = {0};

    v4l_buf.type = buf_type;
    v4l_buf.index = index;
    v4l_buf.memory = V4L2_MEMORY_MMAP;
    v4l_buf.length = buf->n_planes;

    v4l2_plane planes[MAX_PLANES];
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));
    v4l_buf.m.planes = planes;

    for (int i = 0; i < buf->n_planes; i++)
    {
        // v4l_buf.m.planes[i].m.userptr = (unsigned long)buf->planes[i].data;
        // v4l_buf.m.planes[i].length = buf->planes[i].fmt.sizeimage;
        v4l_buf.m.planes[i].bytesused = buf->planes[i].fmt.sizeimage;
    }

    buf->is_queued = true;
    checked_v4l2_ioctl(fd, VIDIOC_QBUF, &v4l_buf);
}

static void request_buffers(int fd, v4l2_buf_type buf_type, uint32_t count)
{
    struct v4l2_requestbuffers reqbuf = {
        .count = count,
        .type = buf_type,
        .memory = V4L2_MEMORY_MMAP,
    };
    checked_v4l2_ioctl(fd, VIDIOC_REQBUFS, &reqbuf);

    if (reqbuf.count != count)
    {
        std::cerr << "Failed to allocate enough buffers, only got " << reqbuf.count << " out of " << count << std::endl;
        exit(EXIT_FAILURE);
    }
}

static void query_and_map_buffers(int fd, v4l2_buf_type buf_type, std::vector<NVVisionBuf> &bufs)
{
    for (uint32_t i = 0; i < bufs.size(); i++)
    {
        v4l2_buffer v4l_buf = {0};
        v4l2_plane planes[MAX_PLANES];
        v4l_buf.type = buf_type;
        v4l_buf.memory = V4L2_MEMORY_MMAP;
        v4l_buf.index = i;
        v4l_buf.length = bufs[i].n_planes;
        v4l_buf.m.planes = planes;
        checked_v4l2_ioctl(fd, VIDIOC_QUERYBUF, &v4l_buf);

        // Export and Map each plane
        for (uint32_t p = 0; p < bufs[i].n_planes; p++)
        {
            v4l2_exportbuffer expbuf = {0};
            expbuf.type = buf_type;
            expbuf.index = i;
            expbuf.plane = p;
            checked_v4l2_ioctl(fd, VIDIOC_EXPBUF, &expbuf);

            bufs[i].planes[p].fd = expbuf.fd;
            bufs[i].planes[p].length = v4l_buf.m.planes[p].length;
            bufs[i].planes[p].mem_offset = v4l_buf.m.planes[p].m.mem_offset;

            bufs[i].planes[p].data = (uint8_t *)mmap(NULL,  bufs[i].planes[p].length, PROT_READ | PROT_WRITE, MAP_SHARED,  bufs[i].planes[p].fd, bufs[i].planes[p].mem_offset);

            if (bufs[i].planes[p].data == MAP_FAILED)
            {
                std::cerr << "Failed to mmap buffer" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}

NVEncoder::NVEncoder(std::string encoderdev, int in_width, int in_height, int out_width, int out_height, int bitrate, int fps):
   in_width(in_width), in_height(in_height), out_width(out_width), out_height(out_height), bitrate(bitrate), fps(fps)
{
    fd = v4l2_open(encoderdev.c_str(), O_RDWR);

    if (fd == -1)
    {
        throw std::runtime_error("Could not open device");
    }

    struct v4l2_capability caps;
    checked_v4l2_ioctl(fd, VIDIOC_QUERYCAP, &caps);

    if (!(caps.capabilities & V4L2_CAP_VIDEO_M2M_MPLANE))
    {
        throw std::runtime_error("Device does not support V4L2_CAP_VIDEO_M2M_MPLANE");
    }

    struct v4l2_format fmt_out = {
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
        .fmt = {
        .pix_mp = {
            // downscales are free with v4l
            .width = (unsigned int)out_width,
            .height = (unsigned int)out_height,
            .pixelformat = V4L2_PIX_FMT_H265,
            // .field = V4L2_FIELD_ANY,
            // .colorspace = V4L2_COLORSPACE_DEFAULT,
        }
        }
    };

    fmt_out.fmt.pix_mp.num_planes = 1;

    checked_v4l2_ioctl(fd, VIDIOC_S_FMT, &fmt_out);
    

    //TODO Is it necessary to configure the framerate?


    struct v4l2_format fmt_in = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .fmt = {
      .pix_mp = {
        .width = (unsigned int)in_width,
        .height = (unsigned int)in_height,
        .pixelformat = V4L2_PIX_FMT_NV12M,
        //.field = V4L2_FIELD_ANY,
        //.colorspace = V4L2_COLORSPACE_470_SYSTEM_BG,
      }
        }
    };
    fmt_in.fmt.pix_mp.num_planes = 2;

    checked_v4l2_ioctl(fd, VIDIOC_S_FMT, &fmt_in);

    std::cout << "in buffer size " << fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage << " out buffer size " << fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage << std::endl;

    // shared ctrls
    {
        struct v4l2_control ctrls[] = {
            { .id = V4L2_CID_MPEG_VIDEO_HEADER_MODE, .value = V4L2_MPEG_VIDEO_HEADER_MODE_JOINED_WITH_1ST_FRAME},
            { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = bitrate},
            { .id = V4L2_CID_MPEG_VIDEO_BITRATE_MODE, .value = V4L2_MPEG_VIDEO_BITRATE_MODE_VBR},
        };
        for (auto ctrl : ctrls) {
            checked_v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrl);
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

    // initialize output buffers
    for (unsigned int i = 0; i < BUF_OUT_COUNT; i++) {
        NVVisionBuf buf = NVVisionBuf(fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage, i);
        buf_out.push_back(buf);
    }

    // map the output buffers
    query_and_map_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, buf_out);

    // queue the output buffers
    for (unsigned int i = 0; i < BUF_OUT_COUNT; i++) {
        queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, i, &buf_out[i]);
    }

    // setup input buffers
    BufferPlaneFormat yuv_format[2];
    yuv_format[0].width = in_width;
    yuv_format[0].height = in_height;
    yuv_format[0].bytesperpixel = 1;
    yuv_format[0].stride = fmt_in.fmt.pix_mp.plane_fmt[0].bytesperline;
    yuv_format[0].sizeimage = fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage;

    yuv_format[1].width = in_width; 
    yuv_format[1].height = in_height / 2;
    yuv_format[1].bytesperpixel = 1;
    yuv_format[1].stride = fmt_in.fmt.pix_mp.plane_fmt[1].bytesperline;
    yuv_format[1].sizeimage = fmt_in.fmt.pix_mp.plane_fmt[1].sizeimage;

    for (uint32_t i = 0; i < BUF_IN_COUNT; i++) {
        NVVisionBuf buf = NVVisionBuf(2, yuv_format, i);
        buf_in.push_back(buf);
    }

    // map the input buffers
    query_and_map_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, buf_in);

    // start encoder
    v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    checked_v4l2_ioctl(fd, VIDIOC_STREAMON, &buf_type);
    buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    checked_v4l2_ioctl(fd, VIDIOC_STREAMON, &buf_type);

    // start the dequeue threads
    dequeue_capture_thread = std::thread { &NVEncoder::do_dequeue_capture, this };
    dequeue_output_thread = std::thread { &NVEncoder::do_dequeue_output, this };
}

NVEncoder::~NVEncoder() {
    v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    checked_v4l2_ioctl(fd, VIDIOC_STREAMOFF, &buf_type);
    request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, 0);
    
    buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    checked_v4l2_ioctl(fd, VIDIOC_STREAMOFF, &buf_type);
    request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, 0);

    dequeue_capture_thread.join();
    dequeue_output_thread.join();

    std::cout << "Closing encoder" << std::endl;
    v4l2_close(fd);
}

void NVEncoder::do_dequeue_capture() {
    while(true) {
        uint32_t index, bytesused;
    
        if (dequeue_buffer(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, &index, &bytesused)) 
        {
            auto &promise = encoder_promises[frame_read_index];
            
            promise.set_value(std::make_unique<NVResult>( buf_out[index].planes[0].data,
                bytesused,
                index));
            encoder_promises.erase(frame_read_index);
            frame_read_index++;
            
            // Requeue the buffer
            queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, index, &buf_out[index]);
        }
    }
}

void NVEncoder::do_dequeue_output() {
    while(true) {
        uint32_t index;

        if (dequeue_buffer(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, &index)) {
            buf_in[index].is_queued = false;
        }
    }
}


std::future<std::unique_ptr<NVResult>> NVEncoder::encode_frame(VisionBuf* ipcbuf, VisionIpcBufExtra *extra) {
    // Find an empty buf
    auto buf = std::find_if(buf_in.begin(), buf_in.end(), [](NVVisionBuf &b) {
        return !b.is_queued;
    });

    assert(buf != buf_in.end());

    //auto copy_start = std::chrono::steady_clock::now();
    memcpy(buf->planes[0].data, ipcbuf->y, ipcbuf->uv_offset);
    memcpy(buf->planes[1].data, ipcbuf->uv, ipcbuf->uv_offset / 2);
    //std::cout << "Copy time: " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_start).count() << std::endl;

    auto &promise = encoder_promises[frame_write_index];
    frame_write_index++;

    queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, buf->index, &(*buf));

    std::cout << "Sent buffer " << (frame_write_index - 1) << std::endl;
    return promise.get_future();
}