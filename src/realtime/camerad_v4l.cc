// camerad uses the v4l interface to read 
#include <chrono>
#include <iostream>
#include <algorithm>
#include <thread>
#include <memory>
#include <cassert>

#include <fmt/core.h>
#include <fmt/chrono.h>

#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <libv4l2.h>

#include "nvvisionbuf.h"

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_server.h"

#include "util.h"
#include "config.h"

#define CAMERA_BUFFER_COUNT 30
#define SENSOR_TYPE_REALSENSE_D455 0x01

// Allow waiting longer for initial camera frames
#define WAIT_FOR_FRAME_TIMEOUT_MS 10'000

ExitHandler do_exit;

class V4LCamera {
    public:
    V4LCamera(std::string device, int in_width, int in_height) { 
        // Open the device
        fd = v4l2_open(device.c_str(), O_RDWR);

        if (fd == -1)
        {
            throw std::runtime_error("Could not open device");
        }

        // Check the capabilities
        struct v4l2_capability caps;
        checked_v4l2_ioctl(fd, VIDIOC_QUERYCAP, &caps);

        if (!(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE))
        {
            throw std::runtime_error("Device does not support V4L2_CAP_VIDEO_CAPTURE");
        }

        fmt::print(stderr, "Opened video device: {} with driver {}\n", reinterpret_cast<char *>(caps.card), reinterpret_cast<char *>(caps.driver));

        // Set the capture format
        struct v4l2_format fmt_in = {
            .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
            .fmt = {
                .pix = {
                    .width = static_cast<unsigned int>(in_width),
                    .height = static_cast<unsigned int>(in_height),
                    .pixelformat = V4L2_PIX_FMT_UYVY,
                    .field = V4L2_FIELD_ANY,
                }
            }
        };

        checked_v4l2_ioctl(fd, VIDIOC_S_FMT, &fmt_in);

        // TODO Adjust the framerate, might need to do a manual frameskip for example

        // Request the buffers
        struct v4l2_requestbuffers reqbuf = {
            .count = NUM_BUFFERS,
            .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
            .memory = V4L2_MEMORY_MMAP,
        };
        checked_v4l2_ioctl(fd, VIDIOC_REQBUFS, &reqbuf);

        if (reqbuf.count != NUM_BUFFERS)
        {
            throw std::runtime_error(fmt::format("Failed to allocate enough buffers, only got {} out of {}", reqbuf.count, NUM_BUFFERS));
        }

        // Query and map the buffers
        for (uint32_t i = 0; i < NUM_BUFFERS; i++) {
            v4l2_buffer v4l_buf = {0};
            v4l_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            v4l_buf.memory = V4L2_MEMORY_MMAP;
            v4l_buf.index = i;

            checked_v4l2_ioctl(fd, VIDIOC_QUERYBUF, &v4l_buf);

            NVVisionBuf localbuf = NVVisionBuf(v4l_buf.length, i);

            localbuf.planes[0].data =
                    reinterpret_cast<uint8_t *>(mmap(NULL /* start anywhere */,
                            v4l_buf.length,
                            PROT_READ | PROT_WRITE /* required */,
                            MAP_SHARED /* recommended */,
                            fd, v4l_buf.m.offset));

            if (localbuf.planes[0].data == MAP_FAILED) {
                throw std::runtime_error("Failed to map buffer");
            }

            fmt::print(stderr, "Queried buffer {} at {} with length: {}\n", i, fmt::ptr(localbuf.planes[0].data), v4l_buf.length);

            bufs.push_back(localbuf);
        }

        // Queue the buffers
        for (uint32_t i = 0; i < NUM_BUFFERS; i++) {
            v4l2_buffer v4l_buf = {0};
            v4l_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            v4l_buf.memory = V4L2_MEMORY_MMAP;
            v4l_buf.index = i;
            checked_v4l2_ioctl(fd, VIDIOC_QBUF, &v4l_buf);
        }

        // Turn on streaming
        v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        checked_v4l2_ioctl(fd, VIDIOC_STREAMON, &buf_type);
    }

    void get_frame() {
        fd_set fds = {0};
        struct timeval tv;
        int r;
      
        FD_ZERO(&fds);
        FD_SET(fd, &fds);

        /* Timeout. */
        tv.tv_sec = 2;
        tv.tv_usec = 0;

        r = select(fd + 1, &fds, NULL, NULL, &tv);

        if (-1 == r) {
            if (errno != EINTR) {      
                throw std::runtime_error("select error");
            }
        }

        if (r == 0) {
            throw std::runtime_error("select timeout");
        }

        // Dequeue the buf
        v4l2_buffer v4l_buf = {0};
        v4l_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        v4l_buf.memory = V4L2_MEMORY_MMAP;

        checked_v4l2_ioctl(fd, VIDIOC_DQBUF, &v4l_buf);

        fmt::print(stderr, "Got frame {}\n", v4l_buf.index);


    }

    ~V4LCamera() {
        v4l2_close(fd);
    }

    private:
        const uint32_t NUM_BUFFERS = 4;
        int fd;
        std::vector<NVVisionBuf> bufs;
};

int main(int argc, char *argv[])
{
    PubMaster pm{ {"headCameraState"} };
    VisionIpcServer vipc_server{ "camerad" };
    V4LCamera camera{ "/dev/video0", 1920, 1080 };
    vipc_server.create_buffers(VISION_STREAM_HEAD_COLOR, CAMERA_BUFFER_COUNT, false, CAMERA_WIDTH, CAMERA_HEIGHT);
    vipc_server.start_listener();

    fmt::print(stderr, "Ready to start streaming\n");

    // Wait for the frame, the dequeue the buffer, process it, and requeue it
    size_t count = 0;

    while (!do_exit) {
       camera.get_frame();
       count++;

       if (count > 50) {
           break;
       }
    }

    return EXIT_SUCCESS;
}