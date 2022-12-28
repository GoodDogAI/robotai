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
#include "camerad/NvBufSurface.h"

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_server.h"

#include "util.h"
#include "config.h"

constexpr uint32_t CAMERA_BUFFER_COUNT = 20;

constexpr uint32_t CAPTURE_WIDTH = 1920;
constexpr uint32_t CAPTURE_HEIGHT = 1080;
constexpr uint32_t CAPTURE_FPS = 30;
constexpr uint32_t CAPTURE_SATURATION = 8;

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

        fmt::print(stderr, "Opened video device: {} with driver {} and caps {:x}\n", reinterpret_cast<char *>(caps.card), reinterpret_cast<char *>(caps.driver), caps.capabilities);

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

        v4l2_streamparm streamparm = {
            .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
            .parm = {
                .output = {
                    .timeperframe = {
                        .numerator = 1,
                        .denominator = static_cast<unsigned int>(CAPTURE_FPS),
                    }
                }
            }
        };
        checked_v4l2_ioctl(fd, VIDIOC_S_PARM, &streamparm);

        // Read back the FPS to make sure it was set correctly
        checked_v4l2_ioctl(fd, VIDIOC_G_PARM, &streamparm);

        if (streamparm.parm.capture.timeperframe.numerator != 1 || streamparm.parm.capture.timeperframe.denominator != CAPTURE_FPS) {
            throw std::runtime_error("Failed to set desired FPS, check supported FPS with v4l2-ctl");
        }

        // Set the camera saturation
        struct v4l2_control ctrl_saturation = {
            .id = V4L2_CID_SATURATION,
            .value = CAPTURE_SATURATION,
        };

        checked_v4l2_ioctl(fd, VIDIOC_S_CTRL, &ctrl_saturation);

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

            localbuf.planes[0].fd = fd;
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

   ~V4LCamera() {
        // Turn off streaming
        v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        checked_v4l2_ioctl(fd, VIDIOC_STREAMOFF, &buf_type);

        v4l2_close(fd);
    }

    V4LCamera(const V4LCamera&) = delete;
    V4LCamera& operator=(const V4LCamera&) = delete;

    struct FrameRequeueDeleter {
        void operator()(NVVisionBuf* b) { 
            v4l2_buffer v4l_buf = {0};
            v4l_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            v4l_buf.memory = V4L2_MEMORY_MMAP;
            v4l_buf.index = b->index;
            checked_v4l2_ioctl(b->planes[0].fd, VIDIOC_QBUF, &v4l_buf);

            b->is_queued = true;
        }
    };

    std::unique_ptr<NVVisionBuf, FrameRequeueDeleter> get_frame() {
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
        
        NVVisionBuf* buf = &bufs[v4l_buf.index];
        buf->is_queued = false;
        buf->frame_id = v4l_buf.sequence;
        buf->frame_time = v4l_buf.timestamp;

        return std::unique_ptr<NVVisionBuf, FrameRequeueDeleter>(buf);
    }

    private:
        const uint32_t NUM_BUFFERS = 4;
        int fd;
        std::vector<NVVisionBuf> bufs;
};

class NVFormatConverter {
    public:
    NVFormatConverter(uint32_t in_width, uint32_t in_height, NvBufSurfaceColorFormat in_fmt, 
                      uint32_t out_width, uint32_t out_height, NvBufSurfaceColorFormat out_fmt) {
        int ret;

        input_params = {
            .width = in_width,
            .height = in_height,
            .colorFormat = in_fmt,
            .layout = NVBUF_LAYOUT_PITCH,
            .memType = NVBUF_MEM_SURFACE_ARRAY,
            .memtag = NvBufSurfaceTag_VIDEO_CONVERT,
        };
        
        src_fmt_bytes_per_pixel = {2};


        ret = NvBufSurf::NvAllocate(&input_params, 1, &in_dmabuf_fd);
        if (ret)
        {
            throw std::runtime_error("Failed to allocate input buffer");
        }

        output_params = {
            .width = out_width,
            .height = out_height,
            .colorFormat = out_fmt,
            .layout = NVBUF_LAYOUT_PITCH,
            .memType = NVBUF_MEM_SURFACE_ARRAY,
            .memtag = NvBufSurfaceTag_VIDEO_CONVERT,
        };

        dest_fmt_bytes_per_pixel = {1, 2};

        ret = NvBufSurf::NvAllocate(&output_params, 1, &out_dmabuf_fd);
        if (ret)
        {
            throw std::runtime_error("Failed to allocate output buffer");
        }

        transform_params = {
            .src_width = in_width,
            .src_height = in_height,
            .src_top = 0,
            .src_left = 0,

            .dst_width = out_width,
            .dst_height = out_height,
            .dst_top = 0,
            .dst_left = 0,

            .flag = static_cast<NvBufSurfTransform_Transform_Flag>(NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_FLIP),

            .flip = NvBufSurfTransform_Rotate180,
            .filter = NvBufSurfTransformInter_Bilinear,
        };

        config_params = {};
        NvBufSurfTransformSetSessionParams (&config_params);

        fmt::print(stderr, "NVFormatConverter initialized, compute mode {:x}\n", (uint32_t)config_params.compute_mode);
       
    }

    ~NVFormatConverter() {
        if (in_dmabuf_fd != -1)
        {
            NvBufSurf::NvDestroy(in_dmabuf_fd);
        }

        if (out_dmabuf_fd != -1)
        {
            NvBufSurf::NvDestroy(out_dmabuf_fd);
        }
    }

    NVFormatConverter(const NVFormatConverter&) = delete;
    NVFormatConverter& operator=(const NVFormatConverter&) = delete;

    // Converts from arbitrary input location to a NV12 VisionIPC Buf
    void convert(uint8_t *in_buf, VisionBuf *out_buf) {
        int ret;

        // Setup the input buffer
        NvBufSurface *in_nvbuf_surf = 0;
        ret = NvBufSurfaceFromFd(in_dmabuf_fd, (void**)(&in_nvbuf_surf));
        if (ret)
        {
            throw std::runtime_error("Failed to get input buffer surface");
        }

        ret = NvBufSurfaceMap(in_nvbuf_surf, 0, 0, NVBUF_MAP_READ_WRITE);
        if (ret)
        {
            throw std::runtime_error("Failed to map input buffer surface");
        }
        
        ret = NvBufSurfaceSyncForCpu(in_nvbuf_surf, 0, 0);
        if (ret)
        {
            throw std::runtime_error("Failed to sync input buffer surface");
        }

        // Load the data into that buffer
        ret = Raw2NvBufSurface(in_buf, 0, 0, input_params.width, input_params.height, in_nvbuf_surf);
        if (ret)
        {
            throw std::runtime_error("Failed to copy input buffer");
        }

        // Sync that back to device memory
        ret = NvBufSurfaceSyncForDevice(in_nvbuf_surf, 0, 0);
        if (ret)
        {
            throw std::runtime_error("Failed to sync input buffer surface");
        }

        ret = NvBufSurfaceUnMap(in_nvbuf_surf, 0, 0);
        if (ret)
        {
            throw std::runtime_error("Failed to unmap input buffer surface");
        }

        // Perform the transform itself
        ret = NvBufSurf::NvTransform(&transform_params, in_dmabuf_fd, out_dmabuf_fd);
        if (ret)
        {
            throw std::runtime_error("Failed to transform buffer");
        }

        // Setup the output buffer for plane 0
        NvBufSurface *out_nvbuf_surf = 0;
        ret = NvBufSurfaceFromFd(out_dmabuf_fd, (void**)(&out_nvbuf_surf));
        if (ret)
        {
            throw std::runtime_error("Failed to get output buffer surface");
        }

        ret = NvBufSurfaceMap(out_nvbuf_surf, 0, 0, NVBUF_MAP_READ_WRITE);
        if (ret)
        {
            throw std::runtime_error("Failed to map output buffer surface");
        }
        
        ret = NvBufSurfaceSyncForCpu(out_nvbuf_surf, 0, 0);
        if (ret)
        {
            throw std::runtime_error("Failed to sync output buffer surface");
        }

        ret = NvBufSurface2Raw(out_nvbuf_surf, 0, 0, output_params.width, output_params.height, out_buf->y);
        if (ret)
        {
            throw std::runtime_error("Failed to copy output buffer0");
        }

        ret = NvBufSurface2Raw(out_nvbuf_surf, 0, 1, output_params.width / 2, output_params.height / 2, out_buf->uv);
        if (ret)
        {
            throw std::runtime_error("Failed to copy output buffer1");
        }

        ret = NvBufSurfaceUnMap(out_nvbuf_surf, 0, 0);
        if (ret)
        {
            throw std::runtime_error("Failed to unmap output buffer surface");
        }
    }

    private:
        int in_dmabuf_fd;
        int out_dmabuf_fd;
        NvBufSurf::NvCommonAllocateParams input_params;
        NvBufSurf::NvCommonAllocateParams output_params;
        NvBufSurf::NvCommonTransformParams transform_params;
        NvBufSurfTransformConfigParams config_params;
        std::vector<int> src_fmt_bytes_per_pixel;
        std::vector<int> dest_fmt_bytes_per_pixel;
};

// TODOs
// - Check on how the buffers are being created/destroyed, MMAPPING vs userbuffers
// - We get full-range Y data from the sensors now, but it will need to be rescaled, or else the video encoder needs to be configured to accept that

int main(int argc, char *argv[])
{
    PubMaster pm{ {"headCameraState"} };
    VisionIpcServer vipc_server{ "camerad" };
    V4LCamera camera{ "/dev/video0", CAPTURE_WIDTH, CAPTURE_HEIGHT };
    NVFormatConverter converter{ CAPTURE_WIDTH, CAPTURE_HEIGHT, NVBUF_COLOR_FORMAT_UYVY,
                                 CAMERA_WIDTH, CAMERA_HEIGHT, NVBUF_COLOR_FORMAT_NV12 };
    vipc_server.create_buffers(VISION_STREAM_HEAD_COLOR, CAMERA_BUFFER_COUNT, false, CAMERA_WIDTH, CAMERA_HEIGHT);
    vipc_server.start_listener();

    fmt::print(stderr, "Ready to start streaming\n");

    // Wait for the frame, the dequeue the buffer, process it, and requeue it
    size_t received_count = 0, processed_count = 0;
    auto start = std::chrono::steady_clock::now();

    std::unique_ptr<uint8_t[]> temp_buf = std::make_unique<uint8_t[]>(CAPTURE_WIDTH * CAPTURE_HEIGHT * 2);

    // For determining the color space, we were tracking the yuv ranges
    uint8_t min_y = 255, max_y = 0;
    uint8_t min_u = 255, max_u = 0;
    uint8_t min_v = 255, max_v = 0;

    static_assert(CAPTURE_FPS % CAMERA_FPS == 0, "CAPTURE_FPS must be a multiple of CAMERA_FPS");

    while (!do_exit)
    {
        auto frame = camera.get_frame();
        received_count++;

        uint32_t frame_id = static_cast<uint32_t>(received_count / (CAPTURE_FPS / CAMERA_FPS));

        if (received_count % (CAPTURE_FPS / CAMERA_FPS) == 0)
        {
            auto cur_yuv_buf = vipc_server.get_buffer(VISION_STREAM_HEAD_COLOR);

            if (cur_yuv_buf == nullptr || frame == nullptr) {
                throw std::runtime_error("Failed to get frame buffer");
            }

            auto sof = static_cast<uint64_t>(frame->frame_time.tv_sec * 1'000'000'000ULL + frame->frame_time.tv_usec * 1'000ULL);

            // Send the frame via vision IPC
            VisionIpcBufExtra extra{
                frame_id, 
                sof,
                sof,
            };
            cur_yuv_buf->set_frame_id(frame_id); 

            uint8_t *uyvy_data = frame->planes[0].data;
           
            // Not sure why, but there is a huge slowdown if the data isn't accessed from the device linearly
            // So, for now we just dump the data into a user buffer. We could perhaps let the kernel do this with a different buffer setup
            std::copy(uyvy_data, uyvy_data + CAPTURE_WIDTH * CAPTURE_HEIGHT * 2, temp_buf.get());
            converter.convert(temp_buf.get(), cur_yuv_buf);

            // fmt::print(stderr, "Getting data took {}, conversion took {}\n",  
            //     std::chrono::duration_cast<std::chrono::microseconds>(convertc - startc),
            //     std::chrono::duration_cast<std::chrono::microseconds>(endc - convertc)
            // );

            vipc_server.send(cur_yuv_buf, &extra);

            MessageBuilder msg;
            auto event = msg.initEvent(true);
            auto cdat = event.initHeadCameraState();
        
            cdat.setFrameId(frame_id);
            cdat.setTimestampSof(sof);

            auto words = capnp::messageToFlatArray(msg);
            auto bytes = words.asBytes();
            pm.send("headCameraState", bytes.begin(), bytes.size());


            processed_count++;
            if (frame_id % 100 == 0)
            {
                auto end = std::chrono::steady_clock::now();
                fmt::print("Processed {} frames, average FPS {:02f}\n", processed_count, processed_count / std::chrono::duration<double>(end - start).count());
            }            
        }
    }

    auto end = std::chrono::steady_clock::now();
    fmt::print("Finished processing {} frames, average FPS {:02f}\n", processed_count, processed_count / std::chrono::duration<double>(end - start).count());

    // Print the YUV ranges
    fmt::print("YUV ranges: Y {} {}, U {} {}, V {} {}\n", min_y, max_y, min_u, max_u, min_v, max_v);

    return EXIT_SUCCESS;
}

