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

#define CAMERA_BUFFER_COUNT 30
#define CAPTURE_WIDTH 1920
#define CAPTURE_HEIGHT 1080

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
            .layout = NVBUF_LAYOUT_BLOCK_LINEAR,
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
            .layout = NVBUF_LAYOUT_BLOCK_LINEAR,
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

            .flag = NVBUFSURF_TRANSFORM_FILTER,

            .flip = NvBufSurfTransform_None,
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

        // fmt::print("inp num_filled: {}, width {}, height {}, pitch{}\n", in_nvbuf_surf->numFilled, 
        //            in_nvbuf_surf->surfaceList->width, in_nvbuf_surf->surfaceList->height, 
        //            in_nvbuf_surf->surfaceList->pitch);
        
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

        // fmt::print("in num planes {}\n", in_nvbuf_surf->surfaceList->planeParams.num_planes);
        // fmt::print("in plane 0 width {} height {} pitch {} bpp {}\n", in_nvbuf_surf->surfaceList->planeParams.width[0], 
        //     in_nvbuf_surf->surfaceList->planeParams.height[0], in_nvbuf_surf->surfaceList->planeParams.pitch[0],
        //     in_nvbuf_surf->surfaceList->planeParams.bytesPerPix[0]);

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

        // fmt::print("out num_filled: {}, width {}, height {}, pitch{}\n", out_nvbuf_surf->numFilled, 
        //     out_nvbuf_surf->surfaceList->width, out_nvbuf_surf->surfaceList->height, 
        //     out_nvbuf_surf->surfaceList->pitch);

        // fmt::print("out num planes {}\n", out_nvbuf_surf->surfaceList->planeParams.num_planes);
        // fmt::print("out plane 0 width {} height {} pitch {} bpp {}\n", out_nvbuf_surf->surfaceList->planeParams.width[0], 
        //     out_nvbuf_surf->surfaceList->planeParams.height[0], out_nvbuf_surf->surfaceList->planeParams.pitch[0],
        //     out_nvbuf_surf->surfaceList->planeParams.bytesPerPix[0]);
        // fmt::print("out plane 1 width {} height {} pitch {} bpp {}\n", out_nvbuf_surf->surfaceList->planeParams.width[1], 
        //     out_nvbuf_surf->surfaceList->planeParams.height[1], out_nvbuf_surf->surfaceList->planeParams.pitch[1],
        //     out_nvbuf_surf->surfaceList->planeParams.bytesPerPix[1]);

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

        // Copy out plane 0 (Y)
        // std::copy((uint8_t*)out_nvbuf_surf->surfaceList->mappedAddr.addr[0],
        //           (uint8_t*)out_nvbuf_surf->surfaceList->mappedAddr.addr[0] + output_params.width * output_params.height * dest_fmt_bytes_per_pixel[0],
        //            out_buf->y);
        ret = NvBufSurface2Raw(out_nvbuf_surf, 0, 0, output_params.width, output_params.height, out_buf->y);
        if (ret)
        {
            throw std::runtime_error("Failed to copy output buffer");
        }

        ret = NvBufSurface2Raw(out_nvbuf_surf, 0, 1, output_params.width / 2, output_params.height / 2, out_buf->uv);
        if (ret)
        {
            throw std::runtime_error("Failed to copy output buffer");
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
// - Use the provided NVIDIA format converter and scaler to take the full 1920x1080 image and scale it down to the CAMERA_WIDTH/HEIGHT
// (otherwise you are losing pixels to cropping)
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
    size_t count = 0;
    auto start = std::chrono::steady_clock::now();

    std::unique_ptr<uint8_t[]> temp_buf = std::make_unique<uint8_t[]>(CAPTURE_WIDTH * CAPTURE_HEIGHT * 2);

    // For determining the color space, we were tracking the yuv ranges
    uint8_t min_y = 255, max_y = 0;
    uint8_t min_u = 255, max_u = 0;
    uint8_t min_v = 255, max_v = 0;

    while (!do_exit)
    {
        auto frame = camera.get_frame();
        count++;

        uint32_t frame_id = static_cast<uint32_t>(count / 3);

        // TODO Calculate the frame skip properly
        if (count % 2 == 0)
        {
            auto cur_yuv_buf = vipc_server.get_buffer(VISION_STREAM_HEAD_COLOR);

            if (cur_yuv_buf == nullptr || frame == nullptr) {
                throw std::runtime_error("Failed to get frame buffer");
            }

            // Send the frame via vision IPC
            VisionIpcBufExtra extra{
                frame_id, 
                static_cast<uint64_t>(frame->frame_time.tv_sec * 1'000'000'000ULL + frame->frame_time.tv_usec * 1'000ULL),
                static_cast<uint64_t>(frame->frame_time.tv_sec * 1'000'000'000ULL + frame->frame_time.tv_usec * 1'000ULL),
            };
            cur_yuv_buf->set_frame_id(frame_id); 

            uint8_t *uyvy_data = frame->planes[0].data;
            //const uint32_t color_frame_stride = CAMERA_WIDTH * 2;

            // Not sure why, but there is a huge slowdown if the data isn't accessed from the device linearly
            // So, for now we just dump the data into a user buffer. We could perhaps let the kernel do this with a different buffer setup

            // REALLY SHULD BE CAPTURE_WIDTH/HEIGHT but it doesn't work?
            std::copy(uyvy_data, uyvy_data + CAPTURE_WIDTH * CAPTURE_HEIGHT * 2, temp_buf.get());

            converter.convert(temp_buf.get(), cur_yuv_buf);
        

            // for (uint32_t i = 0; i < CAMERA_WIDTH * CAMERA_HEIGHT * 2; i += 4) {
            //     min_y = std::min(min_y, temp_buf[i + 1]);
            //     max_y = std::max(max_y, temp_buf[i + 1]);
            //     min_y = std::min(min_y, temp_buf[i + 3]);
            //     max_y = std::max(max_y, temp_buf[i + 3]);

            //     min_u = std::min(min_u, temp_buf[i]);
            //     max_u = std::max(max_u, temp_buf[i]);
            //     min_v = std::min(min_v, temp_buf[i + 2]);
            //     max_v = std::max(max_v, temp_buf[i + 2]);
            // }

            // for(uint32_t row = 0; row < CAMERA_HEIGHT / 2; row++) {
            //     for (uint32_t col = 0; col < CAMERA_WIDTH / 2; col++) {
            //         cur_yuv_buf->y[(row * 2) * cur_yuv_buf->stride + col * 2] = temp_buf[(row * 2) * color_frame_stride + col * 4 + 1];
            //         cur_yuv_buf->y[(row * 2) * cur_yuv_buf->stride + col * 2 + 1] = temp_buf[(row * 2) * color_frame_stride + col * 4 + 3];
            //         cur_yuv_buf->y[(row * 2 + 1) * cur_yuv_buf->stride + col * 2] = temp_buf[(row * 2 + 1) * color_frame_stride + col * 4 + 1];
            //         cur_yuv_buf->y[(row * 2 + 1) * cur_yuv_buf->stride + col * 2 + 1] = temp_buf[(row * 2 + 1) * color_frame_stride + col * 4 + 3];

            //         cur_yuv_buf->uv[row * cur_yuv_buf->stride + col * 2] = (temp_buf[(row * 2) * color_frame_stride + col * 4 + 0] + temp_buf[(row * 2 + 1) * color_frame_stride + col * 4 + 0]) / 2;
            //         cur_yuv_buf->uv[row * cur_yuv_buf->stride + col * 2 + 1] = (temp_buf[(row * 2) * color_frame_stride + col * 4 + 2] + temp_buf[(row * 2 + 1) * color_frame_stride + col * 4 + 2]) / 2;
            //     }
            // }

            vipc_server.send(cur_yuv_buf, &extra);

            if (frame_id % 100 == 0)
            {
                auto end = std::chrono::steady_clock::now();
                fmt::print("Processed {} frames, average FPS {:02f}\n", count, count / std::chrono::duration<double>(end - start).count());
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    fmt::print("Finished processing {} frames, average FPS {:02f}\n", count, count / std::chrono::duration<double>(end - start).count());

    // Print the YUV ranges
    fmt::print("YUV ranges: Y {} {}, U {} {}, V {} {}\n", min_y, max_y, min_u, max_u, min_v, max_v);

    return EXIT_SUCCESS;
}