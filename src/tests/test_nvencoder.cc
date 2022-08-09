#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <thread>
#include <chrono>

#include "config.h"
#include "nvencoder.h"
#include "cereal/visionipc/visionbuf.h"

// All ffmpeg library are compiled with C linkage
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
}

enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
  for (const enum AVPixelFormat *p = pix_fmts; *p != -1; p++) {
    std::cout << *p << " ";
  }
  std::cout << std::endl;
  return AV_PIX_FMT_YUV420P;
}

TEST_CASE( "Encoder renders a single frame", "[encoder]" ) {
    NVEncoder encoder { ENCODER_DEV, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT, ENCODER_BITRATE, CAMERA_FPS };
    VisionBuf buf;

    VisionIpcBufExtra extra {
                        1, // frame_id
                        0,
                        0,
        };

    auto size = CAMERA_WIDTH * CAMERA_HEIGHT * 3 / 2;
    auto stride = CAMERA_WIDTH;
    auto uv_offset = CAMERA_WIDTH * CAMERA_HEIGHT;
    buf.allocate(size);
    buf.init_yuv(CAMERA_WIDTH, CAMERA_HEIGHT, stride, uv_offset);

    auto future = encoder.encode_frame(&buf, &extra);
    auto result = future.get();

    INFO("Got result from encoder len " << result->len);
    REQUIRE(result->len > 0);
    REQUIRE(result->flags & V4L2_BUF_FLAG_KEYFRAME);
}

TEST_CASE( "Encoder sends keyframes once per second at least", "[encoder]" ) {
    int test_fps = 5;

    NVEncoder encoder { ENCODER_DEV, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT, ENCODER_BITRATE, test_fps };
    VisionBuf buf;

    auto size = CAMERA_WIDTH * CAMERA_HEIGHT * 3 / 2;
    auto stride = CAMERA_WIDTH;
    auto uv_offset = CAMERA_WIDTH * CAMERA_HEIGHT;
    buf.allocate(size);
    buf.init_yuv(CAMERA_WIDTH, CAMERA_HEIGHT, stride, uv_offset);

    for(uint32_t i = 0; i < test_fps * 3; i++){
        VisionIpcBufExtra extra {
                        i + 1, // frame_id
                        0,
                        0,
        };

        auto future = encoder.encode_frame(&buf, &extra);
        auto result = future.get();
        
        INFO("frame " << i << " " << result->flags);

        if (i % test_fps == 0) {
            CHECK(result->flags & V4L2_BUF_FLAG_KEYFRAME);
        }  
    }
}

TEST_CASE( "FFMPEG Decode of nvencoder frame", "[encoder]") {
    NVEncoder encoder { ENCODER_DEV, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT, ENCODER_BITRATE, CAMERA_FPS };
    VisionBuf buf;
    int ret;

    VisionIpcBufExtra extra {
                    1, // frame_id
                    0,
                    0,
    };

    auto size = CAMERA_WIDTH * CAMERA_HEIGHT * 3 / 2;
    auto stride = CAMERA_WIDTH;
    auto uv_offset = CAMERA_WIDTH * CAMERA_HEIGHT;
    buf.allocate(size);
    buf.init_yuv(CAMERA_WIDTH, CAMERA_HEIGHT, stride, uv_offset);

    // Put a test pattern, top half of image is green, bottom half is half-green
    std::fill(buf.y, buf.y + CAMERA_WIDTH * CAMERA_HEIGHT / 2, 255);
    std::fill(buf.y + CAMERA_WIDTH * CAMERA_HEIGHT / 2, buf.y + CAMERA_WIDTH * CAMERA_HEIGHT, 128);
    std::fill(buf.uv, buf.uv + CAMERA_WIDTH * CAMERA_HEIGHT / 2, 0);

    auto future = encoder.encode_frame(&buf, &extra);
    auto result = future.get();

    // Required in ffmpeg < 4 in order to see all decoders
    av_register_all();

    AVPacket *pkt = av_packet_alloc();
    REQUIRE(pkt);

    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    REQUIRE(codec);

    AVCodecParserContext *parser = av_parser_init(codec->id);
    REQUIRE(parser);

    AVCodecContext *c = avcodec_alloc_context3(codec);
    REQUIRE(c);
    c->get_format = get_hw_format;

    ret = avcodec_open2(c, codec, NULL);
    REQUIRE(ret == 0);

    AVFrame *frame = av_frame_alloc();
    REQUIRE(frame);

    ret = av_new_packet(pkt, result->len);
    REQUIRE(ret == 0);

    std::copy(result->data, result->data + result->len, pkt->data);

    ret = avcodec_send_packet(c, pkt);
    REQUIRE(ret == 0);

    ret = avcodec_receive_frame(c, frame);
    REQUIRE(ret == 0);

    REQUIRE(frame->width == CAMERA_WIDTH);
    REQUIRE(frame->height == CAMERA_HEIGHT);
    REQUIRE(c->pix_fmt == AV_PIX_FMT_YUV420P);

    CHECK(frame->linesize[0] == CAMERA_WIDTH);
    CHECK(frame->linesize[1] == CAMERA_WIDTH / 2);
    CHECK(frame->linesize[2] == CAMERA_WIDTH / 2);

    for (int row = 0; row < CAMERA_HEIGHT; row++) {
        INFO("row " << row);

        if (row < CAMERA_HEIGHT / 2)
            REQUIRE(frame->data[0][row * frame->linesize[0]] == 255);
        
        if (row > CAMERA_HEIGHT / 2)
            REQUIRE(frame->data[0][row * frame->linesize[0]] == 128);

        REQUIRE(frame->data[1][row * frame->linesize[1]] == 0);
        REQUIRE(frame->data[2][row * frame->linesize[2]] == 0);
    }
    
}