#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <linux/videodev2.h>
#include <fmt/core.h>

#include "config.h"
#include "util.h"

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"

ExitHandler do_exit;

namespace fs = std::experimental::filesystem;


const char *log_name = "alphalog";

const char *service_name = "headEncodeData";
const fs::path log_path{ "/media/card" };


fs::path get_log_filename() {
    const auto log_start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    const std::tm tm{*std::gmtime(&log_start)};

    return log_path/fmt::format("{}-{}-{}-{}-{}_{}.log", log_name, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);
}

int main(int argc, char *argv[])
{
    bool started_logging { false };
    size_t last_10_sec_bytes { 0 };
    size_t last_10_sec_msgs { 0 };
    auto last_10_sec_time { std::chrono::steady_clock::now() };

    std::unique_ptr<Context> ctx{ Context::create() };
    std::unique_ptr<Poller> poller{ Poller::create() };

    std::unique_ptr<SubSocket> encoder_sock{ SubSocket::create(ctx.get(), service_name) };

    auto log_start { std::chrono::steady_clock::now() };
    bool need_to_rotate { false };
    static_assert(LOG_DURATION_SECONDS >= 60, "Logs need to be at least 1 minute long");

    auto log_filename { get_log_filename().concat(".active") };
    fmt::print("Opening log {}\n", log_filename.string());

    std::ofstream log{ log_filename, std::ios::binary };

    poller->registerSocket(encoder_sock.get());

    while (!do_exit) {
        Message *msg = nullptr;
        for (auto sock : poller->poll(1000)) {
            msg = sock->receive(true);

            capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize()));
            auto event = cmsg.getRoot<cereal::Event>();

            if (event.getHeadEncodeData().getIdx().getFlags() & V4L2_BUF_FLAG_KEYFRAME) {
                if (!started_logging) {
                    fmt::print("Received first Iframe, starting to log\n");
                    started_logging = true;
                }
                else if (need_to_rotate) {
                    log.close();
                    auto log_final_filename { log_filename };
                    log_final_filename.replace_extension("");
                    fs::rename(log_filename, log_final_filename);

                    auto new_log_filename { get_log_filename().concat(".active") };
                    assert(new_log_filename != log_filename);

                    log.open(log_filename, std::ios::binary);
                    log_filename = new_log_filename;

                    fmt::print("Rotating logs to {}", log_filename.string());
                    log_start = std::chrono::steady_clock::now();
                    need_to_rotate = false;
                }
            }

            if (started_logging) {
                log.write(msg->getData(), msg->getSize());
                log.flush();
            }

            last_10_sec_bytes += msg->getSize();
            last_10_sec_msgs++;
           //std::cout << "Wrote event " << event.getHeadEncodeData().getIdx().getEncodeId() << std::endl;
        }        

        const auto cur_time = std::chrono::steady_clock::now();
        if (cur_time - last_10_sec_time > std::chrono::seconds(10)) {
            fmt::print("loggerd {} msgs/sec @ {:1.2f} Mb/sec\n", last_10_sec_msgs / 10, last_10_sec_bytes / (10.0f * 1'000'000));
            last_10_sec_bytes = last_10_sec_msgs = 0;
            last_10_sec_time = cur_time;
        }

        if (cur_time - log_start > std::chrono::seconds(LOG_DURATION_SECONDS)) {
            need_to_rotate = true;
        }
    }

    return EXIT_SUCCESS;
}
