#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include <functional>
#include <unordered_set>
#include <experimental/filesystem>
#include <linux/videodev2.h>
#include <fmt/core.h>

#include "config.h"
#include "util.h"

#include "cereal/services.h"
#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"

ExitHandler do_exit;

namespace fs = std::experimental::filesystem;

const char *log_name { "alphalog" };
const fs::path log_path{ LOG_PATH };


static fs::path get_log_filename() {
    const auto log_start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    const std::tm tm{*std::gmtime(&log_start)};

    return log_path/fmt::format("{}-{}-{}-{}-{}_{}.log", log_name, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);
}

static void close_message(Message *m) {
    m->close();
}



int main(int argc, char *argv[])
{
    bool started_logging { false };
    size_t last_10_sec_bytes { 0 };
    size_t last_10_sec_msgs { 0 };
    auto last_10_sec_time { std::chrono::steady_clock::now() };

    std::unique_ptr<Context> ctx{ Context::create() };
    std::unique_ptr<Poller> poller{ Poller::create() };

    auto log_start { std::chrono::steady_clock::now() };
    bool need_to_rotate { false };
    static_assert(LOG_DURATION_SECONDS >= 60, "Logs need to be at least 1 minute long");

    auto log_filename { get_log_filename().concat(".active") };
    fmt::print("Opening log {}\n", log_filename.string());

    std::ofstream log{ log_filename, std::ios::binary };

    // Register all sockets
    std::unordered_set<std::unique_ptr<SubSocket>> socks;

    for (const auto& it : services) {
        if (!it.should_log) 
            continue;

        fmt::print("logging {} (on port {})\n", it.name, it.port);

        auto sock = std::unique_ptr<SubSocket> { SubSocket::create(ctx.get(), it.name) };
        assert(sock != NULL);

        poller->registerSocket(sock.get());
        socks.insert(std::move(sock));
    }

    while (!do_exit) {
        for (auto sock : poller->poll(1000)) {
            auto msg = std::unique_ptr<Message, std::function<void(Message*)>>(sock->receive(true), close_message);

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
                    fmt::print("Renaming {} to {}\n", log_filename.string(), log_final_filename.string());
                    fs::rename(log_filename, log_final_filename);

                    auto new_log_filename { get_log_filename().concat(".active") };
                    assert(new_log_filename != log_filename);
                    log_filename = new_log_filename;

                    log.open(log_filename, std::ios::binary);
                    
                    fmt::print("Rotating logs to {}\n", log_filename.string());
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
