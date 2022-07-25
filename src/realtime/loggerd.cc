#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <fmt/core.h>

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


int main(int argc, char *argv[])
{
    std::unique_ptr<Context> ctx{ Context::create() };
    std::unique_ptr<Poller> poller{ Poller::create() };

    std::unique_ptr<SubSocket> encoder_sock{ SubSocket::create(ctx.get(), service_name) };

    const auto log_start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    const std::tm tm{*std::gmtime(&log_start)};

    auto log_filename = fmt::format("{}-{}-{}-{}-{}_{}.log", log_name, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);

    std::cout << "Opening log " << log_filename << std::endl;

    std::ofstream log{ log_path / log_filename, std::ios::binary };

    poller->registerSocket(encoder_sock.get());

    while (!do_exit) {
        Message *msg = nullptr;
        for (auto sock : poller->poll(1000)) {
            msg = sock->receive(true);

            log.write(msg->getData(), msg->getSize());
            log.flush();

            //capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize()));
            //auto event = cmsg.getRoot<cereal::Event>();
           //std::cout << "Wrote event " << event.getHeadEncodeData().getIdx().getEncodeId() << std::endl;

        }        
    }

    return EXIT_SUCCESS;
}
