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
    SubMaster sm{ {service_name} };

    const auto log_start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    const std::tm tm{*std::gmtime(&log_start)};

    auto log_filename = fmt::format("{}-{}-{}-{}-{}_{}.log", log_name, tm.tm_year + 1900, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min);

    std::ofstream log{ log_path / log_filename, std::ios::binary };

    while (!do_exit) {
        sm.update();

        if (sm.updated(service_name)) {
            auto event = sm[service_name];
            std::cout << event.getHeadEncodeData().getIdx().getEncodeId() << std::endl;
        }

        
    }

    return EXIT_SUCCESS;
}
