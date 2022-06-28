#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <experimental/filesystem>

#include "util.h"

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"

ExitHandler do_exit;

namespace fs = std::experimental::filesystem;

const char *service_name = "headEncodeData";
const fs::path log_path{ "/media/card" };

int main(int argc, char *argv[])
{
    SubMaster sm{ {service_name} };

    

    while (!do_exit) {
        sm.update();

        if (sm.updated(service_name)) {
            auto event = sm[service_name];
            std::cout << event.getHeadEncodeData().getIdx().getEncodeId() << std::endl;
        }

        
    }

    return EXIT_SUCCESS;
}
