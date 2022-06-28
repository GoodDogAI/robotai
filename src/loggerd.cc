#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "util.h"

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"

ExitHandler do_exit;

const char *service_name = "headEncodeData";

int main(int argc, char *argv[])
{
    SubMaster sm({service_name});

    while (!do_exit) {
        sm.update();

        if (sm.updated(service_name)) {
            auto event = sm[service_name];
            std::cout << event.getHeadEncodeData().getIdx().getEncodeId() << std::endl;
        }

        
    }

    return EXIT_SUCCESS;
}
