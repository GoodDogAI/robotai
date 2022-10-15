#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <thread>
#include <chrono>

#include "cereal/messaging/messaging.h"

#include "msgvec.h"
#include "config.h"


TEST_CASE( "Process a simple message", "[msgvec]" ) {
   MsgVec msgvec{R"(
        {
            "obs": [{
                "type": "msg",
                "path": "voltage.volts",
                "index": -1,
                "timeout": 0.01,
                "filter": {
                    "field": "voltage.type",
                    "op": "eq",
                    "value": "mainBattery"
                },
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 13.5],
                    "vec_range": [-1, 1]
                }
            }],
            "act": []
        }
    )", MsgVec::MessageTimingMode::REALTIME};

    REQUIRE(msgvec.obs_size() == 1);

    MessageBuilder vmsg;
    auto vevent = vmsg.initEvent(true);
    auto vdat = vevent.initVoltage();
    vdat.setVolts(13.2);
    vdat.setType(cereal::Voltage::Type::MAIN_BATTERY);

    // auto vwords = capnp::messageToFlatArray(vmsg);
    // auto msgr = capnp::readDataStruct<cereal::Event>(vwords);

    REQUIRE(msgvec.input(vevent).msg_processed == true);
}
