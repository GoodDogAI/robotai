#include <vector>
#include <functional>
#include <iostream>

#include <nlohmann/json.hpp>

#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <capnp/schema.h>
#include <capnp/dynamic.h>

#include "msgvec.h"


/*
 "msgvec": {
    "obs": [
        {
            "type": "msg",
            "path": "voltage.volts",
            "index": -1,
            "timeout": 0.01,
            "filter": {
                "field": "type",
                "op": "eq",
                "value": "mainBattery",
            },
            "transform": {
                "type": "rescale",
                "msg_range": [0, 13.5],
                "vec_range": [-1, 1],
            }
        },

        {
            "type": "vision",
            "index": -1,
        }
    ],

    "act": [
        {
            "type": "msg",
            "path": "odriveCommand.leftMotor.vel",
            "timeout": 0.01,
            "transform": {
                "type": "identity",
            },
        },

        { 
            "type": "msg",
            "path": "headCommand.pitchAngle",
            "index": -1,
            "timeout": 0.01,
            "transform": {
                "type": "rescale",
                "vec_range": [-1, 1],
                "msg_range": [-45.0, 45.0],
            },
        },
    ],
}
*/
MsgVec::MsgVec(const std::string &jsonConfig): m_config(json::parse(jsonConfig)), m_obsVector(this->obs_size()) {

}


void MsgVec::input(const std::vector<uint8_t> &bytes) {
    capnp::FlatArrayMessageReader cmsg(kj::arrayPtr<capnp::word>((capnp::word *)bytes.data(), bytes.size()));
    auto event = cmsg.getRoot<cereal::Event>();
    this->input(event);

    std::cout << "INPUT WHICH: " << event.which() << std::endl;
    //fmt::print("INPUT WHICH: {}\n", (uint16_t)event.which());
}

void MsgVec::input(const cereal::Event::Reader &evt) {
    // Cast to a dynamic reader
    capnp::DynamicStruct::Reader reader = evt;

    if (reader.has("voltage")) {
        std::cout << "has voltage" << std::endl;
    }
    else {
        std::cout << "no voltage" << std::endl;
    }
}

size_t MsgVec::obs_size() const {
    size_t size = 0;

    for (auto &obs : m_config["obs"]) {
        if (obs["type"] == "msg") {
            size += 1;
        } else if (obs["type"] == "vision") {
            size += 1;
        }
    }

    return size;
}