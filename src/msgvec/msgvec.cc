#include <vector>
#include <functional>
#include <string_view>
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

bool message_matches(const capnp::DynamicStruct::Reader &msg, const json &obs) {
    std::string path {obs["path"]};
    std::string event_type{};
    auto pos = path.find('.');
    if (pos != std::string::npos) {
        event_type = path.substr(0, pos);
    }

    std::cout << "EVENT TYPE: " << event_type << std::endl;

    return msg.has(event_type);
}

void MsgVec::input(const cereal::Event::Reader &evt) {
    // Cast to a dynamic reader, so we can access the fields by name
    capnp::DynamicStruct::Reader reader = evt;

    // Iterate over each possible msg observation
    for (auto &obs : m_config["obs"]) {
        if (obs["type"] == "msg" && message_matches(reader, obs)) {
            std::cout << "MATCHED" << std::endl;
        }
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