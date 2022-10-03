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
MsgVec::MsgVec(const std::string &jsonConfig):
 m_config(json::parse(jsonConfig)), m_obsVector(this->obs_size()) {

}

capnp::DynamicValue::Reader get_dotted_value(const capnp::DynamicStruct::Reader &root, std::string dottedPath) {
    capnp::DynamicStruct::Reader value = root;

    while (true) {
        auto dotPos = dottedPath.find('.');
        if (dotPos == std::string::npos) {
            return value.get(dottedPath);
        }

        auto field = dottedPath.substr(0, dotPos);
        dottedPath = dottedPath.substr(dotPos + 1);
        value = value.get(field).as<capnp::DynamicStruct>();
    }

    return value;
}

bool message_matches(const capnp::DynamicStruct::Reader &msg, const json &obs) {
    const std::string &path = obs["path"];
    std::string event_type{};
    auto pos = path.find('.');
    if (pos != std::string::npos) {
        event_type = path.substr(0, pos);
    }

    if (!msg.has(event_type)) {
        return false;
    }
 
    if (!obs.contains("filter")) {
        return true;
    }

    if (obs["filter"].contains("field")) {
        const std::string &filter_field = obs["filter"]["field"];
        const std::string &filter_op = obs["filter"]["op"];
        auto filter_value = get_dotted_value(msg, filter_field);

        if (filter_op == "eq") {
            if (filter_value.getType() == capnp::DynamicValue::Type::ENUM) {
                const std::string &filterEnumStr = obs["filter"]["value"];
                auto enumSchema = filter_value.as<capnp::DynamicEnum>().getSchema();
                auto filterEnumerant = enumSchema.getEnumerantByName(filterEnumStr);
     
                KJ_IF_MAYBE(msgEnumerant, filter_value.as<capnp::DynamicEnum>().getEnumerant()) {
                    return *msgEnumerant == filterEnumerant;
                } else {
                    return false;
                }
            } else if (filter_value.getType() == capnp::DynamicValue::Type::TEXT) {
                return filter_value.as<capnp::Text>() == obs["filter"]["value"];
            } else {
                std::cerr << "Unknown type for field " << filter_field << std::endl;
                return false;
            }
        }
        else {
            throw std::runtime_error("Unknown filter op: " + filter_op);
        }
    }

    return true;
}

float get_vector_value(const capnp::DynamicStruct::Reader &msg, const json &obs) {
}

bool MsgVec::input(const std::vector<uint8_t> &bytes) {
    capnp::FlatArrayMessageReader cmsg(kj::arrayPtr<capnp::word>((capnp::word *)bytes.data(), bytes.size()));
    auto event = cmsg.getRoot<cereal::Event>();
    return this->input(event);
}

bool MsgVec::input(const cereal::Event::Reader &evt) {
    // Cast to a dynamic reader, so we can access the fields by name
    capnp::DynamicStruct::Reader reader = evt;

    // Iterate over each possible msg observation
    size_t index = 0;

    for (auto &obs : m_config["obs"]) {
        if (obs["type"] == "msg" && message_matches(reader, obs)) {
            m_obsVector[index] = get_vector_value(reader);
            return true;
        }

        index++;
    }

    return false;
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