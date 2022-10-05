#include <vector>
#include <functional>
#include <string_view>
#include <algorithm>
#include <set>
#include <iostream>

#include <nlohmann/json.hpp>

#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <capnp/schema.h>
#include <capnp/pretty-print.h>
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
            "size": 123,
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
 m_config(json::parse(jsonConfig)) {

    // Fill up each deque with zeros to initialize
    size_t obs_index = 0;
    m_obsSize = 0;

    for (auto &obs: m_config["obs"]) {
        if (obs["type"] == "msg") {
            if (obs["index"].is_number()) {
                if (obs["index"] >= 0) {
                    throw std::runtime_error("msgvec: msg index must be negative");
                }

                m_obsHistory[obs_index] = std::deque<float>(std::abs(obs["index"].get<int64_t>()), 0.0f);
                m_obsSize += 1;
            } 
            else if (obs["index"].is_array()) {
                std::vector<int64_t> indices = obs["index"];

                if (std::any_of(indices.begin(), indices.end(), [](int64_t i) { return i >= 0; })) {
                    throw std::runtime_error("msgvec: msg indexes must be negative");
                }

                m_obsHistory[obs_index] = std::deque<float>(std::abs(*std::min_element(indices.begin(), indices.end())), 0.0f);
                m_obsSize += obs["index"].size();
            }            
            else {
                throw std::runtime_error("msgvec: msg index must be an array or number");
            }
        }
        else if (obs["type"] == "vision") {
            m_obsSize += obs["size"].get<int64_t>();
        }
        else {
            throw std::runtime_error("Unknown observation type");
        }
      
        obs_index++;
    }

    m_actSize = 0;
    // Also verify that all action paths are unique
    std::set<std::string> actPaths;
    for (auto &act: m_config["act"]) {
        if (act["type"] == "msg") {
            m_actSize += 1;
        }
        else {
            throw std::runtime_error("Unknown action type");
        }

        if (actPaths.find(act["path"]) != actPaths.end()) {
            throw std::runtime_error("msgvec: action paths must be unique");
        }

        actPaths.insert(act["path"]);
    }

    m_actVector = std::vector<float>(m_actSize);
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

void set_dotted_value(capnp::DynamicStruct::Builder &root, std::string dottedPath, const capnp::DynamicValue::Reader &value) {
    capnp::DynamicStruct::Builder builder = root;

    while (true) {
        auto dotPos = dottedPath.find('.');
        if (dotPos == std::string::npos) {
            builder.set(dottedPath, value);
            return;
        }

        auto field = dottedPath.substr(0, dotPos);
        dottedPath = dottedPath.substr(dotPos + 1);

        if (!builder.has(field)) {
            builder.init(field);
        }
        builder = builder.get(field).as<capnp::DynamicStruct>();
    }
}

std::string get_event_type(const std::string &dotted_path) {
    auto dotPos = dotted_path.find('.');
    if (dotPos == std::string::npos) {
        return dotted_path;
    }

    return dotted_path.substr(0, dotPos);
}

bool message_matches(const capnp::DynamicStruct::Reader &msg, const json &obs) {
    const std::string &path = obs["path"];
    std::string event_type {get_event_type(path)};

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

float transform_msg_to_vec(const json &transform, float msgValue) {
    const std::string &transformType = transform["type"];

    if (transformType == "identity") {
        return msgValue;
    } else if (transformType == "rescale") {
        const std::vector<float> &msgRange = transform["msg_range"];
        const std::vector<float> &vecRange = transform["vec_range"];

        return std::clamp((msgValue - msgRange[0]) / (msgRange[1] - msgRange[0]) * (vecRange[1] - vecRange[0]) + vecRange[0], vecRange[0], vecRange[1]);
    } else {
        throw std::runtime_error("Unknown transform type: " + transformType);
    }
}

float transform_vec_to_msg(const json &transform, float vecValue) {
    const std::string &transformType = transform["type"];

    if (transformType == "identity") {
        return vecValue;
    } else if (transformType == "rescale") {
        const std::vector<float> &msgRange = transform["msg_range"];
        const std::vector<float> &vecRange = transform["vec_range"];

        return std::clamp((vecValue - vecRange[0]) / (vecRange[1] - vecRange[0]) * (msgRange[1] - msgRange[0]) + msgRange[0], msgRange[0], msgRange[1]);
    } else {
        throw std::runtime_error("Unknown transform type: " + transformType);
    }
}

float get_vector_value(const capnp::DynamicStruct::Reader &msg, const json &obs) {
    float rawValue = get_dotted_value(msg, obs["path"]).as<float>();

    if (obs.contains("transform")) {
        return transform_msg_to_vec(obs["transform"], rawValue);
    }

    return rawValue;
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
    bool processed = false;
    size_t obs_index = 0;

    for (auto &obs : m_config["obs"]) {
        if (obs["type"] == "msg" && message_matches(reader, obs)) {
            float rawValue = get_dotted_value(reader, obs["path"]).as<float>();
            m_obsHistory[obs_index].push_front(transform_msg_to_vec(obs["transform"], rawValue));
            m_obsHistory[obs_index].pop_back();
            processed = true;
        }

        obs_index++;
    }
    
    size_t act_index = 0;

    for (auto &act : m_config["act"]) {
        if (act["type"] == "msg") {
            float rawValue = get_dotted_value(reader, act["path"]).as<float>();
            m_actVector[act_index] = transform_vec_to_msg(act["transform"], rawValue);
            processed = true;
            act_index++;
        }
    }

    return processed;
}

size_t MsgVec::obs_size() const {
    return m_obsSize;
}

size_t MsgVec::act_size() const {
    return m_actSize;
}

bool MsgVec::get_obs_vector(float *obsVector) {
    size_t index = 0;
    size_t curpos = 0;

    for (auto &obs : m_config["obs"]) {
        if (obs["type"] == "msg") {
            if (obs["index"].is_number()) {
                obsVector[index] = m_obsHistory[index][std::abs(obs["index"].get<int64_t>()) - 1];
                curpos++;
            } else if (obs["index"].is_array()) {
                std::vector<int64_t> indices = obs["index"];
                for (size_t i = 0; i < indices.size(); i++) {
                    obsVector[index + i] = m_obsHistory[index][std::abs(indices[i]) - 1];
                }
                curpos += indices.size();
            }
        } else if (obs["type"] == "vision") {
        }

        index++;
    }

    return true;
}

bool MsgVec::get_act_vector(float *actVector) {
    std::copy(m_actVector.begin(), m_actVector.end(), actVector);
    return true;
}

std::vector<kj::Array<capnp::word>> MsgVec::get_action_command(const float *actVector) {
    std::map<std::string, MessageBuilder> msgs;

    size_t act_index = 0;

    for (auto &act : m_config["act"]) {
        std::string event_type {get_event_type(act["path"])};
        float actValue = actVector[act_index];

        if (msgs.count(event_type) == 0) {
            msgs[event_type].initEvent(true);
        }

        capnp::DynamicStruct::Builder dyn = msgs[event_type].getRoot<cereal::Event>();

        if (act.contains("transform")) {
            actValue = transform_vec_to_msg(act["transform"], actValue);
        }

        set_dotted_value(dyn, act["path"], actValue);

        act_index++;
    }

    std::vector<kj::Array<capnp::word>> ret;
    for (auto& [event_type, msg] : msgs) {
        //std::cout << capnp::prettyPrint(msg.).flatten().cStr() << std::endl;
        ret.push_back(std::move(capnp::messageToFlatArray(dynamic_cast<capnp::MessageBuilder&>(msg))));
    }
    return ret;
    
}
