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
            "index": -1,  # -1 means last frame, -4 means last 4 frames, or specify [-1, -5, -20] for example
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

static void verify_transform(const json& transform) {
    if (transform["type"] == "identity") {
        return;
    } else if (transform["type"] == "rescale") {
        if (transform["msg_range"].size() != 2) {
            throw std::runtime_error("msg_range must be a 2-element array");
        }

        if (transform["vec_range"].size() != 2) {
            throw std::runtime_error("vec_range must be a 2-element array");
        }

        if (transform["msg_range"][0] >= transform["msg_range"][1]) {
            throw std::runtime_error("msg_range must be increasing");
        }

        if (transform["vec_range"][0] >= transform["vec_range"][1]) {
            throw std::runtime_error("vec_range must be increasing");
        }
    } else {
        throw std::runtime_error("Unknown transform type");
    }
}

static std::pair<uint32_t, std::vector<int64_t>> get_queue_obs_len(const json &obs) {
    uint32_t queue_size;
    std::vector<int64_t> indices;

    if (obs["index"].is_number()) {
        if (obs["index"].get<int64_t>() >= 0) {
            throw std::runtime_error("msgvec: msg indexes must be negative");
        }

        queue_size = std::abs(obs["index"].get<int64_t>());
        indices = std::vector<int64_t>(queue_size);
        std::generate(indices.begin(), indices.end(), [n = -1]() mutable { return n--; });
    }
    else if (obs["index"].is_array()) {
        indices = obs["index"].get<std::vector<int64_t>>();

        if (std::any_of(indices.begin(), indices.end(), [](int64_t i) { return i >= 0; })) {
            throw std::runtime_error("msgvec: msg indexes must be negative");
        }

        queue_size = std::abs(*std::min_element(indices.begin(), indices.end()));
    }
    else {
        throw std::runtime_error("msgvec: msg index must be an array or number");
    }

    return std::pair(queue_size, std::move(indices));
}

MsgVec::MsgVec(const std::string &jsonConfig, const MessageTimingMode timingMode):
 m_config(json::parse(jsonConfig)), m_timingMode(timingMode), m_lastMsgLogMonoTime(0), m_obsSize(0), m_actSize(0), m_visionSize(0) {

    if (!m_config.contains("obs") || !m_config.contains("act")) {
        throw std::runtime_error("Must have obs and act sections");
    }

    // Fill up each deque with zeros to initialize
    size_t obs_index = 0;
    m_obsSize = 0;

    for (auto &obs: m_config["obs"]) {
        if (obs["type"] == "msg") {
            const auto [queue_size, indices] = get_queue_obs_len(obs);

            m_obsHistory[obs_index] = std::deque<float>(queue_size, 0.0f);
            m_obsHistoryTimestamps[obs_index] = std::deque<uint64_t>(queue_size, 0);
            m_obsSize += indices.size();
          
            verify_transform(obs["transform"]);
        }
        else if (obs["type"] == "vision") {
            if (!obs.contains("size")) {
                throw std::runtime_error("msgvec: vision section must have predetermined size");
            }
            m_visionSize = obs["size"].get<int64_t>();

            if (m_visionSize < 1) {
                throw std::runtime_error("msgvec: vision size must be > 1");
            }

            const auto [queue_size, obs_size] = get_queue_obs_len(obs);

            m_visionHistory = std::deque<std::vector<float>>(queue_size, std::vector<float>(m_visionSize, 0.0f));
            m_visionHistoryIds = std::deque<uint32_t>(queue_size, 0);

            m_obsSize += m_visionSize * obs_size.size();
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

            verify_transform(act["transform"]);
        }
        else {
            throw std::runtime_error("Unknown action type");
        }

        if (actPaths.find(act["path"]) != actPaths.end()) {
            throw std::runtime_error("msgvec: action paths must be unique");
        }

        actPaths.insert(act["path"]);
    }

    m_actVector = std::vector<float>(m_actSize, 0.0f);
    m_actVectorReady = std::vector<bool>(m_actSize, false);
}

static capnp::DynamicValue::Reader get_dotted_value(const capnp::DynamicStruct::Reader &root, std::string dottedPath) {
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

static void set_dotted_value(capnp::DynamicStruct::Builder &root, std::string dottedPath, const capnp::DynamicValue::Reader &value) {
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

static std::string get_event_type(const std::string &dotted_path) {
    auto dotPos = dotted_path.find('.');
    if (dotPos == std::string::npos) {
        return dotted_path;
    }

    return dotted_path.substr(0, dotPos);
}

static bool message_matches(const capnp::DynamicStruct::Reader &msg, const json &obs) {
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

static float transform_msg_to_vec(const json &transform, float msgValue) {
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

static float transform_vec_to_msg(const json &transform, float vecValue) {
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


MsgVec::InputResult MsgVec::input(const std::vector<uint8_t> &bytes) {
    capnp::FlatArrayMessageReader cmsg(kj::arrayPtr<capnp::word>((capnp::word *)bytes.data(), bytes.size() / sizeof(capnp::word)));
    auto event = cmsg.getRoot<cereal::Event>();
    return this->input(event);
}

MsgVec::InputResult MsgVec::input(const cereal::Event::Reader &evt) {
    // Cast to a dynamic reader, so we can access the fields by name
    capnp::DynamicStruct::Reader reader = evt;

    // Iterate over each possible msg observation
    bool allActionsInitiallyReady = std::all_of(m_actVectorReady.begin(), m_actVectorReady.end(), [](bool b) { return b; });
    bool processed = false;
    size_t obs_index = 0;

    for (auto &obs : m_config["obs"]) {
        if (obs["type"] == "msg" && message_matches(reader, obs)) {
            float rawValue = get_dotted_value(reader, obs["path"]).as<float>();
            m_obsHistory[obs_index].push_front(transform_msg_to_vec(obs["transform"], rawValue));
            m_obsHistory[obs_index].pop_back();

            m_obsHistoryTimestamps[obs_index].push_front(evt.getLogMonoTime());
            m_obsHistoryTimestamps[obs_index].pop_back();

            processed = true;
        }

        obs_index++;
    }
    
    // Do the same for the actions
    size_t act_index = 0;

    for (auto &act : m_config["act"]) {
        if (act["type"] == "msg" && message_matches(reader, act)) {
            float rawValue = get_dotted_value(reader, act["path"]).as<float>();
            m_actVector[act_index] = transform_msg_to_vec(act["transform"], rawValue);
            m_actVectorReady[act_index] = true;
            processed = true;
        }

        act_index++;
    }

    bool allActionsEndedReady = std::all_of(m_actVectorReady.begin(), m_actVectorReady.end(), [](bool b) { return b; });

    // Handle the appcontrol messages as a special case, which can override actions
    if (evt.which() == cereal::Event::APP_CONTROL) {
        m_lastAppControlMsg.setRoot(reader.as<cereal::Event>());
        processed = true;
    }

    m_lastMsgLogMonoTime = evt.getLogMonoTime();

    return { .msg_processed = processed, .act_ready = !allActionsInitiallyReady && allActionsEndedReady };
}

void MsgVec::input_vision(const float *visionIntermediate, uint32_t frameId) {
    m_visionHistory.push_front(std::vector<float>(visionIntermediate, visionIntermediate + m_visionSize));
    m_visionHistory.pop_back();

    m_visionHistoryIds.push_front(frameId);
    m_visionHistoryIds.pop_back();
}

size_t MsgVec::obs_size() const {
    return m_obsSize;
}

size_t MsgVec::act_size() const {
    return m_actSize;
}

size_t MsgVec::vision_size() const {
    return m_visionSize;
}

uint64_t MsgVec::_get_msgvec_log_mono_time() {
    if (m_timingMode == MessageTimingMode::REPLAY) {
        return m_lastMsgLogMonoTime;
    } else {
        struct timespec t;
        clock_gettime(CLOCK_BOOTTIME, &t);
        uint64_t current_time = t.tv_sec * 1000000000ULL + t.tv_nsec;
        return current_time;
    }
}

MsgVec::TimeoutResult MsgVec::get_obs_vector(float *obsVector) {
    TimeoutResult timestamps_valid = TimeoutResult::MESSAGES_ALL_READY;
    const uint64_t cur_time = _get_msgvec_log_mono_time();
    size_t index = 0;
    size_t curpos = 0;

    for (auto &obs : m_config["obs"]) {
        if (obs["type"] == "msg") {
            const auto [queue_size, indices] = get_queue_obs_len(obs);

            for (size_t i = 0; i < indices.size(); i++) {
                auto history_index = std::abs(indices[i]) - 1;
                obsVector[curpos + i] = m_obsHistory[index][history_index];
            
                if (cur_time - m_obsHistoryTimestamps[index][history_index] > (history_index + 1) * obs["timeout"].get<float>() * 1e9) {
                    if (i == 0) {
                        timestamps_valid = TimeoutResult::MESSAGES_NOT_READY;
                    } else if (timestamps_valid == TimeoutResult::MESSAGES_ALL_READY) {
                        timestamps_valid = TimeoutResult::MESSAGES_PARTIALLY_READY;
                    }
                }
            }

            curpos += indices.size();
        } else if (obs["type"] == "vision") {
            const auto [queue_size, indices] = get_queue_obs_len(obs);

            for (size_t i = 0; i < indices.size(); i++) {
                auto history_index = std::abs(indices[i]) - 1;
                std::copy(m_visionHistory[history_index].begin(), m_visionHistory[history_index].end(), &obsVector[curpos]);
                curpos += m_visionSize;
            }
        }

        index++;
    }

    // Reset the action vector ready messages
    m_actVectorReady = std::vector<bool>(m_actSize, false);

    return timestamps_valid;
}

bool MsgVec::get_act_vector(float *actVector) {
    if (m_timingMode == MessageTimingMode::REALTIME) {
        throw std::logic_error("Cannot get action vector in realtime mode");
    }

    std::copy(m_actVector.begin(), m_actVector.end(), actVector);
    return std::all_of(m_actVectorReady.begin(), m_actVectorReady.end(), [](bool b) { return b; });
}

bool MsgVec::get_reward(float *reward) {
    auto appCtrl = m_lastAppControlMsg.getRoot<cereal::Event>().asReader();

    *reward = kj::nan();

    if (!appCtrl.hasAppControl() || appCtrl.getAppControl().getConnectionState() != cereal::AppControl::ConnectionState::CONNECTED || !m_config.contains("rew"))
    {
        return false;
    }

    if (appCtrl.getAppControl().getRewardState() == cereal::AppControl::RewardState::OVERRIDE_POSITIVE &&
        _get_msgvec_log_mono_time() <= appCtrl.getLogMonoTime() + m_config["rew"]["override"]["positive_reward_timeout"].get<float>() * 1e9) {
        *reward = m_config["rew"]["override"]["positive_reward"].get<float>();
        return true;
    }
    else if (appCtrl.getAppControl().getRewardState() == cereal::AppControl::RewardState::OVERRIDE_NEGATIVE &&
        _get_msgvec_log_mono_time() <= appCtrl.getLogMonoTime() + m_config["rew"]["override"]["negative_reward_timeout"].get<float>() * 1e9) {
        *reward = m_config["rew"]["override"]["negative_reward"].get<float>();
        return true;
    }

    return false;
}

std::vector<kj::Array<capnp::word>> MsgVec::_get_appcontrol_overrides() {
    std::vector<kj::Array<capnp::word>> overrides;

    auto appCtrl = m_lastAppControlMsg.getRoot<cereal::Event>().asReader();

    float linearX = appCtrl.getAppControl().getLinearXOverride();
    float angularZ = appCtrl.getAppControl().getAngularZOverride();

    if (appCtrl.getAppControl().getMotionState() == cereal::AppControl::MotionState::STOP_ALL_OUTPUTS) {
        linearX = 0.0;
        angularZ = 0.0;
    }

    MessageBuilder odriveMsg;
    auto odriveevt = odriveMsg.initEvent();
    auto odrive = odriveevt.initOdriveCommand();
     // -1 to flip direction
    float cmd_left = -1.0f * (linearX - angularZ);
    float cmd_right = (linearX + angularZ);
    odrive.setCurrentLeft(cmd_left);
    odrive.setCurrentRight(cmd_right);
    overrides.push_back(capnp::messageToFlatArray(odriveMsg));

    MessageBuilder headMsg;
    auto headevt = headMsg.initEvent();
    auto head = headevt.initHeadCommand();
    head.setPitchAngle(0.0);
    head.setYawAngle(std::clamp(-100.0f * angularZ, -30.0f, 30.0f));
    overrides.push_back(capnp::messageToFlatArray(headMsg));

    return overrides;
}

std::vector<kj::Array<capnp::word>> MsgVec::get_action_command(const float *actVector) {
    std::map<std::string, MessageBuilder> msgs;
    size_t act_index = 0;

    if (m_timingMode == MessageTimingMode::REPLAY) {
        throw std::logic_error("Cannot get action commands in replay mode");
    }

    auto appCtrl = m_lastAppControlMsg.getRoot<cereal::Event>().asReader();

    if (appCtrl.hasAppControl() && appCtrl.getAppControl().getConnectionState() == cereal::AppControl::ConnectionState::CONNECTED &&
        m_config.contains("appcontrol") && 
        _get_msgvec_log_mono_time() - appCtrl.getLogMonoTime() < m_config["appcontrol"]["timeout"].get<float>() * 1e9
        && appCtrl.getAppControl().getMotionState() != cereal::AppControl::MotionState::NO_OVERRIDE) {
       return _get_appcontrol_overrides();
    }

    for (auto &act : m_config["act"]) {
        std::string event_type {get_event_type(act["path"])};
        float actValue = actVector[act_index];

        KJ_ASSERT(kj::isNaN(actValue) == false, "NaN in actVector");
        KJ_ASSERT(actValue < kj::inf() && actValue > -kj::inf(), "Inf in actVector");

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
