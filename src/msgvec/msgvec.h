#pragma once

#include <string>
#include <functional>
#include <map>
#include <vector>
#include <deque>
#include <nlohmann/json.hpp>

#include <capnp/dynamic.h>

#include "cereal/messaging/messaging.h"

using json = nlohmann::json;

class MsgVec {
    public:
        enum class TimeoutResult {
            // Vector is not populated at all
            MESSAGES_NOT_READY,

            // Got at least one message per obs entry, but not yet to fill entire history buffer
            // It should be good to use this data already, older buffer entries will be zero
            MESSAGES_PARTIALLY_READY,

            // All buffers full
            MESSAGES_ALL_READY,
        };

        struct InputResult {
            bool msg_processed;
            bool act_ready;
        };

        //, std::function<int(std::vector<float>)> visionIntermediateProvider
        MsgVec(const std::string &jsonConfig);

        // Feeds in messages, will update internal state
        InputResult input(const std::vector<uint8_t> &bytes);
        InputResult input(const cereal::Event::Reader &evt);

        size_t obs_size() const;
        size_t act_size() const;

        // Writes out the current observation vector, given the most recent messages
        // Returns true if all observations match their timestamps
        TimeoutResult get_obs_vector(float *obsVector);

        // Write out the current action vector, given the most recent messages
        // Returns true if all actions have been populated since the last call to `get_obs_vector`
        bool get_act_vector(float *actVector);

        // Write out the current reward, if it has been overridden by appcontrol messages
        // Returns true if this is the case
        bool get_reward(float *reward);

        // Given an action vector output from the RL model, returns the list of messages to send
        std::vector<kj::Array<capnp::word>> get_action_command(const float *actVector);

        

    private:
        json m_config;
        size_t m_obsSize, m_actSize;
        std::vector<float> m_actVector;
        std::vector<bool> m_actVectorReady;

        std::map<int, std::deque<float>> m_obsHistory;
        std::map<int, std::deque<uint64_t>> m_obsHistoryTimestamps;

        // Needs to be a MallocMessageBuilder so that it keeps its own copy of the message
        capnp::MallocMessageBuilder m_lastAppControlMsg;

        // Returns action vector output given the last app control message
        std::vector<kj::Array<capnp::word>> _get_appcontrol_overrides();
};
