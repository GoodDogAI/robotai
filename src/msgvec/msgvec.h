#pragma once

#include <string>
#include <functional>
#include <map>
#include <unordered_set>
#include <vector>
#include <deque>
#include <nlohmann/json.hpp>

#include <capnp/dynamic.h>

#include "cereal/messaging/messaging.h"

using json = nlohmann::json;

class MsgVec {
    public:
        enum class MessageTimingMode {
            // You are reading and writing messages live, ex. in braind, and so you measure time using the system clock
            REALTIME,

            // You are replaying messages from a log, and so you measure time using the message timestamps
            REPLAY,
        };

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

        MsgVec(const std::string &jsonConfig, const MessageTimingMode timingMode);

        // Feeds in messages, will update internal state
        InputResult input(const std::vector<uint8_t> &bytes);
        InputResult input(const cereal::Event::Reader &evt);
        InputResult input(const capnp::DynamicStruct::Reader &evt);

        // Feeds in current vision frame, for tracking intermediate states
        void input_vision(const float *visionIntermediate, const uint32_t frameId);

        size_t obs_size() const;
        size_t act_size() const;
        size_t vision_size() const;

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

        // Prints debug timing information
        void _debug_print_timing();

    private:
        json m_config;
        MessageTimingMode m_timingMode;
        uint64_t m_lastMsgLogMonoTime;
        size_t m_obsSize, m_actSize;
        size_t m_discreteActUpdates;
        std::vector<float> m_actVector;
        std::vector<bool> m_actVectorReady;
        std::vector<float> m_relativeActValues;

        std::map<int, std::deque<float>> m_obsHistory;
        std::map<int, std::deque<uint64_t>> m_obsHistoryTimestamps;
        
        size_t m_visionSize;
        std::deque<std::vector<float>> m_visionHistory;
        std::deque<uint32_t> m_visionHistoryIds;
        std::deque<uint64_t> m_visionHistoryTimestamps;

        // Needs to be a MallocMessageBuilder so that it keeps its own copy of the message
        capnp::MallocMessageBuilder m_lastAppControlMsg;
        uint64_t m_lastRewardOverrideMonoTime;
        float m_lastRewardOverride;

        // Returns action vector output given the last app control message
        std::vector<kj::Array<capnp::word>> _get_appcontrol_overrides();
        uint64_t _get_msgvec_log_mono_time();
        bool _input_acts(const capnp::DynamicStruct::Reader &reader);        
};
