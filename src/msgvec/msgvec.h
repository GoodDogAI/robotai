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

        //, std::function<int(std::vector<float>)> visionIntermediateProvider
        MsgVec(const std::string &jsonConfig);

        // Feeds in messages, will update internal state
        bool input(const std::vector<uint8_t> &bytes);
        bool input(const cereal::Event::Reader &evt);

        size_t obs_size() const;
        size_t act_size() const;

        // Writes out the current observation vector, given the most recent messages
        // Returns true if all observations match their timestamps
        TimeoutResult get_obs_vector(float *obsVector);

        // Returns the current action vector, given the most recent messages
        bool get_act_vector(float *actVector);

        // Given an action vector output from the RL model, returns the list of messages to send
        std::vector<kj::Array<capnp::word>> get_action_command(const float *actVector);


    private:
        json m_config;
        size_t m_obsSize, m_actSize;
        std::vector<float> m_actVector;

        std::map<int, std::deque<float>> m_obsHistory;
        std::map<int, std::deque<uint64_t>> m_obsHistoryTimestamps;
};