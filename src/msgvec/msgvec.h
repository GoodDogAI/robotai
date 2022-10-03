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
        //, std::function<int(std::vector<float>)> visionIntermediateProvider
        MsgVec(const std::string &jsonConfig);

        // Feeds in messages, will update internal state
        bool input(const std::vector<uint8_t> &bytes);
        bool input(const cereal::Event::Reader &evt);

        size_t obs_size() const;
        size_t act_size() const;

        // Returns the current observation vector, given the most recent messages
        bool get_obs_vector(float *obsVector);

        // Given an action vector output from the RL model, returns the list of messages to send
        std::vector<capnp::DynamicStruct::Builder> get_action_command(const float *actVector);


    private:
        json m_config;
        size_t m_obsSize, m_actSize;
        std::vector<float> m_actVector;

        std::map<int, std::deque<float>> m_obsHistory;
};