#include <string>
#include <functional>
#include <nlohmann/json.hpp>

#include "cereal/messaging/messaging.h"

using json = nlohmann::json;

class MsgVec {
    public:
        //, std::function<int(std::vector<float>)> visionIntermediateProvider
        MsgVec(const std::string &jsonConfig);

        // Feeds in messages, will update internal state
        void input(const std::vector<uint8_t> &bytes);
        void input(const cereal::Event::Reader &evt);

        size_t obs_size() const;
        size_t act_size() const;

        // Returns the current observation vector, given the most recent messages
        bool get_obs_vector(float *obsVector);

        // Given an action vector output from the RL model, returns the list of messages to send
        std::vector<const cereal::Event::Reader> get_action_command(const std::vector<float> &act);


    private:
        json m_config;
        std::vector<float> m_obsVector;
        std::vector<float> m_actVector;

        
};