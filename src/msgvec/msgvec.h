#include <string>
#include <functional>
#include "cereal/messaging/messaging.h"

//using json = nlohmann::json;

class MsgVec {
    public:
        //, std::function<int(std::vector<float>)> visionIntermediateProvider
        MsgVec(const std::string &jsonConfig);

        // Feeds in messages, will update internal state
        void input(const cereal::Event::Reader &evt);

        size_t obsSize() const;
        size_t actSize() const;

        // Returns the current observation vector, given the most recent messages
        bool getObsVector(float *obsVector);

        // Given an action vector output from the RL model, returns the list of messages to send
        std::vector<const cereal::Event::Reader> getActionCommands(const std::vector<float> &act);


    private:
        std::string m_config;
};