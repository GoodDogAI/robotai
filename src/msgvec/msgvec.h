#include <string>
#include <functional>
#include "cereal/messaging/messaging.h"


class MsgVec {
    public:
    MsgVec(std::string jsonConfig, std::function<int, std::vector<float>> visionIntermediateProvider);

    // Feeds in messages, will update internal state
    void input(const cereal::Event::Reader &evt);

    size_t obsSize() const;
    size_t actSize() const;

    // Returns the current observation vector, given the most recent messages
    std::vector<float> getObsVector();

    // Given an action vector output from the RL model, returns the list of messages to send
    std::vector<const cereal::Event::Reader &evt> getActionCommands(const std::vector<float> &act);


    private:
    std::string m_jsonConfig;
};