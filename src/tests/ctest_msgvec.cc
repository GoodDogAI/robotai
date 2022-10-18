#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <iostream>
#include <thread>
#include <chrono>

#include "cereal/messaging/messaging.h"

#include "msgvec.h"
#include "config.h"


TEST_CASE( "Process a simple message realtime", "[msgvec]" ) {
   MsgVec msgvec{R"(
        {
            "obs": [{
                "type": "msg",
                "path": "voltage.volts",
                "index": -1,
                "timeout": 0.01,
                "filter": {
                    "field": "voltage.type",
                    "op": "eq",
                    "value": "mainBattery"
                },
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 13.5],
                    "vec_range": [-1, 1]
                }
            }],
            "act": []
        }
    )", GENERATE(MsgVec::MessageTimingMode::REALTIME, MsgVec::MessageTimingMode::REPLAY)};

    REQUIRE(msgvec.obs_size() == 1);

    MessageBuilder vmsg;
    auto vevent = vmsg.initEvent(true);
    auto vdat = vevent.initVoltage();
    vdat.setVolts(13.2);
    vdat.setType(cereal::Voltage::Type::MAIN_BATTERY);

    REQUIRE(msgvec.input(vevent).msg_processed == true);
}

TEST_CASE( "Process a more realistic configuration", "[msgvec]") {
    MsgVec msgvec{R"(
    {
        "obs": [
            { 
                "type": "msg",
                "path": "odriveFeedback.leftMotor.vel",
                "index": -10,
                "timeout": 0.125,
                "transform": {
                    "type": "identity"
                }
            },

            {
                "type": "msg",
                "path": "voltage.volts",
                "index": [-1, -5, -50],
                "timeout": 0.125,
                "filter": {
                    "field": "voltage.type",
                    "op": "eq",
                    "value": "mainBattery"
                },
                "transform": {
                    "type": "rescale",
                    "msg_range": [0, 15],
                    "vec_range": [-1, 1]
                }
            },
            
            { 
                "type": "msg",
                "path": "headFeedback.pitchAngle",
                "index": -1,
                "timeout": 0.125,
                "transform": {
                    "type": "rescale",
                    "msg_range": [-45.0, 45.0],
                    "vec_range": [-1, 1]
                }
            },

            {
                "type": "vision",
                "size": 17003,
                "index": -1
            }
        ],

        "act": [
            {
                "type": "msg",
                "path": "odriveCommand.desiredVelocityLeft",
                "timeout": 0.125,
                "transform": {
                    "type": "rescale",
                    "msg_range": [-2, 2],
                    "vec_range": [-1, 1]
                }
            },

            {
                "type": "msg",
                "path": "odriveCommand.desiredVelocityRight",
                "timeout": 0.125,
                "transform": {
                    "type": "rescale",
                    "msg_range": [-2, 2],
                    "vec_range": [-1, 1]
                }
            },

            { 
                "type": "msg",
                "path": "headCommand.pitchAngle",
                "index": -1,
                "timeout": 0.125,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-45.0, 45.0]
                }
            },

            { 
                "type": "msg",
                "path": "headCommand.yawAngle",
                "index": -1,
                "timeout": 0.125,
                "transform": {
                    "type": "rescale",
                    "vec_range": [-1, 1],
                    "msg_range": [-45.0, 45.0]
                }
            }
        ],

        "rew": {
            "base": "reward",

            "override": {
                "positive_reward": 1.0,
                "positive_reward_timeout": 0.0667,

                "negative_reward": -1.0,
                "negative_reward_timeout": 0.0667
            }
        },

        "appcontrol": {
            "mode": "steering_override_v1",
            "timeout": 0.300
        },

        "done": {
            "mode": "on_reward_override"
        }
    }
    )", MsgVec::MessageTimingMode::REALTIME};

    REQUIRE(msgvec.obs_size() == 14 + 17003);
    REQUIRE(msgvec.act_size() == 4);
    REQUIRE(msgvec.vision_size() == 17003);

    for (int i = 0; i < 500; i++) {
        MessageBuilder vmsg;
        auto vevent = vmsg.initEvent(true);
        auto vdat = vevent.initVoltage();
        vdat.setVolts(13.2);
        vdat.setType(cereal::Voltage::Type::MAIN_BATTERY);
        REQUIRE(msgvec.input(vevent).msg_processed == true);

        MessageBuilder omsg;
        auto oevent = omsg.initEvent(true);
        auto odat = oevent.initOdriveFeedback();
        odat.initLeftMotor().setVel(1.0);
        REQUIRE(msgvec.input(oevent).msg_processed == true);

        MessageBuilder hmsg;
        auto hevent = hmsg.initEvent(true);
        auto hdat = hevent.initHeadFeedback();
        hdat.setPitchAngle(2.0);
        REQUIRE(msgvec.input(hevent).msg_processed == true);

        std::vector<float> vision = std::vector<float>(msgvec.vision_size(), 0.0);
        msgvec.input_vision(vision.data(), i);

        float reward;
        REQUIRE(msgvec.get_reward(&reward) == false);

        std::vector<float> obs = std::vector<float>(msgvec.obs_size(), 0.0);

        if (i < 49) {
            REQUIRE(msgvec.get_obs_vector(obs.data()) == MsgVec::TimeoutResult::MESSAGES_PARTIALLY_READY);
        }
        else {
            REQUIRE(msgvec.get_obs_vector(obs.data()) == MsgVec::TimeoutResult::MESSAGES_ALL_READY);
        }

        const float act[] = {0.0, 0.0, 0.0, 0.0};
        auto messages = msgvec.get_action_command(act);

        for (auto &msg : messages) {
            cereal::Event::Reader event = capnp::FlatArrayMessageReader(msg.asPtr()).getRoot<cereal::Event>();
            REQUIRE((event.which() == cereal::Event::ODRIVE_COMMAND || event.which() == cereal::Event::HEAD_COMMAND));
        }
    }

}