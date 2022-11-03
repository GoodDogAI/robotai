#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <cassert>
#include <atomic>

#include <math.h>
#include <fmt/core.h>
#include <fmt/chrono.h>

#include "util.h"
#include "config.h"
#include "serial.h"
#include "cereal/messaging/messaging.h"

const char *service_name = "odriveFeedback";
ExitHandler do_exit;
const auto loop_time = std::chrono::milliseconds(50);

struct ODriveCommandState {
    float currentLeft;
    float currentRight;
    float velocityLeft;
    float velocityRight;
    cereal::ODriveCommand::ControlMode controlMode;
    std::chrono::steady_clock::time_point time;
};

std::atomic<ODriveCommandState> command_state;

static void send_raw_command(Serial &port, const std::string& command) {
    port.write_str(command);
}

static float send_float_command(Serial &port, const std::string& command) {
    const auto float_re = std::regex("([0-9\\.\\-]+|nan)\r\n");
    port.write_str(command);

    auto response = port.read_regex(float_re);
    return std::stof(response);
}

static std::pair<float, float> request_feedback(Serial &port, int motor_index) {
    const auto feedback_re = std::regex("([0-9\\.\\-]+) ([0-9\\.\\-]+)\r\n");

    assert(motor_index == 0 || motor_index == 1);
    port.write_str(fmt::format("f {} \n", motor_index));
    auto result = port.read_regex(feedback_re);
    std::smatch match;
    
    std::regex_match(result, match, feedback_re);

    return std::make_pair(std::stof(match[1].str()), std::stof(match[2].str()));
}


// void twistCallback(const geometry_msgs::Twist::ConstPtr& msg)
// {
//   // ROS_INFO("Linear %f, %f, %f,  Ang %f, %f, %f", 
//   //   msg->linear.x, msg->linear.y, msg->linear.z,
//   //   msg->angular.x, msg->angular.y, msg->angular.z);

//   float ang = msg->angular.z;

//   vel_left = -1.0f * (msg->linear.x - ang); // -1 to flip direction
//   vel_right = (msg->linear.x + ang);

//   if (vel_left > MAX_SPEED)
//     vel_left = MAX_SPEED;
//   if (vel_left < -MAX_SPEED)
//     vel_left = -MAX_SPEED;

//   if (vel_right > MAX_SPEED)
//     vel_right = MAX_SPEED;
//   if (vel_right < -MAX_SPEED)
//     vel_right = -MAX_SPEED;

//   last_received = ros::Time::now();
// }

static void odrive_command_processor() {
  // Note, we use a SubSocket, not a SubMaster, because the brain channel can have multiple messages queued up at once
  // and a SubMaster only keeps one message buffered at a time
  std::unique_ptr<Context> ctx{ Context::create() };
  std::unique_ptr<Poller> poller{ Poller::create() };
  std::unique_ptr<SubSocket> sock{SubSocket::create(ctx.get(), "brainCommands")};

  poller->registerSocket(sock.get());

  while(!do_exit) {
    auto cur_sock = poller->poll(1000);
    if (cur_sock.size() == 0) {
      continue;
    }

    auto msg = std::unique_ptr<Message> {cur_sock[0]->receive(true)};
    auto now = std::chrono::steady_clock::now();

    if (msg) {
        capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word)));
        auto event = cmsg.getRoot<cereal::Event>();

        if (event.which() != cereal::Event::ODRIVE_COMMAND) 
            continue;

        ODriveCommandState new_state;

        new_state.currentLeft = event.getOdriveCommand().getDesiredCurrentLeft();
        new_state.currentRight = event.getOdriveCommand().getDesiredCurrentRight();
        new_state.velocityLeft = event.getOdriveCommand().getDesiredVelocityLeft();
        new_state.velocityRight = event.getOdriveCommand().getDesiredVelocityRight();
        new_state.controlMode = event.getOdriveCommand().getControlMode();
        new_state.time = now;

        command_state = new_state;
    }

  }
}

/**
 * This node provides a simple interface to the ODrive module, it accepts cmd_vel messages to drive the motors,
 and publishes /vbus to report the current battery voltage
 */
int main(int argc, char **argv)
{
    PubMaster pm { {service_name} };
    Serial port { ODRIVE_SERIAL_PORT, ODRIVE_BAUD_RATE };
    bool motors_enabled { false };

    fmt::print("Opened ODrive serial device, starting communication\n");

    // Make the first communication with the ODrive
    for (int attempt = 0; attempt < 5; attempt++) {
        try {
            float result = send_float_command(port, "r vbus_voltage\n");

            if (result > 0) {
                fmt::print("Successfully started odrive communications, vbus = {:0.2f}\n", result);
                break;
            }
        }
        catch(const std::exception& e) {
            if (attempt >= 4) {
                fmt::print(stderr, "Could not communicate with ODrive. Exiting.\n");
                return EXIT_FAILURE;
            }

            fmt::print(stderr, "Error reading from ODrive {}, retrying\n", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
    }

    // Start the command processor thread
    std::thread command_thread {odrive_command_processor};

    // Run the main loop
    while (!do_exit) {
        auto start_loop { std::chrono::steady_clock::now() };
        ODriveCommandState cur_command = command_state;

        if (start_loop - cur_command.time > std::chrono::seconds(1)) {
            if (motors_enabled) {
                command_state = {0, 0, 0, 0, cereal::ODriveCommand::ControlMode::VELOCITY, start_loop};

                fmt::print("Disabling motors after inactivity\n");
                send_raw_command(port, "w axis0.requested_state 1\n");
                send_raw_command(port, "w axis1.requested_state 1\n");
                motors_enabled = false;
            }
        }
        else if (!motors_enabled) {
            fmt::print("Received message, enabling motors\n");

            //Put motors into AXIS_STATE_CLOSED_LOOP_CONTROL
            send_raw_command(port, "w axis0.requested_state 8\n");
            send_raw_command(port, "w axis1.requested_state 8\n");
            motors_enabled = true;
        }

        // Send motor commands
        if (cur_command.controlMode == cereal::ODriveCommand::ControlMode::VELOCITY) {
            send_raw_command(port, fmt::format("v 0 {}\n", cur_command.velocityLeft));
            send_raw_command(port, fmt::format("v 1 {}\n", cur_command.velocityRight));
        }
        else if (cur_command.controlMode == cereal::ODriveCommand::ControlMode::CURRENT) {
            send_raw_command(port, fmt::format("c 0 {}\n", cur_command.currentLeft));
            send_raw_command(port, fmt::format("c 1 {}\n", cur_command.currentRight));
        }
        else {
            throw std::invalid_argument("Invalid control mode");
        }

        // Read and update vbus voltage
        float vbus_voltage = send_float_command(port, "r vbus_voltage\n");
        if (std::isnan(vbus_voltage)) {
            throw std::runtime_error("Got unexpected NaN voltage from Odrive");
        }

        MessageBuilder vmsg;
        auto vevent = vmsg.initEvent(true);
        auto vdat = vevent.initVoltage();
        vdat.setVolts(vbus_voltage);
        vdat.setType(cereal::Voltage::Type::MAIN_BATTERY);
        
        auto vwords = capnp::messageToFlatArray(vmsg);
        auto vbytes = vwords.asBytes();
        pm.send(service_name, vbytes.begin(), vbytes.size());

        // Read and update motor feedback
        MessageBuilder fmsg;
        auto fevent = fmsg.initEvent(true);
        auto fdat = fevent.initOdriveFeedback();
        auto left = fdat.initLeftMotor();
        auto right = fdat.initRightMotor();

        auto left_data = request_feedback(port, 0);
        left.setPos(left_data.first);
        left.setVel(left_data.second);

        auto right_data = request_feedback(port, 1);
        right.setPos(right_data.first);
        right.setVel(right_data.second);

        left.setCurrent(send_float_command(port, "r axis0.motor.current_control.Iq_setpoint\n"));
        right.setCurrent(send_float_command(port, "r axis1.motor.current_control.Iq_setpoint\n"));

        auto fwords = capnp::messageToFlatArray(fmsg);
        auto fbytes = fwords.asBytes();
        pm.send(service_name, fbytes.begin(), fbytes.size());

        //fmt::print("Loop took {}\n", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_loop));

        if (std::chrono::steady_clock::now() - start_loop > loop_time) {
            fmt::print(stderr, "Could not keep up with realtime loop in ODrive\n");
            return EXIT_FAILURE;
        }

        std::this_thread::sleep_until(start_loop + loop_time);
    }

    // Disable motors when we quit the program
    send_raw_command(port, "w axis0.requested_state 1\n");
    send_raw_command(port, "w axis1.requested_state 1\n");

    command_thread.join();

    return EXIT_SUCCESS;
}