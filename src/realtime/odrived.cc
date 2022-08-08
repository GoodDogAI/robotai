#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <cassert>

#include <math.h>
#include <fmt/core.h>
#include <fmt/chrono.h>

#include "config.h"
#include "serial.h"
#include "cereal/messaging/messaging.h"

const char *service_name = "odriveFeedback";
const auto loop_time = std::chrono::milliseconds(100);

// static volatile float MAX_SPEED;

// static volatile bool motors_enabled;
// static volatile float vel_left, vel_right;
// ros::Time last_received;

// std::string read_string(int fd) {
//   std::string response = "";
//   char buf;
//   int num_read;

//   while(num_read = read(fd, &buf, 1)) {
//     // Make sure to read full lines of \r\n
//     if (buf == '\r')
//       continue;

//     if (isspace(buf))
//       break;

//     response += buf;
//   }

//   //Useful for debugging
//   //std::cout << "read_string: '" << response << "'" << std::endl;
//   return response;
// }

// void send_raw_command(int fd, const std::string& command) {
//   int write_res = write(fd, command.c_str(), command.length());

//   if (write_res != command.length()) {
//     ROS_ERROR("Error sending float command");
//   }
// }

// int send_int_command(int fd, const std::string& command) {
//   int write_res = write(fd, command.c_str(), command.length());

//   if (write_res != command.length()) {
//     ROS_ERROR("Error sending int command");
//     return -1;
//   }

//   std::string response = read_string(fd);
//   return std::stoi(response);
// }

static void send_raw_command(Serial &port, const std::string& command) {
    port.write_str(command);
}

static float send_float_command(Serial &port, const std::string& command) {
    const auto float_re = std::regex("([0-9\\.\\-]+)\r\n");
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



/**
 * This node provides a simple interface to the ODrive module, it accepts cmd_vel messages to drive the motors,
 and publishes /vbus to report the current battery voltage
 */
int main(int argc, char **argv)
{
    PubMaster pm { {service_name} };
    Serial port { ODRIVE_SERIAL_PORT, ODRIVE_BAUD_RATE };
    auto last_received { std::chrono::steady_clock::now() };
    bool motors_enabled { false };
    float vel_left { 0.0 }, vel_right { 0.0 };


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

    for (;;) {
        auto start_loop { std::chrono::steady_clock::now() };

        if (start_loop - last_received > std::chrono::seconds(1)) {
            if (motors_enabled) {
                vel_left = vel_right = 0;

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
        send_raw_command(port, fmt::format("v 0 {}\n", vel_left));
        send_raw_command(port, fmt::format("v 1 {}\n", vel_right));

        // Read and update vbus voltage
        float vbus_voltage = send_float_command(port, "r vbus_voltage\n");
        fmt::print("Vbus = {}\n", vbus_voltage);
        assert(!std::isnan(vbus_voltage));

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

//   while (ros::ok())
//   {
//     ros::Time start = ros::Time::now();

//     // If you haven't received a message in the last second, then stop the motors
//     if (ros::Time::now() - last_received > ros::Duration(1)) {
//       if (motors_enabled) {
//         ROS_WARN("Didn't receive a message for the past second, shutting down motors");
//         vel_left = vel_right = 0;

//         send_raw_command(serial_port, "w axis0.requested_state 1\n");
//         send_raw_command(serial_port, "w axis1.requested_state 1\n");
//         motors_enabled = false;
//       }
//     } 
//     else {
//       if (!motors_enabled) {
//         ROS_INFO("Received message, enabling motors");

//         //Put motors into AXIS_STATE_CLOSED_LOOP_CONTROL
//         send_raw_command(serial_port, "w axis0.requested_state 8\n");
//         send_raw_command(serial_port, "w axis1.requested_state 8\n");
//         motors_enabled = true;
//       }
//     }


//     //ROS_INFO("Sending motor vels %f %f", vel_left, vel_right);
//     std::string cmd;
//     cmd = "v 0 " + std::to_string(vel_left) + "\n";
//     send_raw_command(serial_port, cmd.c_str());

//     cmd = "v 1 " + std::to_string(vel_right) + "\n";
//     send_raw_command(serial_port, cmd.c_str());


//     // Read and publish the vbus main voltage
//     float vbus_voltage = send_float_command(serial_port, "r vbus_voltage\n");

//     if (!std::isnan(vbus_voltage)) {
//       std_msgs::Float32 vbus_msg;
//       vbus_msg.data = vbus_voltage;
//       vbus_pub.publish(vbus_msg);
//     }

//     // Read and publish the motor feedback values
//     bumble::ODriveFeedback feedback_msg;
//     send_raw_command(serial_port, "f 0\n"); 
//     feedback_msg.motor_pos_actual_0 = std::stof(read_string(serial_port));
//     feedback_msg.motor_vel_actual_0 = std::stof(read_string(serial_port));
//     feedback_msg.motor_vel_cmd_0 = vel_left;

//     send_raw_command(serial_port, "f 1\n");
//     feedback_msg.motor_pos_actual_1 = std::stof(read_string(serial_port));
//     feedback_msg.motor_vel_actual_1 = std::stof(read_string(serial_port));
//     feedback_msg.motor_vel_cmd_1 = vel_right;

//     feedback_msg.motor_current_actual_0 = send_float_command(serial_port, "r axis0.motor.current_control.Iq_setpoint\n");
//     feedback_msg.motor_current_actual_1 = send_float_command(serial_port, "r axis1.motor.current_control.Iq_setpoint\n");

//     feedback_msg.header.stamp = ros::Time::now();
//     feedback_pub.publish(feedback_msg);

//     //std::cout << "Took " << ros::Time::now() - start << std::endl;     

//     ros::spinOnce();
//     loop_rate.sleep();
//   }

    // Disable motors when we quit the program
    send_raw_command(port, "w axis0.requested_state 1\n");
    send_raw_command(port, "w axis1.requested_state 1\n");

    return EXIT_SUCCESS;
}