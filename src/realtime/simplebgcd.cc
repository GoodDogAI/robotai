#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>

#include <fmt/core.h>
#include <fmt/chrono.h>

#include "config.h"
#include "serial.h"
#include "simplebgc.h"
#include "cereal/messaging/messaging.h"

const char *service_name = "simpleBGC";
const auto loop_time = std::chrono::milliseconds(100);

// ros::Time bgc_last_received;
// ros::Time control_last_sent;
// ros::Time ros_last_received;
// ros::Time last_sound;



enum YawGyroState {
  GYRO_INIT,
  GYRO_WAIT_FOR_REBOOT,
  GYRO_WAIT_FOR_CENTER,
  GYRO_OPERATING,
};

int8_t yaw_gyro_state = GYRO_INIT;
//bumble::HeadCommand last_head_cmd;



void crc16_update(uint16_t length, uint8_t *data, uint8_t crc[2]) {
  uint16_t counter;
  uint16_t polynom = 0x8005;
  uint16_t crc_register = (uint16_t)crc[0] | ((uint16_t)crc[1] << 8);
  uint8_t shift_register;
  uint8_t data_bit, crc_bit;
  for (counter = 0; counter < length; counter++) {
    for (shift_register = 0x01; shift_register > 0x00; shift_register <<= 1) {
      data_bit = (data[counter] & shift_register) ? 1 : 0;
      crc_bit = crc_register >> 15;
      crc_register <<= 1;

      if (data_bit != crc_bit) crc_register ^= polynom;
    }
  }

  crc[0] = crc_register;
  crc[1] = (crc_register >> 8);
}

void crc16_calculate(uint16_t length, uint8_t *data, uint8_t crc[2]) {
  crc[0] = 0; crc[1] = 0;
  crc16_update(length, data, crc);
}

void send_message(Serial &port, uint8_t cmd, uint8_t *payload, uint16_t payload_size) {
  bgc_msg *cmd_msg = (bgc_msg *)malloc(sizeof(bgc_msg) + payload_size);
  cmd_msg->command_id = cmd;
  cmd_msg->payload_size = payload_size;
  cmd_msg->header_checksum = cmd_msg->command_id + cmd_msg->payload_size;

  memcpy(cmd_msg->payload, payload, payload_size);
  
  uint8_t crc[2];
  crc16_calculate(sizeof(bgc_msg) + payload_size, (uint8_t *)cmd_msg, crc);

  port.write_byte(BGC_V2_START_BYTE);
  port.write_bytes(cmd_msg, sizeof(bgc_msg) + payload_size);
  port.write_bytes(crc, 2);
}

int16_t degree_to_int16(float angle) {
   float result = round(DEG_TO_INT16(angle));

   if (result <= -32768)
     return -32768;
   else if (result >= 32767)
     return 32767;
   else
     return result;
}

// void build_control_msg(const bumble::HeadCommand &head_cmd, bgc_control_data *control_data) {
//     memset(control_data, 0, sizeof(control_data));

//     control_data->control_mode_roll = CONTROL_MODE_IGNORE;
//     control_data->control_mode_pitch = CONTROL_MODE_ANGLE_REL_FRAME;
//     control_data->control_mode_yaw = CONTROL_MODE_ANGLE_REL_FRAME;
//     control_data->angle_pitch = degree_to_int16(head_cmd.cmd_angle_pitch);
//     control_data->angle_yaw = degree_to_int16(head_cmd.cmd_angle_yaw);

//     control_data->speed_pitch = round(200.0f / CONTROL_SPEED_DEG_PER_SEC_PER_UNIT); 
//     control_data->speed_yaw = round(200.0f / CONTROL_SPEED_DEG_PER_SEC_PER_UNIT); 
// }

// void head_cmd_callback(const bumble::HeadCommand& msg)
// {
//   // Send a control command immediately to set the new position
//   if (yaw_gyro_state == GYRO_OPERATING) {
//     bgc_control_data control_data;
//     build_control_msg(msg, &control_data);

//     send_message(serial_port, CMD_CONTROL, (uint8_t *)&control_data, sizeof(control_data));
//     control_last_sent = ros::Time::now();
//   }

//   // ROS_INFO("Received head cmd %f %d, %f %d", 
//   //   msg->cmd_angle_pitch, control_data.angle_pitch,
//   //   msg->cmd_angle_yaw, control_data.angle_yaw);
  
//   last_head_cmd = msg;
//   ros_last_received = ros::Time::now();
// }

// void sound_cmd_callback(const bumble::SoundCommand& msg)
// {
//   if (ros::Time::now() - last_sound < ros::Duration(2.0)) {
//     return;
//   }

//   bgc_beep_custom_sound beep_data;
//   memset(&beep_data, 0, sizeof(bgc_beep_custom_sound));

//   beep_data.mode = BEEPER_MODE_CUSTOM_MELODY;
//   beep_data.note_length = 20;
//   beep_data.decay_factor = 10;

//   if (msg.sound == bumble::SoundCommand::PUNISHED) {
//     beep_data.notes[0] = 2000;
//     beep_data.notes[1] = 1000;
//   }
//   else if (msg.sound == bumble::SoundCommand::REWARDED) {
//     beep_data.notes[0] = 1000;
//     beep_data.notes[1] = 2000;
//   }
//   else {
//     beep_data.notes[0] = 800;
//     beep_data.notes[1] = 800;
//   }

//   send_message(serial_port, CMD_BEEP_SOUND, (uint8_t *)&beep_data, sizeof(bgc_beep_custom_sound));

//   last_sound = ros::Time::now();
//   ros_last_received = ros::Time::now();
// }



int main(int argc, char **argv)
{
    PubMaster pm { {service_name} };
    Serial port { SIMPLEBGC_SERIAL_PORT, SIMPLEBGC_BAUD_RATE };
    auto gyro_center_start_time { std::chrono::steady_clock::now() };
    
    // Reset the module, so you have a clean connection to it each time
    bgc_reset reset_cmd;
    memset(&reset_cmd, 0, sizeof(bgc_reset));
    send_message(port, CMD_RESET, (uint8_t *)&reset_cmd, sizeof(reset_cmd));

    fmt::print("Waiting for SimpleBGC to Reboot\n");
    std::this_thread::sleep_for(std::chrono::seconds(8));

    fmt::print("Sending command to start realtime data stream for SimpleBGC\n");

    // Register a realtime data stream syncing up with the loop rate
    bgc_data_stream_interval stream_data;
    memset(&stream_data, 0, sizeof(bgc_data_stream_interval));
    stream_data.cmd_id = CMD_REALTIME_DATA_4;
    stream_data.interval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(loop_time).count();
    stream_data.sync_to_data = 0;

    send_message(port, CMD_DATA_STREAM_INTERVAL, (uint8_t *)&stream_data, sizeof(stream_data));

    // bgc_last_received = ros::Time::now();
    // gyro_center_start_time = ros::Time::now();
    // yaw_gyro_state = GYRO_WAIT_FOR_CENTER;

    // yaw_gyro_state = GYRO_WAIT_FOR_REBOOT;

    uint8_t bgc_state = BGC_WAITING_FOR_START_BYTE;
    uint8_t bgc_payload_counter = 0;
    uint8_t bgc_payload_crc[2];
    uint8_t bgc_rx_buffer[BGC_RX_BUFFER_SIZE];
    bgc_msg *const bgc_rx_msg = (bgc_msg *)bgc_rx_buffer;

    for (;;)
    {
      uint8_t data = port.read_byte();

      //fmt::print("Read {:0x}\n", data);

      if (bgc_state == BGC_WAITING_FOR_START_BYTE && data == simplebgc_start_byte)
      {
        bgc_state = BGC_READ_COMMAND_ID;
      }
      else if (bgc_state == BGC_READ_COMMAND_ID)
      {
        bgc_rx_msg->command_id = data;
        bgc_state = BGC_READ_PAYLOAD_SIZE;
      }
      else if (bgc_state == BGC_READ_PAYLOAD_SIZE)
      {
        bgc_rx_msg->payload_size = data;
        bgc_state = BGC_READ_HEADER_CHECKSUM;
      }
      else if (bgc_state == BGC_READ_HEADER_CHECKSUM)
      {
        bgc_rx_msg->header_checksum = data;

        if (bgc_rx_msg->header_checksum != bgc_rx_msg->command_id + bgc_rx_msg->payload_size)
        {
          fmt::print(stderr, "Header checksum failed\n");
          bgc_state = BGC_WAITING_FOR_START_BYTE;
        }
        else
        {
          bgc_state = BGC_READ_PAYLOAD;
          bgc_payload_counter = 0;
        }
      }
      else if (bgc_state == BGC_READ_PAYLOAD)
      {
        bgc_rx_msg->payload[bgc_payload_counter] = data;
        bgc_payload_counter++;

        if (bgc_payload_counter == bgc_rx_msg->payload_size)
        {
          bgc_state = BGC_READ_CRC_0;
        }
      }
      else if (bgc_state == BGC_READ_CRC_0)
      {
        bgc_payload_crc[0] = data;
        bgc_state = BGC_READ_CRC_1;
      }
      else if (bgc_state == BGC_READ_CRC_1)
      {
        bgc_payload_crc[1] = data;

        uint8_t crc[2];
        crc16_calculate(sizeof(bgc_msg) + bgc_rx_msg->payload_size, bgc_rx_buffer, crc);

        if (crc[0] != bgc_payload_crc[0] || crc[1] != bgc_payload_crc[1])
        {
          fmt::print(stderr, "Payload checksum failed\n");
        }
        else
        {
          fmt::print("Recieved valid message of type {}\n", bgc_rx_msg->command_id);
          // bgc_last_received = ros::Time::now();

          if (bgc_rx_msg->command_id == CMD_REALTIME_DATA_4)
          {
            bgc_realtime_data_4 *realtime_data = (bgc_realtime_data_4 *)bgc_rx_msg->payload;

            if (realtime_data->system_error)
            {
              fmt::print(stderr, "BGC Error {:02x}\n", realtime_data->system_error);
              fmt::print(stderr, "Shutting down BGC\n");
              return EXIT_FAILURE;
            }

            //  ROS_INFO("Pitch %0.4f %0.4f %0.4f",
            //       INT16_TO_DEG(realtime_data->imu_angle_pitch),
            //       INT16_TO_DEG(realtime_data->target_angle_pitch),
            //       INT16_TO_DEG(realtime_data->stator_angle_pitch));
            //  ROS_INFO("Yaw %0.4f %0.4f %0.4f",
            //     INT16_TO_DEG(realtime_data->imu_angle_yaw),
            //     INT16_TO_DEG(realtime_data->target_angle_yaw),
            //     INT16_TO_DEG(realtime_data->stator_angle_yaw));

            if (yaw_gyro_state == GYRO_WAIT_FOR_CENTER)
            {
              if (std::abs(INT16_TO_DEG(realtime_data->stator_angle_yaw)) < 1.0)
              {
                yaw_gyro_state = GYRO_OPERATING;
                fmt::print("YAW Gyro centered, angle {0.2f}", INT16_TO_DEG(realtime_data->stator_angle_yaw));
              }
              else
              {
                if (std::chrono::steady_clock::now() - gyro_center_start_time > std::chrono::seconds(5))
                {
                  fmt::print(stderr, "YAW Gyro failed to center, resetting BGC\n");
                  return EXIT_FAILURE;
                }

                fmt::print("Waiting for YAW angle to center, please leave robot still, angle {}\n", INT16_TO_DEG(realtime_data->stator_angle_yaw));
              }
            }

            // Publish a feedback message with the data
            // bumble::HeadFeedback feedback_msg;
            // feedback_msg.cur_angle_pitch = INT16_TO_DEG(realtime_data->stator_angle_pitch);
            // feedback_msg.cur_angle_yaw = INT16_TO_DEG(realtime_data->stator_angle_yaw);
            // feedback_msg.motor_power_pitch = realtime_data->motor_power_pitch / 255.0f;
            // feedback_msg.motor_power_yaw = realtime_data->motor_power_yaw / 255.0f;
            // feedback_msg.header.stamp = ros::Time::now();
            // feedback_pub.publish(feedback_msg);
          }
          else if (bgc_rx_msg->command_id == CMD_GET_ANGLES_EXT)
          {
            bgc_angles_ext *angles_ext = (bgc_angles_ext *)bgc_rx_msg->payload;
            fmt::print("Yaw {:0.4f} {:0.4} {0.4f}\n",
                      INT16_TO_DEG(angles_ext->imu_angle_yaw),
                      INT16_TO_DEG(angles_ext->target_angle_yaw),
                      INT16_TO_DEG(angles_ext->stator_angle_yaw));

            fmt::print("Pitch {:0.4f} {:0.4} {0.4f}\n",
                      INT16_TO_DEG(angles_ext->imu_angle_pitch),
                      INT16_TO_DEG(angles_ext->target_angle_pitch),
                      INT16_TO_DEG(angles_ext->stator_angle_pitch));
          }
          else if (bgc_rx_msg->command_id == CMD_ERROR)
          {
            fmt::print(stderr, "Received CMD_ERROR from BGC\n");
          }
          else if (bgc_rx_msg->command_id == CMD_CONFIRM)
          {
            // No need to do anything
          }
          else
          {
            fmt::print(stderr, "Received unknown message of type {}", bgc_rx_msg->command_id);
          }
        }

        bgc_state = BGC_WAITING_FOR_START_BYTE;
      }
    }

    // pollfd serial_port_poll = {serial_port, POLLIN, 0};

    // while (ros::ok())
    // {
    //   if (yaw_gyro_state == GYRO_WAIT_FOR_REBOOT) {
    //     if (ros::Time::now() - bgc_last_received > ros::Duration(8.0)) {
    //       ROS_INFO("Sending command to start realtime data stream for SimpleBGC");

    //       // Register a realtime data stream syncing up with the loop rate
    //       bgc_data_stream_interval stream_data;
    //       memset(&stream_data, 0, sizeof(bgc_data_stream_interval));
    //       stream_data.cmd_id = CMD_REALTIME_DATA_4;
    //       stream_data.interval_ms = loop_rate.expectedCycleTime().toSec() * 1000;
    //       stream_data.sync_to_data = 0;

    //       send_message(serial_port, CMD_DATA_STREAM_INTERVAL, (uint8_t *)&stream_data, sizeof(stream_data));

    //       bgc_last_received = ros::Time::now();
    //       gyro_center_start_time = ros::Time::now();
    //       yaw_gyro_state = GYRO_WAIT_FOR_CENTER;
    //     }
    //   }
    //   else {
    //     // Exit with an error if you haven't received a message in a while
    //     if (ros::Time::now() - bgc_last_received > ros::Duration(1.0)) {
    //       ROS_ERROR("No messages received in 1 seconds, shutting down BGC subsystem");
    //       return 1;
    //     }
    //   }

    //   // Make sure that if the GYRO is operating, that we are sending a control command at a minimum frequency
    //   if (yaw_gyro_state == GYRO_OPERATING) {
    //     if (ros::Time::now() - control_last_sent > ros::Duration(0.05)) {
    //       bgc_control_data control_data;
    //       build_control_msg(last_head_cmd, &control_data);

    //       send_message(serial_port, CMD_CONTROL, (uint8_t *)&control_data, sizeof(control_data));
    //       control_last_sent = ros::Time::now();
    //     }
    //   }

    //   int ret = poll(&serial_port_poll, 1, 5);
      
    //   if (serial_port_poll.revents & POLLIN) {
    //     uint8_t buf[1024];
    //     ssize_t bytes_read = read(serial_port, buf, sizeof(buf));

    //     if (bytes_read < 0) {
    //       ROS_ERROR("Error %i from read: %s\n", errno, strerror(errno));
    //       return errno;
    //     }

    //     for (ssize_t i = 0; i < bytes_read; i++) {
    //       if (bgc_state == BGC_WAITING_FOR_START_BYTE && buf[i] == simplebgc_start_byte) {
    //         bgc_state = BGC_READ_COMMAND_ID;
    //       }
    //       else if (bgc_state == BGC_READ_COMMAND_ID) {
    //         bgc_rx_msg->command_id = buf[i];
    //         bgc_state = BGC_READ_PAYLOAD_SIZE;
    //       }
    //       else if (bgc_state == BGC_READ_PAYLOAD_SIZE) {
    //         bgc_rx_msg->payload_size = buf[i];
    //         bgc_state = BGC_READ_HEADER_CHECKSUM;
    //       }
    //       else if (bgc_state == BGC_READ_HEADER_CHECKSUM) {
    //         bgc_rx_msg->header_checksum = buf[i];

    //         if (bgc_rx_msg->header_checksum != bgc_rx_msg->command_id + bgc_rx_msg->payload_size) {
    //           ROS_ERROR("Header checksum failed");
    //           bgc_state = BGC_WAITING_FOR_START_BYTE;
    //         }
    //         else {
    //           bgc_state = BGC_READ_PAYLOAD;
    //           bgc_payload_counter = 0;
    //         }
    //       }
    //       else if (bgc_state == BGC_READ_PAYLOAD) {
    //         bgc_rx_msg->payload[bgc_payload_counter] = buf[i];
    //         bgc_payload_counter++;

    //         if (bgc_payload_counter == bgc_rx_msg->payload_size) {
    //           bgc_state = BGC_READ_CRC_0;
    //         }
    //       }
    //       else if (bgc_state == BGC_READ_CRC_0) {
    //         bgc_payload_crc[0] = buf[i];
    //         bgc_state = BGC_READ_CRC_1;
    //       }
    //       else if (bgc_state == BGC_READ_CRC_1) {
    //         bgc_payload_crc[1] = buf[i];

    //         uint8_t crc[2];
    //         crc16_calculate(sizeof(bgc_msg) + bgc_rx_msg->payload_size, bgc_rx_buffer, crc);

    //         if (crc[0] != bgc_payload_crc[0] || crc[1] != bgc_payload_crc[1]) {
    //           ROS_ERROR("Payload checksum failed");
    //         }
    //         else {
    //           //ROS_INFO("Recieved valid message of type %d", bgc_rx_msg->command_id);
    //           bgc_last_received = ros::Time::now();

    //           if (bgc_rx_msg->command_id == CMD_REALTIME_DATA_4) {
    //             bgc_realtime_data_4 *realtime_data = (bgc_realtime_data_4 *)bgc_rx_msg->payload;

    //             if (realtime_data->system_error) {
    //               ROS_ERROR("BGC Error %02x", realtime_data->system_error);
    //               ROS_ERROR("Shutting down BGC");
    //               return realtime_data->system_error;
    //             }

    //           //  ROS_INFO("Pitch %0.4f %0.4f %0.4f", 
    //           //       INT16_TO_DEG(realtime_data->imu_angle_pitch),
    //           //       INT16_TO_DEG(realtime_data->target_angle_pitch),
    //           //       INT16_TO_DEG(realtime_data->stator_angle_pitch));
    //             //  ROS_INFO("Yaw %0.4f %0.4f %0.4f", 
    //             //     INT16_TO_DEG(realtime_data->imu_angle_yaw),
    //             //     INT16_TO_DEG(realtime_data->target_angle_yaw),
    //             //     INT16_TO_DEG(realtime_data->stator_angle_yaw));

    //             if (yaw_gyro_state == GYRO_WAIT_FOR_CENTER) {
    //               if (abs(INT16_TO_DEG(realtime_data->stator_angle_yaw)) < 1.0) {
    //                 yaw_gyro_state = GYRO_OPERATING;
    //                 ROS_INFO("YAW Gyro centered, angle %f", INT16_TO_DEG(realtime_data->stator_angle_yaw));
    //               }
    //               else {
    //                 if (ros::Time::now() - gyro_center_start_time > ros::Duration(5.0)) {
    //                   ROS_ERROR("YAW Gyro failed to center, resetting BGC");
    //                   return 1;
    //                 } 

    //                 ROS_WARN("Waiting for YAW angle to center, please leave robot still, angle %f", INT16_TO_DEG(realtime_data->stator_angle_yaw));
    //               }
    //             } 
            
    //             // Publish a feedback message with the data
    //             bumble::HeadFeedback feedback_msg;
    //             feedback_msg.cur_angle_pitch = INT16_TO_DEG(realtime_data->stator_angle_pitch);
    //             feedback_msg.cur_angle_yaw = INT16_TO_DEG(realtime_data->stator_angle_yaw);
    //             feedback_msg.motor_power_pitch = realtime_data->motor_power_pitch / 255.0f;
    //             feedback_msg.motor_power_yaw = realtime_data->motor_power_yaw / 255.0f;
    //             feedback_msg.header.stamp = ros::Time::now();
    //             feedback_pub.publish(feedback_msg);
    //           }
    //           else if (bgc_rx_msg->command_id == CMD_GET_ANGLES_EXT) {
    //             bgc_angles_ext *angles_ext = (bgc_angles_ext *)bgc_rx_msg->payload;
    //             ROS_INFO("Yaw %0.4f %0.4f %0.4f", 
    //                 INT16_TO_DEG(angles_ext->imu_angle_yaw),
    //                 INT16_TO_DEG(angles_ext->target_angle_yaw),
    //                 INT16_TO_DEG(angles_ext->stator_angle_yaw));

    //              ROS_INFO("Pitch %0.4f %0.4f %0.4f", 
    //                 INT16_TO_DEG(angles_ext->imu_angle_pitch),
    //                 INT16_TO_DEG(angles_ext->target_angle_pitch),
    //                 INT16_TO_DEG(angles_ext->stator_angle_pitch));
    //           }
    //           else if (bgc_rx_msg->command_id == CMD_ERROR) {
    //              ROS_ERROR("Received CMD_ERROR from BGC");
    //           }
    //           else if (bgc_rx_msg->command_id == CMD_CONFIRM) {
    //              // No need to do anything
    //           }
    //           else {
    //             ROS_INFO("Received unknown message of type %d", bgc_rx_msg->command_id);
    //           }
    //         }

    //         bgc_state = BGC_WAITING_FOR_START_BYTE;
    //       }
    //     }
    //   }

    //   ros::spinOnce();
    //   loop_rate.sleep();
    // }

    return EXIT_SUCCESS;
}