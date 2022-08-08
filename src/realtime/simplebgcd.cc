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

enum class YawGyroState {
  INIT,
  WAIT_FOR_REBOOT,
  WAIT_FOR_CENTER,
  OPERATING,
};

static void crc16_update(uint16_t length, uint8_t *data, uint8_t crc[2]) {
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

static void crc16_calculate(uint16_t length, uint8_t *data, uint8_t crc[2]) {
  crc[0] = 0; crc[1] = 0;
  crc16_update(length, data, crc);
}

static void send_message(Serial &port, uint8_t cmd, uint8_t *payload, uint16_t payload_size) {
  bgc_msg cmd_msg = {
    .command_id = cmd,
    .payload_size = static_cast<uint8_t>(payload_size),
    .header_checksum = static_cast<uint8_t>(cmd + payload_size),
  };

  std::copy(payload, payload + payload_size, cmd_msg.payload);
  
  uint8_t crc[2];
  crc16_calculate(BGC_HEADER_SIZE + payload_size, reinterpret_cast<uint8_t*>(&cmd_msg), crc);

  port.write_byte(BGC_V2_START_BYTE);
  port.write_bytes(&cmd_msg, BGC_HEADER_SIZE + payload_size);
  port.write_bytes(crc, 2);
}

static int16_t degree_to_int16(float angle) {
   float result = round(DEG_TO_INT16(angle));

   if (result <= -32768)
     return -32768;
   else if (result >= 32767)
     return 32767;
   else
     return result;
}

static void build_control_msg(float pitch, float yaw, bgc_control_data *control_data) {
    *control_data = {};

    control_data->control_mode_roll = CONTROL_MODE_IGNORE;
    control_data->control_mode_pitch = CONTROL_MODE_ANGLE_REL_FRAME;
    control_data->control_mode_yaw = CONTROL_MODE_ANGLE_REL_FRAME;
    control_data->angle_pitch = degree_to_int16(pitch);
    control_data->angle_yaw = degree_to_int16(yaw);

    control_data->speed_pitch = round(200.0f / CONTROL_SPEED_DEG_PER_SEC_PER_UNIT); 
    control_data->speed_yaw = round(200.0f / CONTROL_SPEED_DEG_PER_SEC_PER_UNIT); 
}

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

class SimpleBGC {
  enum class ReadState {
      WAITING_FOR_START_BYTE,
      READ_COMMAND_ID,
      READ_PAYLOAD_SIZE,
      READ_HEADER_CHECKSUM,
      READ_PAYLOAD,
      READ_CRC_0,
      READ_CRC_1,
  };

  public:
    SimpleBGC(Serial &p): port(p), bgc_state(ReadState::WAITING_FOR_START_BYTE), bgc_payload_counter(0), bgc_payload_crc{ 0, 0 }, cur_msg() {}

    std::unique_ptr<bgc_msg> read_msg_nonblocking() {
      std::unique_ptr<bgc_msg> result = nullptr;
      auto read = port.read_bytes_nonblocking();

      if (!read) {
        return nullptr;
      }

      for (uint8_t data : read.value()) {
        if (bgc_state == ReadState::WAITING_FOR_START_BYTE && data == BGC_V2_START_BYTE)
        {
          bgc_state = ReadState::READ_COMMAND_ID;
        }
        else if (bgc_state == ReadState::READ_COMMAND_ID)
        {
          cur_msg.command_id = data;
          bgc_state = ReadState::READ_PAYLOAD_SIZE;
        }
        else if (bgc_state == ReadState::READ_PAYLOAD_SIZE)
        {
          cur_msg.payload_size = data;
          bgc_state = ReadState::READ_HEADER_CHECKSUM;
        }
        else if (bgc_state == ReadState::READ_HEADER_CHECKSUM)
        {
          cur_msg.header_checksum = data;

          if (cur_msg.header_checksum != cur_msg.command_id + cur_msg.payload_size)
          {
            fmt::print(stderr, "Header checksum failed\n");
            bgc_state = ReadState::WAITING_FOR_START_BYTE;
          }
          else
          {
            bgc_state = ReadState::READ_PAYLOAD;
            bgc_payload_counter = 0;
          }
        }
        else if (bgc_state == ReadState::READ_PAYLOAD)
        {
          cur_msg.payload[bgc_payload_counter] = data;
          bgc_payload_counter++;

          if (bgc_payload_counter == cur_msg.payload_size)
          {
            bgc_state = ReadState::READ_CRC_0;
          }
        }
        else if (bgc_state == ReadState::READ_CRC_0)
        {
          bgc_payload_crc[0] = data;
          bgc_state = ReadState::READ_CRC_1;
        }
        else if (bgc_state == ReadState::READ_CRC_1)
        {
          bgc_payload_crc[1] = data;

          uint8_t crc[2];
          crc16_calculate(BGC_HEADER_SIZE + cur_msg.payload_size, reinterpret_cast<uint8_t *>(&cur_msg), crc);

          if (crc[0] != bgc_payload_crc[0] || crc[1] != bgc_payload_crc[1])
          {
            fmt::print(stderr, "Payload checksum failed\n");
          }
          else
          {
            if (result != nullptr) {
              fmt::print(stderr, "Message of type {} dropped\n", result->command_id);
            }

            result = std::make_unique<bgc_msg>(cur_msg);
          }

          bgc_state = ReadState::WAITING_FOR_START_BYTE;  
        }
      }

      return std::move(result);
    }

  private:
    Serial &port;

    ReadState bgc_state;
    uint8_t bgc_payload_counter;
    uint8_t bgc_payload_crc[2];
    bgc_msg cur_msg;
};

int main(int argc, char **argv)
{
  PubMaster pm{{service_name}};
  Serial port{SIMPLEBGC_SERIAL_PORT, SIMPLEBGC_BAUD_RATE};
  SimpleBGC bgc{port};
  YawGyroState yaw_gyro_state;
  auto gyro_center_start_time {std::chrono::steady_clock::now()};

  std::chrono::steady_clock::time_point control_last_sent {};
  std::chrono::steady_clock::time_point bgc_last_received {};

  // Reset the module, so you have a clean connection to it each time

  bgc_reset reset_cmd {};
  send_message(port, CMD_RESET, reinterpret_cast<uint8_t *>(&reset_cmd), sizeof(bgc_reset));
  std::this_thread::sleep_for(std::chrono::milliseconds(500));


  for (int i = 0; i < 8; ++i)
  {
    fmt::print("Waiting for SimpleBGC to Reboot.{:.>{}}\n", "", i);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  fmt::print("Sending command to start realtime data stream for SimpleBGC\n");

  // Register a realtime data stream syncing up with the loop rate
  bgc_data_stream_interval stream_data {};
  stream_data.cmd_id = CMD_REALTIME_DATA_4;
  stream_data.interval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(loop_time).count();
  stream_data.sync_to_data = 0;
  send_message(port, CMD_DATA_STREAM_INTERVAL, reinterpret_cast<uint8_t *>(&stream_data), sizeof(stream_data));

  // TOOD THIS DOESN"T WORK
  // Ask the yaw to be recenretered
  bgc_execute_menu exe_menu {};
  exe_menu.cmd_id = SBGC_MENU_CENTER_YAW;
  send_message(port, CMD_EXECUTE_MENU, reinterpret_cast<uint8_t *>(&exe_menu), sizeof(bgc_execute_menu));

  yaw_gyro_state = YawGyroState::WAIT_FOR_CENTER;
  gyro_center_start_time = std::chrono::steady_clock::now();
  bgc_last_received = std::chrono::steady_clock::now();

  for (;;)
  {
    auto start_loop { std::chrono::steady_clock::now() };

    // Make sure that if the GYRO is operating, that we are sending a control command at a minimum frequency
    if (yaw_gyro_state == YawGyroState::OPERATING || yaw_gyro_state == YawGyroState::WAIT_FOR_CENTER) {
      if (start_loop - control_last_sent > std::chrono::milliseconds(50)) {
        bgc_control_data control_data;
        build_control_msg(0.0f, 0.0f, &control_data);

        send_message(port, CMD_CONTROL, reinterpret_cast<uint8_t *>(&control_data), sizeof(bgc_control_data));
        control_last_sent = start_loop;
      }
    }

    if (start_loop - bgc_last_received > std::chrono::seconds(1)) {
        fmt::print("No messages received in past 1 second, shutting down BGC subsystem");
        return EXIT_FAILURE;
    }

    auto msg = bgc.read_msg_nonblocking();

    if (msg)
    {
      bgc_last_received = start_loop;

      if (msg->command_id == CMD_REALTIME_DATA_4)
      {
        bgc_realtime_data_4 *realtime_data = reinterpret_cast<bgc_realtime_data_4 *>(msg->payload);

        if (realtime_data->system_error)
        {
          fmt::print(stderr, "BGC Error {:02x}\n", realtime_data->system_error);
          fmt::print(stderr, "Shutting down BGC\n");
          return EXIT_FAILURE;
        }

        // fmt::print("Yaw {:0.4f} {:0.4f} {:0.4f}\n",
        //            INT16_TO_DEG(realtime_data->imu_angle_yaw),
        //            INT16_TO_DEG(realtime_data->target_angle_yaw),
        //            INT16_TO_DEG(realtime_data->stator_angle_yaw));

        // fmt::print("Pitch {:0.4f} {:0.4f} {:0.4f}\n",
        //            INT16_TO_DEG(realtime_data->imu_angle_pitch),
        //            INT16_TO_DEG(realtime_data->target_angle_pitch),
        //            INT16_TO_DEG(realtime_data->stator_angle_pitch));

        if (yaw_gyro_state == YawGyroState::WAIT_FOR_CENTER)
        {
          if (std::abs(INT16_TO_DEG(realtime_data->stator_angle_yaw)) < 1.0)
          {
            yaw_gyro_state = YawGyroState::OPERATING;
            fmt::print("YAW Gyro centered, angle {:0.2f}", INT16_TO_DEG(realtime_data->stator_angle_yaw));
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
        MessageBuilder pmsg;
        auto event = pmsg.initEvent(true);
        auto dat = event.initHeadFeedback();
        dat.setPitchAngle(INT16_TO_DEG(realtime_data->stator_angle_pitch));
        dat.setYawAngle(INT16_TO_DEG(realtime_data->stator_angle_yaw));
        dat.setPitchMotorPower(realtime_data->motor_power_pitch / 255.0f);
        dat.setYawMotorPower(realtime_data->motor_power_yaw / 255.0f);
        
        auto words = capnp::messageToFlatArray(pmsg);
        auto bytes = words.asBytes();
        pm.send(service_name, bytes.begin(), bytes.size());
      }
      else if (msg->command_id == CMD_GET_ANGLES_EXT)
      {
        bgc_angles_ext *angles_ext = reinterpret_cast<bgc_angles_ext *>(msg->payload);
        fmt::print("Yaw {:0.4f} {:0.4f} {:0.4f}\n",
                   INT16_TO_DEG(angles_ext->imu_angle_yaw),
                   INT16_TO_DEG(angles_ext->target_angle_yaw),
                   INT16_TO_DEG(angles_ext->stator_angle_yaw));

        fmt::print("Pitch {:0.4f} {:0.4f} {:0.4f}\n",
                   INT16_TO_DEG(angles_ext->imu_angle_pitch),
                   INT16_TO_DEG(angles_ext->target_angle_pitch),
                   INT16_TO_DEG(angles_ext->stator_angle_pitch));
      }
      else if (msg->command_id == CMD_ERROR)
      {
        fmt::print(stderr, "Received CMD_ERROR from BGC\n");
      }
      else if (msg->command_id == CMD_CONFIRM)
      {
        // No need to do anything
      }
      else
      {
        fmt::print(stderr, "Received unknown message of type {}\n", msg->command_id);
      }
    }
  }

  return EXIT_SUCCESS;
}