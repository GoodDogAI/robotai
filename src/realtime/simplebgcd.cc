#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <deque>

#include <fmt/core.h>
#include <fmt/chrono.h>

#include "util.h"
#include "config.h"
#include "serial.h"
#include "simplebgc.h"
#include "cereal/messaging/messaging.h"

ExitHandler do_exit;
const auto loop_time = std::chrono::milliseconds(100);

enum class YawGyroState {
  INIT,
  WAIT_FOR_REBOOT,
  WAIT_FOR_CENTER,
  OPERATING,
};

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
    SimpleBGC(Serial &p): port(p), bgc_state(ReadState::WAITING_FOR_START_BYTE),
     bgc_payload_counter(0), bgc_payload_crc{ 0, 0 }, cur_msg() {

    }

    std::unique_ptr<bgc_msg> read_msg(std::chrono::milliseconds timeout) {
      if (msg_queue.size() > 0) {
        auto msg = std::move(msg_queue.front());
        msg_queue.pop_front();
        return msg;
      }

      auto read = port.read_bytes(timeout);

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
            msg_queue.push_back(std::make_unique<bgc_msg>(cur_msg));
          }

          bgc_state = ReadState::WAITING_FOR_START_BYTE;  
        }
      }

      if (msg_queue.size() > 0) {
        auto msg = std::move(msg_queue.front());
        msg_queue.pop_front();
        return msg;
      }
      else {
        return nullptr;
      }
    }

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

  void send_message(uint8_t cmd, uint8_t *payload, uint16_t payload_size) {
    bgc_msg cmd_msg = {
      .command_id = cmd,
      .payload_size = static_cast<uint8_t>(payload_size),
      .header_checksum = static_cast<uint8_t>(cmd + payload_size),
    };

    std::copy(payload, payload + payload_size, cmd_msg.payload);
    
    uint8_t crc[2];
    crc16_calculate(BGC_HEADER_SIZE + payload_size, reinterpret_cast<uint8_t*>(&cmd_msg), crc);

    std::lock_guard<std::mutex> guard {send_msg_mutex};

    port.write_byte(BGC_V2_START_BYTE);
    port.write_bytes(&cmd_msg, BGC_HEADER_SIZE + payload_size);
    port.write_bytes(crc, 2);
  }

  private:
    Serial &port;

    ReadState bgc_state;
    uint8_t bgc_payload_counter;
    uint8_t bgc_payload_crc[2];
    bgc_msg cur_msg;
    std::deque<std::unique_ptr<bgc_msg>> msg_queue;

    std::mutex send_msg_mutex;
};

static void bgc_command_processor(SimpleBGC &bgc) {
  // Note, we use a SubSocket, not a SubMaster, because the brain channel can have multiple messages queued up at once
  // and a SubMaster only keeps one message buffered at a time
  std::unique_ptr<Context> ctx{ Context::create() };
  std::unique_ptr<Poller> poller{ Poller::create() };
  std::unique_ptr<SubSocket> sock{SubSocket::create(ctx.get(), "brainCommands")};

  std::chrono::steady_clock::time_point control_last_sent {};

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

      if (event.which() != cereal::Event::HEAD_COMMAND) 
        continue;
        
      auto headcmd = event.getHeadCommand();
    
      bgc_control_data control_data;
      build_control_msg(headcmd.getPitchAngle(), headcmd.getYawAngle(), &control_data);
      bgc.send_message(CMD_CONTROL, reinterpret_cast<uint8_t *>(&control_data), sizeof(bgc_control_data));

      //fmt::print("Sent pitch: {}, yaw: {}\n", headcmd.getPitchAngle(), headcmd.getYawAngle());

      control_last_sent = now;
    }

    if (now - control_last_sent > std::chrono::milliseconds(100)) {
        fmt::print(stderr, "Sending timeout BGC command\n");
        bgc_control_data control_data;
        build_control_msg(30.0f, -10.0f, &control_data);

        bgc.send_message(CMD_CONTROL, reinterpret_cast<uint8_t *>(&control_data), sizeof(bgc_control_data));

        control_last_sent = now;
    }
  }
}

int main(int argc, char **argv)
{
  PubMaster pm{{"simpleBGCFeedback"}};
  Serial port{SIMPLEBGC_SERIAL_PORT, SIMPLEBGC_BAUD_RATE};
  SimpleBGC bgc{port};

  YawGyroState yaw_gyro_state;
  auto gyro_center_start_time {std::chrono::steady_clock::now()};
 
  std::chrono::steady_clock::time_point bgc_last_received {};

  // Reset the module, so you have a clean connection to it each time
  bgc_reset reset_cmd {};
  bgc.send_message(CMD_RESET, reinterpret_cast<uint8_t *>(&reset_cmd), sizeof(bgc_reset));
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
  bgc.send_message(CMD_DATA_STREAM_INTERVAL, reinterpret_cast<uint8_t *>(&stream_data), sizeof(stream_data));

  // Ask the yaw to be recenretered
  bgc_execute_menu exe_menu {};
  exe_menu.cmd_id = SBGC_MENU_CENTER_YAW;
  bgc.send_message(CMD_EXECUTE_MENU, reinterpret_cast<uint8_t *>(&exe_menu), sizeof(bgc_execute_menu));

  yaw_gyro_state = YawGyroState::WAIT_FOR_CENTER;
  gyro_center_start_time = std::chrono::steady_clock::now();
  bgc_last_received = std::chrono::steady_clock::now();

  // Start the command processor thread
  std::thread command_thread {bgc_command_processor, std::ref(bgc)};

  while (!do_exit)
  {
    auto start_loop { std::chrono::steady_clock::now() };

    if (start_loop - bgc_last_received > std::chrono::seconds(1)) {
        fmt::print("No messages received in past 1 second, shutting down BGC subsystem");
        return EXIT_FAILURE;
    }

    auto msg = bgc.read_msg(std::chrono::duration_cast<std::chrono::milliseconds>(loop_time));

    if (msg)
    {
      bgc_last_received = start_loop;

      if (msg->command_id == CMD_REALTIME_DATA_4)
      {
        bgc_realtime_data_4 *realtime_data = reinterpret_cast<bgc_realtime_data_4 *>(msg->payload);

        if (realtime_data->system_error)
        {
          uint16_t err = realtime_data->system_error;
          fmt::print(stderr, "BGC Error {:02x}\n", err);
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
            fmt::print("YAW Gyro centered, angle {:0.2f}\n", INT16_TO_DEG(realtime_data->stator_angle_yaw));
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
        pm.send("simpleBGCFeedback", bytes.begin(), bytes.size());
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

  command_thread.join();

  return EXIT_SUCCESS;
}