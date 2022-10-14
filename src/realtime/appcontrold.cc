#include <unistd.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/rfcomm.h>
#include <vector>
#include <chrono>

#include <capnp/pretty-print.h>
#include <fmt/core.h>

#include "util.h"
#include "config.h"
#include "cereal/messaging/messaging.h"

#define MAX_MISSED_INTERVALS 3

#define BUF_DISCRIMINANT_INDEX 4

#define BUF_EMPTY_MESSAGE 0x00
#define BUF_MOVE_MESSAGE 0x01
#define BUF_SCORE_MESSAGE 0x02

#define BUF_SCORE_INDEX 5
#define BUF_CMD_VEL_LINEAR_X_INDEX 5
#define BUF_CMD_VEL_ANGULAR_Z_INDEX 6

const char *service_name = "appControl";
ExitHandler do_exit;

void sendMessage(PubMaster &pm, bool connected,
                 cereal::AppControl::RewardState rewardState = cereal::AppControl::RewardState::NO_OVERRIDE,
                 cereal::AppControl::MotionState moveState = cereal::AppControl::MotionState::NO_OVERRIDE,
                 float cmdVelLinearX = 0.0f, float cmdVelAngularZ = 0.0f) {
    MessageBuilder msg;
    auto event = msg.initEvent(true);
    auto adat = event.initAppControl();
    adat.setConnectionState(connected ? cereal::AppControl::ConnectionState::CONNECTED : cereal::AppControl::ConnectionState::NOT_CONNECTED);
    adat.setRewardState(rewardState);
    adat.setMotionState(moveState);
    adat.setLinearXOverride(cmdVelLinearX);
    adat.setAngularZOverride(cmdVelAngularZ);
    
    auto words = capnp::messageToFlatArray(msg);
    auto bytes = words.asBytes();
    pm.send(service_name, bytes.begin(), bytes.size());

    //fmt::print("msg: {}\n", capnp::prettyPrint(event).flatten().cStr());
}

int main(int argc, char **argv)
{
    PubMaster pm { {service_name} };

    std::chrono::steady_clock::time_point last_bt_msg_recv = std::chrono::steady_clock::now();
    int last_bt_timestamp = 0;
    int bt_jitters[16] = { 0 };

    auto last_reward_state { cereal::AppControl::RewardState::NO_OVERRIDE };
    auto last_move_state { cereal::AppControl::MotionState::NO_OVERRIDE };
    auto last_cmd_vel_linear_x { 0.0f };
    auto last_cmd_vel_angular_z { 0.0f };
    bool connected { false };

    struct sockaddr_rc loc_addr = { 0 }, rem_addr = { 0 };
    char buf[8] = { 0 };
    int s, client, bytes_read;
    socklen_t opt = sizeof(rem_addr);

    // allocate socket
    KJ_SYSCALL(s = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM), "Could not open socket");

    // bind socket to port 1 of the first available 
    // local bluetooth adapter
    bdaddr_t ANY = {0,0,0,0,0,0};
    loc_addr.rc_family = AF_BLUETOOTH;
    loc_addr.rc_bdaddr = ANY;
    loc_addr.rc_channel = (uint8_t) 1;

    KJ_SYSCALL(bind(s, (struct sockaddr *)&loc_addr, sizeof(loc_addr)), "Could not bind socket");
    fmt::print("Successfully bound socket\n");

    // put socket into listening mode
    KJ_SYSCALL(listen(s, 1), "Could not listen on socket");
    fmt::print("Listening on BT socket\n");

    // Wait for connections.
    std::vector<pollfd> input_fds;
    input_fds.push_back({s, POLLIN, 0});

    // Track time since last connection.
    int missed_intervals = 0;
    std::chrono::steady_clock::time_point last_msg_sent;
    sendMessage(pm, connected);
    last_msg_sent = std::chrono::steady_clock::now();

    while(!do_exit) {
        int input_ret;
        KJ_SYSCALL(input_ret = poll(input_fds.data(), input_fds.size(), 500), "input poll failed");
        
        if (std::chrono::steady_clock::now() - last_msg_sent > std::chrono::milliseconds(1000)) {
            sendMessage(pm, connected);
            last_msg_sent = std::chrono::steady_clock::now();
        }

        if (input_fds[0].revents & POLLIN) {
            // accept one connection
            KJ_SYSCALL(client = accept(s, (struct sockaddr *)&rem_addr, &opt), "Could not accept connection");

            ba2str( &rem_addr.rc_bdaddr, buf );
            fmt::print("accepted connection from {} (on fd {})\n", buf, client);
            
            std::vector<pollfd> conn_fds;
            conn_fds.push_back({client, POLLIN, 0});
            
            while (!do_exit) {
                if (std::chrono::steady_clock::now() - last_msg_sent > std::chrono::milliseconds(1000)) {
                    sendMessage(pm, connected);
                    last_msg_sent = std::chrono::steady_clock::now();
                }

                int con_ret;
                KJ_SYSCALL(con_ret = poll(conn_fds.data(), conn_fds.size(), 500), "connection poll failed");

                if (conn_fds[0].revents & POLLIN) {

                    memset(buf, 0, sizeof(buf));
                    // read data from the client
                    bytes_read = read(client, buf, sizeof(buf));
                    if( bytes_read > 0 ) {
                        if (!connected) {
                            fmt::print("Connected\n");
                            connected = true;
                            last_reward_state = cereal::AppControl::RewardState::NO_OVERRIDE;
                            last_move_state = cereal::AppControl::MotionState::NO_OVERRIDE;
                            last_cmd_vel_linear_x = 0;
                            last_cmd_vel_angular_z = 0;
                            sendMessage(pm, connected);
                            last_msg_sent = std::chrono::steady_clock::now();
                        }

                        int current_bt_timestamp = (buf[0] << 24) + (buf[1] << 16) + (buf[2] << 8) + buf[3];
                        int expected_delay = current_bt_timestamp - last_bt_timestamp;
                        auto now = std::chrono::steady_clock::now();
                        int actual_delay = (int)((now - last_bt_msg_recv).count() / 1000L / 1000L);
                        for (int i=0; i<15; i++) {
                            bt_jitters[i+1] = bt_jitters[i];
                        }
                        bt_jitters[0] = (expected_delay - actual_delay);
    
                        //ROS_INFO("read [%d,%d,%d,%d, %d,%d, %d,%d], jitter=%d", buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], total_jitter);
                        last_bt_timestamp = current_bt_timestamp;
                        last_bt_msg_recv = now;

                        if (buf[BUF_DISCRIMINANT_INDEX] == BUF_MOVE_MESSAGE) {
                            last_move_state = cereal::AppControl::MotionState::MANUAL_CONTROL;
                            last_cmd_vel_linear_x = ((int8_t)buf[BUF_CMD_VEL_LINEAR_X_INDEX]) / 127.0F * OVERRIDE_LINEAR_SPEED;
                            last_cmd_vel_angular_z = ((int8_t)buf[BUF_CMD_VEL_ANGULAR_Z_INDEX]) / 127.0F * OVERRIDE_ANGULAR_SPEED;      

                            fmt::print("Move: linear_x={}, angular_z={}\n", last_cmd_vel_linear_x, last_cmd_vel_angular_z); 
                        }
                        else if (buf[BUF_DISCRIMINANT_INDEX] == BUF_SCORE_MESSAGE) {
                            last_reward_state = static_cast<int8_t>(buf[BUF_SCORE_INDEX]) < 0  ? cereal::AppControl::RewardState::OVERRIDE_NEGATIVE : cereal::AppControl::RewardState::OVERRIDE_POSITIVE;
                        }
                        else if (buf[BUF_DISCRIMINANT_INDEX] == BUF_EMPTY_MESSAGE) {
                            last_move_state = cereal::AppControl::MotionState::NO_OVERRIDE;
                            last_cmd_vel_linear_x = 0;
                            last_cmd_vel_angular_z = 0;

                            fmt::print("Empty message\n");
                        }
                        else {
                            fmt::print(stderr, "Unknown message recieved {}\n", buf[BUF_DISCRIMINANT_INDEX]);
                        }

                        sendMessage(pm, connected, last_reward_state,
                                    last_move_state, last_cmd_vel_linear_x, last_cmd_vel_angular_z);
                        last_msg_sent = now;
                        last_reward_state = cereal::AppControl::RewardState::NO_OVERRIDE;

                        missed_intervals = 0;
                    }
                } else {
                    missed_intervals++;

                    if (missed_intervals >= MAX_MISSED_INTERVALS) {
                        fmt::print(stderr, "disconnected\n");
                        connected = false;
                        last_reward_state = cereal::AppControl::RewardState::NO_OVERRIDE;
                        last_move_state = cereal::AppControl::MotionState::NO_OVERRIDE;
                        last_cmd_vel_linear_x = 0;
                        last_cmd_vel_angular_z = 0;
                        sendMessage(pm, connected);
                        last_msg_sent = std::chrono::steady_clock::now();
                        break;
                    }
                }

            }
            // close connection
            close(client);
        }
    }

    close(s);
    return EXIT_SUCCESS;
}