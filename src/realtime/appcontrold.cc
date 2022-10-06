#include <unistd.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <bluetooth/bluetooth.h>
#include <bluetooth/rfcomm.h>
#include <vector>
#include <chrono>

#include <fmt/core.h>

#define MAX_MISSED_INTERVALS 3

#define BUF_DISCRIMINANT_INDEX 4

#define BUF_EMPTY_MESSAGE 0x00
#define BUF_MOVE_MESSAGE 0x01
#define BUF_SCORE_MESSAGE 0x02

#define BUF_SCORE_INDEX 5
#define BUF_CMD_VEL_LINEAR_X_INDEX 5
#define BUF_CMD_VEL_ANGULAR_Z_INDEX 6

int main(int argc, char **argv)
{
    std::chrono::steady_clock::time_point last_penalty;
    int8_t last_score_received = 0;

    std::chrono::steady_clock::time_point last_bt_msg_recv = std::chrono::steady_clock::now();
    int last_bt_timestamp = 0;
    int bt_jitters[16] = { 0 };

    struct sockaddr_rc loc_addr = { 0 }, rem_addr = { 0 };
    char buf[8] = { 0 };
    int s, client, bytes_read;
    socklen_t opt = sizeof(rem_addr);

    int ret_code;

    // allocate socket
    s = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM);

    // bind socket to port 1 of the first available 
    // local bluetooth adapter
    bdaddr_t ANY = {0,0,0,0,0,0};
    loc_addr.rc_family = AF_BLUETOOTH;
    loc_addr.rc_bdaddr = ANY;
    loc_addr.rc_channel = (uint8_t) 1;
    ret_code = bind(s, (struct sockaddr *)&loc_addr, sizeof(loc_addr));
    fmt::print("bind: {}\n", ret_code);

    // put socket into listening mode
    ret_code = listen(s, 1);
    fmt::print("listen: {}\n", ret_code);

    // Wait for connections.
    std::vector<pollfd> input_fds;
    input_fds.push_back({s, POLLIN, 0});

    // Track time since last connection.
    int missed_intervals = 0;
    // std_msgs::Bool connected_msg;
    std::chrono::steady_clock::time_point last_connected_msg_sent;
    // connected_msg.data = false;
    // reward_connected.publish(connected_msg);    
    // TODO: publish connected_msg
    last_connected_msg_sent = std::chrono::steady_clock::now();

    while(true) {
        int input_ret = poll(input_fds.data(), input_fds.size(), 500);

        if (input_ret < -1) {
            fmt::print(stderr, "poll error\n");
            return EXIT_FAILURE;
        }

        if (input_fds[0].revents & POLLIN) {
            // accept one connection
            client = accept(s, (struct sockaddr *)&rem_addr, &opt);

            ba2str( &rem_addr.rc_bdaddr, buf );
            fmt::print("accepted connection from {} (on fd {})\n", buf, client);
            
            std::vector<pollfd> conn_fds;
            conn_fds.push_back({client, POLLIN, 0});
            
            while (true) {
                // if (ros::SteadyTime::now() - last_connected_msg_sent > ros::WallDuration(1.0)) {
                //     reward_connected.publish(connected_msg);
                //     last_connected_msg_sent = ros::SteadyTime::now();
                // }
                // TODO publish connected_msg

                int con_ret = poll(conn_fds.data(), conn_fds.size(), 500);

                if (con_ret < -1) {
                    fmt::print(stderr, "poll error\n");
                    return EXIT_FAILURE;
                }

                if (conn_fds[0].revents & POLLIN) {

                    memset(buf, 0, sizeof(buf));
                    // read data from the client
                    bytes_read = read(client, buf, sizeof(buf));
                    if( bytes_read > 0 ) {
                        // if (!connected_msg.data) {
                        //     fmt::print("Connected\n");
                        //     connected_msg.data = true;
                        //     reward_connected.publish(connected_msg);
                        //     last_connected_msg_sent = ros::SteadyTime::now();
                        // }
                        // data_msg.data = buf[0];
                        // reward_raw_pub.publish(data_msg);
                        // TODO: publish data_msg

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
                            // geometry_msgs::Twist cmd_vel_msg;
                            // cmd_vel_msg.linear.x = ((int8_t)buf[BUF_CMD_VEL_LINEAR_X_INDEX]) / 127.0F * override_linear_speed;
                            // cmd_vel_msg.angular.z = ((int8_t)buf[BUF_CMD_VEL_ANGULAR_Z_INDEX]) / 127.0F * override_angular_speed;
                            
                            // reward_cmd_vel_pub.publish(cmd_vel_msg);
                            // override_cmd_vel_msg.data = true;
                        }
                        else if (buf[BUF_DISCRIMINANT_INDEX] == BUF_SCORE_MESSAGE) {
                            last_score_received = buf[BUF_SCORE_INDEX];
                            fmt::print("Score {} detected from app\n", last_score_received);
                            last_penalty = std::chrono::steady_clock::now();
                        }
                        else if (buf[BUF_DISCRIMINANT_INDEX] == BUF_EMPTY_MESSAGE) {
                            //override_cmd_vel_msg.data = false;
                            // TODO publish override_cmd_vel_msg
                            fmt::print("Empty message detected from app\n");
                        }
                        else {
                            fmt::print(stderr, "Unknown message recieved {}\n", buf[BUF_DISCRIMINANT_INDEX]);
                        }


                        // reward_override_cmd_vel_pub.publish(override_cmd_vel_msg);


                        // Publish the latest value of the reward button
                        // std_msgs::Float32 reward;
                        // reward.data = (ros::SteadyTime::now() - last_penalty) > penalty_duration ? 0.0 : (float)last_score_received;
                        // reward_pub.publish(reward);

                        missed_intervals = 0;
                    }
                } else {
                    missed_intervals++;
                    // TODO Check this if statement again
                    //if (missed_intervals >= MAX_MISSED_INTERVALS && connected_msg.data) {
                    if (missed_intervals >= MAX_MISSED_INTERVALS) {
                        fmt::print(stderr, "disconnected\n");
                        // connected_msg.data = false;
                        // reward_connected.publish(connected_msg);
                        last_connected_msg_sent = std::chrono::steady_clock::now();
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