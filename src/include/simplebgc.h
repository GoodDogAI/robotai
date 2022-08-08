#pragma once

#include <stdint.h>

#define BGC_V2_START_BYTE 0x24
#define BGC_HEADER_SIZE 3

typedef struct {
  uint8_t command_id;
  uint8_t payload_size;
  uint8_t header_checksum;
  uint8_t payload[256];
} bgc_msg;

typedef struct __attribute__((__packed__)){
  uint8_t board_ver;
  uint16_t firmware_ver;
  
  uint8_t state_flag_debug_mode : 1;
  uint8_t state_flag_is_frame_inverted : 1;
  uint8_t state_flag_init_step1_done : 1;
  uint8_t state_flag_init_step2_done : 1;
  uint8_t state_flag_startup_auto_routine_done : 1;
  uint8_t state_flag_rest : 3;

  uint16_t board_features;
  uint8_t connection_flag;
  uint32_t frw_extra_id;
  uint16_t board_features_ext;
  uint8_t reserved[3];
  uint16_t base_frw_ver;
} bgc_board_info;

typedef struct __attribute__((__packed__)) {
  int16_t acc_roll;
  int16_t gyro_roll;
  int16_t acc_pitch;
  int16_t gyro_pitch;
  int16_t acc_yaw;
  int16_t gyro_yaw;

  uint16_t serial_err_cnt;
  uint16_t system_error;
  uint8_t system_sub_error;
  uint8_t reserved_0;
  uint8_t reserved_1;
  uint8_t reserved_2;

  int16_t rc_roll;
  int16_t rc_pitch;
  int16_t rc_yaw;

  int16_t rc_cmd;

  int16_t ext_fc_roll;
  int16_t ext_fc_pitch;

  int16_t imu_angle_roll;
  int16_t imu_angle_pitch;
  int16_t imu_angle_yaw;

  int16_t frame_imu_angle_roll;
  int16_t frame_imu_angle_pitch;
  int16_t frame_imu_angle_yaw;

  int16_t target_angle_roll;
  int16_t target_angle_pitch;
  int16_t target_angle_yaw;

  uint16_t cycle_time;
  uint16_t i2c_error_count;

  uint8_t deprecated_1;
  uint16_t bat_level;
  uint8_t rt_data_flags;
  uint8_t cur_imu;
  uint8_t cur_profile;

  uint8_t motor_power_roll;
  uint8_t motor_power_pitch;
  uint8_t motor_power_yaw;

  int16_t stator_angle_roll;
  int16_t stator_angle_pitch;
  int16_t stator_angle_yaw;

  uint8_t reserved_3;

  int16_t balance_error_roll;
  int16_t balance_error_pitch;
  int16_t balance_error_yaw;

  uint16_t current_ma;

  int16_t mag_data_0;
  int16_t mag_data_1;
  int16_t mag_data_2;

  int8_t imu_temp;
  int8_t frame_imu_temp;
  uint8_t img_g_err;
  uint8_t img_h_err;
  
  int16_t motor_out_roll;
  int16_t motor_out_pitch;
  int16_t motor_out_yaw;

  uint8_t calib_mode;
  uint8_t can_imu_ext_sense_err;

  uint8_t reserved_buf[28];
} bgc_realtime_data_4;


typedef struct __attribute__((__packed__)) {
  uint8_t control_mode_roll;
  uint8_t control_mode_pitch;
  uint8_t control_mode_yaw;

  int16_t speed_roll;
  int16_t angle_roll;

  int16_t speed_pitch;
  int16_t angle_pitch;

  int16_t speed_yaw;
  int16_t angle_yaw;
} bgc_control_data;

typedef struct __attribute__((__packed__)) {
  int16_t imu_angle_roll;
  int16_t target_angle_roll;
  int32_t stator_angle_roll;
  int8_t reserved_roll[10];

  int16_t imu_angle_pitch;
  int16_t target_angle_pitch;
  int32_t stator_angle_pitch;
  int8_t reserved_pitch[10];

  int16_t imu_angle_yaw;
  int16_t target_angle_yaw;
  int32_t stator_angle_yaw;
  int8_t reserved_yaw[10];
} bgc_angles_ext;

typedef struct __attribute__((__packed__)) {
  uint8_t cmd_id;
  uint16_t interval_ms;

  uint8_t config[8];
  uint8_t sync_to_data;
  uint8_t reserved[9];
} bgc_data_stream_interval;

typedef struct __attribute__((__packed__)) {
  uint8_t flags;
  uint16_t delay_ms;
} bgc_reset;

typedef struct __attribute__((__packed__)) {
  uint8_t num_params; // Set to 1 for now
  uint8_t param_id;
  uint32_t param_value;
} bgc_cmd_set_adj_vars;

#define BEEPER_MODE_CALIBRATE (1<<0)
#define BEEPER_MODE_CONFIRM (1<<1)
#define BEEPER_MODE_ERROR (1<<2)
#define BEEPER_MODE_CLICK (1<<4)
#define BEEPER_MODE_COMPLETE (1<<5)
#define BEEPER_MODE_INTRO (1<<6)
#define BEEPER_MODE_CUSTOM_MELODY (1<<15)

typedef struct __attribute__((__packed__)) {
  uint16_t mode;
  uint8_t note_length;
  uint8_t decay_factor;

  uint8_t reserved[8];
} bgc_beep_builtin_sound;

typedef struct __attribute__((__packed__)) {
  uint16_t mode;
  uint8_t note_length;
  uint8_t decay_factor;

  uint8_t reserved[8];

  uint16_t notes[2];
} bgc_beep_custom_sound;

typedef struct __attribute__((__packed__)) {
  uint8_t cmd_id;
} bgc_execute_menu;

#define INT16_TO_DEG(x) ((x) * 0.02197265625f)
#define DEG_TO_INT16(x) ((x) / 0.02197265625f)

#define CONTROL_SPEED_DEG_PER_SEC_PER_UNIT 0.1220740379f

#define CONTROL_MODE_NO_CONTROL 0
#define CONTROL_MODE_IGNORE 7
#define CONTROL_MODE_SPEED 1
#define CONTROL_MODE_ANGLE 2
#define CONTROL_MODE_SPEED_ANGLE 3
#define CONTROL_MODE_RC 4
#define CONTROL_MODE_ANGLE_REL_FRAME 5
#define CONTROL_MODE_RC_HIGH_RES 6

#define CONTROL_FLAG_AUTO_TASK (1<<6)
#define CONTROL_FLAG_FORCE_RC_SPEED (1<<6)
#define CONTROL_FLAG_HIGH_RES_SPEED (1<<7)
#define CONTROL_FLAG_TARGET_PRECISE (1<<5)

#define CMD_READ_PARAMS  82
#define CMD_WRITE_PARAMS  87
#define CMD_REALTIME_DATA  68
#define CMD_BOARD_INFO  86
#define CMD_CALIB_ACC  65
#define CMD_CALIB_GYRO  103
#define CMD_CALIB_EXT_GAIN  71
#define CMD_USE_DEFAULTS  70
#define CMD_CALIB_POLES  80
#define CMD_RESET  114
#define CMD_HELPER_DATA 72
#define CMD_CALIB_OFFSET  79
#define CMD_CALIB_BAT  66
#define CMD_MOTORS_ON   77
#define CMD_MOTORS_OFF  109
#define CMD_CONTROL   67
#define CMD_TRIGGER_PIN  84
#define CMD_EXECUTE_MENU 69
#define CMD_GET_ANGLES  73
#define CMD_CONFIRM  67
#define CMD_BOARD_INFO_3  20
#define CMD_READ_PARAMS_3 21
#define CMD_WRITE_PARAMS_3 22
#define CMD_REALTIME_DATA_3  23
#define CMD_REALTIME_DATA_4  25
#define CMD_SELECT_IMU_3 24
#define CMD_READ_PROFILE_NAMES 28
#define CMD_WRITE_PROFILE_NAMES 29
#define CMD_QUEUE_PARAMS_INFO_3 30
#define CMD_SET_ADJ_VARS_VAL 31
#define CMD_SAVE_PARAMS_3 32
#define CMD_READ_PARAMS_EXT 33
#define CMD_WRITE_PARAMS_EXT 34
#define CMD_AUTO_PID 35
#define CMD_SERVO_OUT 36
#define CMD_I2C_WRITE_REG_BUF 39
#define CMD_I2C_READ_REG_BUF 40
#define CMD_WRITE_EXTERNAL_DATA 41
#define CMD_READ_EXTERNAL_DATA 42
#define CMD_READ_ADJ_VARS_CFG 43
#define CMD_WRITE_ADJ_VARS_CFG 44
#define CMD_API_VIRT_CH_CONTROL 45
#define CMD_ADJ_VARS_STATE 46
#define CMD_EEPROM_WRITE 47
#define CMD_EEPROM_READ 48
#define CMD_CALIB_INFO 49
#define CMD_SIGN_MESSAGE 50
#define CMD_BOOT_MODE_3 51
#define CMD_SYSTEM_STATE 52
#define CMD_READ_FILE 53
#define CMD_WRITE_FILE 54
#define CMD_FS_CLEAR_ALL 55
#define CMD_AHRS_HELPER 56
#define CMD_RUN_SCRIPT 57
#define CMD_SCRIPT_DEBUG 58
#define CMD_CALIB_MAG 59
#define CMD_GET_ANGLES_EXT 61
#define CMD_READ_PARAMS_EXT2 62
#define CMD_WRITE_PARAMS_EXT2 63
#define CMD_GET_ADJ_VARS_VAL 64
#define CMD_CALIB_MOTOR_MAG_LINK 74
#define CMD_GYRO_CORRECTION 75
#define CMD_DATA_STREAM_INTERVAL 85
#define CMD_REALTIME_DATA_CUSTOM 88
#define CMD_BEEP_SOUND 89
#define CMD_ENCODERS_CALIB_OFFSET_4  26
#define CMD_ENCODERS_CALIB_FLD_OFFSET_4 27
#define CMD_CONTROL_CONFIG 90
#define CMD_CALIB_ORIENT_CORR 91
#define CMD_COGGING_CALIB_INFO 92
#define CMD_CALIB_COGGING 93
#define CMD_CALIB_ACC_EXT_REF 94
#define CMD_PROFILE_SET 95
#define CMD_CAN_DEVICE_SCAN 96
#define CMD_CAN_DRV_HARD_PARAMS 97
#define CMD_CAN_DRV_STATE 98
#define CMD_CAN_DRV_CALIBRATE 99
#define CMD_READ_RC_INPUTS 100
#define CMD_REALTIME_DATA_CAN_DRV 101
#define CMD_EVENT 102
#define CMD_READ_PARAMS_EXT3 104
#define CMD_WRITE_PARAMS_EXT3 105
#define CMD_EXT_IMU_DEBUG_INFO 106
#define CMD_SET_DEVICE_ADDR 107
#define CMD_AUTO_PID2 108
#define CMD_EXT_IMU_CMD 110
#define CMD_READ_STATE_VARS 111
#define CMD_WRITE_STATE_VARS 112
#define CMD_SERIAL_PROXY 113
#define CMD_IMU_ADVANCED_CALIB 115
#define CMD_API_VIRT_CH_HIGH_RES 116
#define CMD_SET_DEBUG_PORT 249
#define CMD_MAVLINK_INFO 250
#define CMD_MAVLINK_DEBUG 251
#define CMD_DEBUG_VARS_INFO_3 253
#define CMD_DEBUG_VARS_3 254
#define CMD_ERROR  255

// Menu actions (used in the SBGC_MENU_BUTTON_PRESS command, menu button assignment, RC_CMD channel assignment)
#define SBGC_MENU_PROFILE1 1
#define SBGC_MENU_PROFILE2 2
#define SBGC_MENU_PROFILE3 3
#define SBGC_MENU_SWAP_PITCH_ROLL 4
#define SBGC_MENU_SWAP_YAW_ROLL 5
#define SBGC_MENU_CALIB_ACC 6
#define SBGC_MENU_RESET 7
#define SBGC_MENU_SET_ANGLE 8
#define SBGC_MENU_CALIB_GYRO 9
#define SBGC_MENU_MOTOR_TOGGLE 10
#define SBGC_MENU_MOTOR_ON 11
#define SBGC_MENU_MOTOR_OFF 12
#define SBGC_MENU_FRAME_UPSIDE_DOWN 13
#define SBGC_MENU_PROFILE4 14
#define SBGC_MENU_PROFILE5 15
#define SBGC_MENU_AUTO_PID 16
#define SBGC_MENU_LOOK_DOWN 17
#define SBGC_MENU_HOME_POSITION 18
#define SBGC_MENU_RC_BIND 19
#define SBGC_MENU_CALIB_GYRO_TEMP 20
#define SBGC_MENU_CALIB_ACC_TEMP 21
#define SBGC_MENU_BUTTON_PRESS 22
#define SBGC_MENU_RUN_SCRIPT1 23
#define SBGC_MENU_RUN_SCRIPT2 24
#define SBGC_MENU_RUN_SCRIPT3 25
#define SBGC_MENU_RUN_SCRIPT4 26
#define SBGC_MENU_RUN_SCRIPT5 27
#define SBGC_MENU_RUN_SCRIPT6 28
#define SBGC_MENU_RUN_SCRIPT7 29
#define SBGC_MENU_RUN_SCRIPT8 30
#define SBGC_MENU_RUN_SCRIPT9 31
#define SBGC_MENU_RUN_SCRIPT10 32
#define SBGC_MENU_CALIB_MAG 33
#define SBGC_MENU_LEVEL_ROLL_PITCH 34
#define SBGC_MENU_CENTER_YAW 35
#define SBGC_MENU_UNTWIST_CABLES 36
#define SBGC_MENU_SET_ANGLE_NO_SAVE 37
#define SBGC_MENU_HOME_POSITION_SHORTEST 38
#define SBGC_MENU_CENTER_YAW_SHORTEST 39
