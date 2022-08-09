#pragma once

#include <string>
#include <regex>
#include <chrono>
#include <termios.h>

class Serial {
    public:
        Serial(std::string device, int baudrate = B115200);
        ~Serial();

        uint8_t read_byte();
        std::optional<std::vector<uint8_t>> read_bytes(std::chrono::milliseconds timeout = std::chrono::milliseconds(100));
        std::string read_regex(const std::regex &re);

        void write_byte(uint8_t);
        void write_bytes(const void *data, size_t len);
        void write_str(const std::string &data);

        const std::string device;
        const int baudrate;

    private:
        int fd;
};