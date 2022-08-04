#pragma once

#include <string>
#include <regex>
#include <termios.h>

class Serial {
    public:
        Serial(std::string device, int baudrate = B115200);
        ~Serial();

        uint8_t read_byte();
        std::string read_regex(std::regex &re);

        void write_byte(uint8_t);
        void write_str(const std::string &data);

        const std::string device;
        const int baudrate;

    private:
        int fd;
};