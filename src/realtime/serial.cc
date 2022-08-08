#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <regex>

#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <poll.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "serial.h"


Serial::Serial(std::string d, int b):  device(d), baudrate(b) {
    fd = open(d.c_str(), O_RDWR);

    if (fd < 0) {
        fmt::print(stderr, "Error {} from open: {}\n", errno, strerror(errno));
        throw std::runtime_error("Could not open device");
    }

    struct termios tty {};

    // Read in existing settings, and handle any error
    if(tcgetattr(fd, &tty) != 0) {
        fmt::print(stderr, "Error {} from tcgetattr: {}\n", errno, strerror(errno));
        throw std::runtime_error("Could not open device");
    }

    cfsetispeed(&tty, baudrate);
    cfsetospeed(&tty, baudrate);

    // Set Local modes according to https://blog.mbedded.ninja/programming/operating-systems/linux/linux-serial-ports-using-c-cpp/
    tty.c_cflag &= ~CRTSCTS;
    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO; // Disable echo
    tty.c_lflag &= ~ECHOE; // Disable erasure
    tty.c_lflag &= ~ECHONL; // Disable new-line echo
    tty.c_lflag &= ~ISIG; // Disable interpretation of INTR, QUIT and SUSP

    // Set Input modes
    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // Turn off s/w flow ctrl
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL); // Disable any special handling of received bytes

    // Set output modes
    tty.c_oflag &= ~OPOST; // Prevent special interpretation of output bytes (e.g. newline chars)
    tty.c_oflag &= ~ONLCR; // Prevent conversion of newline to carriage return/line feed

    // Save tty settings, also checking for error
    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        fmt::print(stderr, "Error {} from tcsetattr: {}\n", errno, strerror(errno));
        throw std::runtime_error("Could not open device");
    }
}

Serial::~Serial() {
    close(fd);
}

uint8_t Serial::read_byte() {
    uint8_t buf;
    [[maybe_unused]] ssize_t num_read;
    
    num_read = read(fd, &buf, 1);
    assert(num_read == 1);

    return buf;
}

std::optional<std::vector<uint8_t>> Serial::read_bytes_nonblocking() {
    pollfd serial_port_poll = {fd, POLLIN, 0};
    std::vector<uint8_t> buf(1024);

    int ret = poll(&serial_port_poll, 1, 0);

    assert(ret >= 0);

    if (ret == 1) {
        ssize_t bytes_read = read(fd, buf.data(), std::size(buf));
        buf.resize(bytes_read);
        return buf;
    }

    return std::nullopt;
}

std::string Serial::read_regex(const std::regex &re) {
    std::vector<char> buf;
    std::cmatch m;

    while (buf.size() == 0 || !std::regex_match(std::addressof(*buf.cbegin()),
                                                std::addressof(*buf.cend()),
                                                m, re, std::regex_constants::match_default)) {
        char data;
        [[maybe_unused]] ssize_t num_read;
        
        num_read = read(fd, &data, 1);
        assert(num_read == 1);

        // fmt::print("{} - {}\n", buf, std::regex_match(std::addressof(*buf.cbegin()),
        //                       std::addressof(*buf.cend()),
        //                       m, re, std::regex_constants::match_default));

        buf.push_back(data);
    }

    return m[0].str();
}

void Serial::write_byte(uint8_t data) {
    uint8_t buf = data;
    [[maybe_unused]] ssize_t num_write;
    
    num_write = write(fd, &buf, 1);

    assert(num_write == 1);
}

void Serial::write_bytes(const void *data, size_t len) {
    [[maybe_unused]] ssize_t num_writen;
    
    num_writen = write(fd, data, len);

    assert(num_writen == len);
}

void Serial::write_str(const std::string &data) {
    [[maybe_unused]] ssize_t num_write;
    
    num_write = write(fd, data.c_str(), data.length());

    assert(num_write == data.length());
}