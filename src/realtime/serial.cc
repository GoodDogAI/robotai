#include <string>
#include <cassert>
#include <stdexcept>

#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <fmt/core.h>

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

void Serial::write_byte(uint8_t data) {
    uint8_t buf = data;
    [[maybe_unused]] ssize_t num_write;
    
    num_write = write(fd, &buf, 1);

    assert(num_write == 1);
}

void Serial::write_str(const std::string &data) {
    [[maybe_unused]] ssize_t num_write;
    
    num_write = write(fd, data.c_str(), data.length());

    assert(num_write == data.length());
}