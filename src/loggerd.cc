#include <cstdint>
#include <cstdlib>
#include <iostream>

constexpr uint8_t mask0{ 0b0000'0001 };

constexpr int32_t bitrate = 1'000'000;

int main(int argc, char *argv[])
{
    std::cout << "Hello, World!" << mask0 << bitrate << std::endl;
    return EXIT_SUCCESS;
}
