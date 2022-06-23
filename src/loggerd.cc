#include <cstdint>
#include <cstdlib>
#include <iostream>


template <typename T>
struct Pair {
    T x{};
    T y{};
};

int main(int argc, char *argv[])
{
    Pair<int> z {5,4};

    std::cout << z.x << z.y;
    
    return EXIT_SUCCESS;
}
