#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <thread>
#include <chrono>

#include "msgvec.h"
#include "config.h"


TEST_CASE( "Proess a simple message", "[msgvec]" ) {
   MsgVec msgvec{R"(
        {
            "obs": [],
            "act": []
        }
    )"};
}
