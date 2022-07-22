cmake_minimum_required(VERSION 3.10)

project(RobotAI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Download and compile some lightweight dependencies
Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.1.0
)
FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 9.0.0
        )
FetchContent_MakeAvailable(fmt)

# Find any dependencies that must be installed on the system
find_package(CapnProto REQUIRED)
#find_package(ZeroMQ REQUIRED)

# Build and configure cereal, as an in-place build
include(ExternalProject)
ExternalProject_Add(cereal 
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/cereal"
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE TRUE
    BUILD_COMMAND "scons"
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(cereal SOURCE_DIR)
set(CEREAL_INCLUDE_DIRS ${SOURCE_DIR})
set(CEREAL_LIBRARIES "${SOURCE_DIR}/libcereal.a" "${SOURCE_DIR}/libmessaging.a" "${SOURCE_DIR}/libvisionipc.a")

include_directories(
    src/include
    "${CEREAL_INCLUDE_DIRS}"
    "${CEREAL_INCLUDE_DIRS}/.."
)

#add_executable(camerad src/realtime/camerad.cc)
#add_executable(encoderd src/realtime/encoderd.cc src/realtime/nvvisionbuf.cc)

add_executable(loggerd src/realtime/loggerd.cc ${CAPNP_SRCS})
target_link_libraries(
    loggerd 
    PRIVATE 
        stdc++fs
        fmt::fmt
        ${CEREAL_LIBRARIES}
        zmq
        CapnProto::capnp
)