cmake_minimum_required(VERSION 3.10)
include(CMakePrintHelpers)

project(RobotAI LANGUAGES CXX)

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

FetchContent_Declare(VideoCodecSDK
    URL https://storage.googleapis.com/robotai_public/Video_Codec_SDK_11.1.5.zip
)
FetchContent_MakeAvailable(VideoCodecSDK)

# Set the VIDEO_CODEC_SDK_DIR to the one that we just downloaded
FetchContent_GetProperties(VideoCodecSDK SOURCE_DIR VIDEO_CODEC_SDK_DIR)
set(GENERATE_PYTHON_BINDINGS 1)

# TODO Enabling this would need to override some install locations
#set(GENERATE_PYTORCH_EXTENSION 1)

FetchContent_Declare(vpf
        GIT_REPOSITORY https://github.com/NVIDIA/VideoProcessingFramework.git
        GIT_TAG v1.1.1
        )
FetchContent_MakeAvailable(vpf)

# Find any dependencies that must be installed on the system
find_package(CUDAToolkit)
find_package(CapnProto REQUIRED)

# TODO: This cannot be found, even though modern versions of ZMQ should provide a CMakeLists.txt
# find_package(ZeroMQ REQUIRED)

find_package(realsense2 QUIET)

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

set(JETSON_MULTIMEDIA_API_INCLUDE_DIRS "/usr/src/jetson_multimedia_api/include")


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

if(EXISTS ${JETSON_MULTIMEDIA_API_INCLUDE_DIRS})
    add_executable(encoderd src/realtime/encoderd.cc src/realtime/nvvisionbuf.cc)
    target_include_directories(${JETSON_MULTIMEDIA_API_INCLUDE_DIRS})
    target_link_libraries(
        encoderd 
        PRIVATE 
            fmt::fmt
            ${CEREAL_LIBRARIES}
            zmq
            CapnProto::capnp
    )
endif()

if(${REALSENSE2_FOUND})
    add_executable(camerad src/realtime/camerad.cc)
endif()
