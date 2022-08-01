cmake_minimum_required(VERSION 3.20)
include(CMakePrintHelpers)

project(RobotAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Locate CUDA, and configure the CUDA COMPILER location manually
# as a prerequisite for building the Nvidia VPF library
find_package(CUDAToolkit)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

# Locate Python
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter Development)
message("Using Python from ${Python3_INCLUDE_DIRS}")

# Set the default location for the jetson multimedia sdk
set(JETSON_MULTIMEDIA_API_INCLUDE_DIRS "/usr/src/jetson_multimedia_api/include")

# Download and compile some lightweight dependencies
Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.1.0
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 9.0.0
        )
FetchContent_MakeAvailable(fmt)

# If we are not on the Jetson platform, then we need to build the VPF
if(NOT EXISTS ${JETSON_MULTIMEDIA_API_INCLUDE_DIRS})
    FetchContent_Declare(VideoCodecSDK
        URL https://storage.googleapis.com/robotai_public/Video_Codec_SDK_11.1.5.zip
    )
    FetchContent_MakeAvailable(VideoCodecSDK)

    # Set the VIDEO_CODEC_SDK_DIR to the one that we just downloaded
    FetchContent_GetProperties(VideoCodecSDK SOURCE_DIR VIDEO_CODEC_SDK_DIR)
    set(VIDEO_CODEC_SDK_DIR ${VIDEO_CODEC_SDK_DIR} CACHE PATH "NVIDIA SDK Location" FORCE)

    set(GENERATE_PYTHON_BINDINGS 1 CACHE BOOL "Override for generating python bindings for VPF" FORCE)
    # TODO Enabling this would need to override some install locations
    #set(GENERATE_PYTORCH_EXTENSION 1)

    FetchContent_Declare(vpf
            GIT_REPOSITORY https://github.com/jakepoz/VideoProcessingFramework.git
            GIT_TAG origin/pybind_cmake
            )
    FetchContent_MakeAvailable(vpf)

    # Install the built python modules into the source tree, so you can run them
    install(FILES $<TARGET_FILE:PyNvCodec> DESTINATION src)
    install(FILES $<TARGET_FILE:TC_CORE> DESTINATION src)
    install(FILES $<TARGET_FILE:TC>	DESTINATION src)
endif()


# Find any dependencies that must be installed on the system
find_package(CapnProto 0.8.0 REQUIRED)

# TODO: This cannot be found, even though modern versions of ZMQ should provide a CMakeLists.txt
# find_package(ZeroMQ REQUIRED)

find_package(realsense2 QUIET)
message("REALSENSE Version: ${realsense2_VERSION}")

find_package(ALSA)

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
set(CEREAL_LIBRARIES "${SOURCE_DIR}/libcereal.a" "${SOURCE_DIR}/libvisionipc.a" "${SOURCE_DIR}/libmessaging.a")

include_directories(
    src/include
    "${CEREAL_INCLUDE_DIRS}"
    "${CEREAL_INCLUDE_DIRS}/.."
)

add_compile_options(-Wunused -Werror -Wshadow)

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

# Mike D with the master plan
add_executable(micd src/realtime/micd.cc)
target_link_libraries(
    micd
    PRIVATE
        ALSA::ALSA
        fmt::fmt   
        ${CEREAL_LIBRARIES}
        zmq
        CapnProto::capnp
)

if(EXISTS ${JETSON_MULTIMEDIA_API_INCLUDE_DIRS})
    add_executable(encoderd 
        src/realtime/encoderd.cc 
        src/realtime/nvencoder.cc
        src/realtime/nvvisionbuf.cc)
    target_include_directories(encoderd PRIVATE ${JETSON_MULTIMEDIA_API_INCLUDE_DIRS})
    target_link_libraries(
        encoderd 
        PRIVATE 
            fmt::fmt
            ${CEREAL_LIBRARIES}
            zmq
            v4l2
            CapnProto::capnp
    )

    add_executable(encoderd_tests 
        src/tests/test_nvencoder.cc 
        src/realtime/nvencoder.cc 
        src/realtime/nvvisionbuf.cc)
    target_include_directories(encoderd_tests PRIVATE ${JETSON_MULTIMEDIA_API_INCLUDE_DIRS} src/realtime)
    target_link_libraries(
        encoderd_tests
        PRIVATE 
            fmt::fmt
            ${CEREAL_LIBRARIES}
            zmq
            v4l2
            CapnProto::capnp
            Catch2::Catch2WithMain
    )
    catch_discover_tests(encoderd_tests)
endif()

if(${realsense2_FOUND})
    add_executable(camerad src/realtime/camerad.cc)
    target_link_libraries(
        camerad
        PRIVATE
            fmt::fmt
            ${CEREAL_LIBRARIES}
            zmq
            realsense2::realsense2
            CapnProto::capnp)
endif()
