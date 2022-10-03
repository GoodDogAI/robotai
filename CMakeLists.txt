cmake_minimum_required(VERSION 3.20)
include(CMakePrintHelpers)

project(RobotAI LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Locate CUDA, and configure the CUDA COMPILER location manually
# as a prerequisite for building the Nvidia VPF library
find_package(CUDAToolkit)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

# Locate Python
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python3 3.8 COMPONENTS Interpreter Development REQUIRED)
message("Using Python from ${Python3_INCLUDE_DIRS}")


# Set the default location for the jetson multimedia sdk
set(JETSON_MULTIMEDIA_API_INCLUDE_DIRS "/usr/src/jetson_multimedia_api/include")

# Enable Clang-tidy for static analysis
# Note, I found that it wasn't necessarily useful, too many warnings
# find_program(CLANG_TIDY_EXE NAMES clang-tidy clang-tidy-10 REQUIRED)
# set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}" -checks=cppcoreguidelines-*)

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

FetchContent_Declare(argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse
        GIT_TAG v2.6 
        )
FetchContent_MakeAvailable(argparse)


FetchContent_Declare(npy
        GIT_REPOSITORY https://github.com/matajoh/libnpy
        GIT_TAG v1.5.2 
        )
FetchContent_MakeAvailable(npy)

FetchContent_Declare(json 
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
        )
FetchContent_MakeAvailable(json)

# If we are not on the Jetson platform, then we need to build the VPF and get TensorRT specially
if(NOT EXISTS ${JETSON_MULTIMEDIA_API_INCLUDE_DIRS})
    FetchContent_Declare(VideoCodecSDK
        URL https://storage.googleapis.com/robotai_public/Video_Codec_SDK_11.1.5.zip
    )
    FetchContent_MakeAvailable(VideoCodecSDK)

    # Set the VIDEO_CODEC_SDK_DIR to the one that we just downloaded
    FetchContent_GetProperties(VideoCodecSDK SOURCE_DIR VIDEO_CODEC_SDK_DIR)
    set(VIDEO_CODEC_SDK_DIR ${VIDEO_CODEC_SDK_DIR} CACHE PATH "NVIDIA SDK Location" FORCE)

    set(GENERATE_PYTHON_BINDINGS 1 CACHE BOOL "Override for generating python bindings for VPF" FORCE)
    set(GENERATE_PYTORCH_EXTENSION 1 CACHE BOOL "Override for generating pytorch bindings for VPF" FORCE)

    FetchContent_Declare(vpf
            GIT_REPOSITORY https://github.com/jakepoz/VideoProcessingFramework.git
            GIT_TAG origin/master
            )
    FetchContent_MakeAvailable(vpf)

    # Install the built python modules into the source tree, so you can run them
    install(FILES $<TARGET_FILE:PyNvCodec> DESTINATION src)
    install(FILES $<TARGET_FILE:TC_CORE> DESTINATION src)
    install(FILES $<TARGET_FILE:TC>	DESTINATION src)

    install(FILES "${vpf_SOURCE_DIR}/PytorchNvCodec/src/PytorchNvCodec.${Python3_SOABI}.so"	DESTINATION src)
endif()


# Find any dependencies that must be installed on the system
find_package(CapnProto 0.7.0 REQUIRED)

# TODO: This cannot be found, even though modern versions of ZMQ should provide a CMakeLists.txt
# find_package(ZeroMQ REQUIRED)

find_package(realsense2 QUIET)
message("REALSENSE Version: ${realsense2_VERSION}")

find_package(ALSA)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavfilter
    libavformat
    libavcodec
    libswresample
    libswscale
    libavutil
)

# Build and configure cereal, as an in-place build
include(ExternalProject)
ExternalProject_Add(cereal 
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/cereal"
    CONFIGURE_COMMAND ""
    BUILD_ALWAYS TRUE # Needs to be set, because you can modify the log.capnp and the other stuff needs to be updated for it
    BUILD_IN_SOURCE TRUE
    BUILD_COMMAND scons -j6
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(cereal SOURCE_DIR)
set(CEREAL_INCLUDE_DIRS ${SOURCE_DIR})
set(CEREAL_LIBRARIES "${SOURCE_DIR}/libcereal.a" "${SOURCE_DIR}/libvisionipc.a" "${SOURCE_DIR}/libmessaging.a")

# Generate the config.h file from python
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/src/config/config.h
    COMMAND ${Python3_EXECUTABLE} -m src.config.generate_config_h
    DEPENDS ${CMAKE_SOURCE_DIR}/src/config/config.py ${CMAKE_SOURCE_DIR}/src/config/generate_config_h.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)    

include_directories(
    src/include
    src/config
    src/msgvec
    "${CEREAL_INCLUDE_DIRS}"
    "${CEREAL_INCLUDE_DIRS}/.."
)

# Make sure warnings are treated as errors
add_compile_options(-Wunused -Werror -Wshadow)

#set(CMAKE_VERBOSE_MAKEFILE ON)

set(MASTER_CONFIG_H "${CMAKE_SOURCE_DIR}/src/config/config.h")

add_executable(loggerd src/realtime/loggerd.cc ${MASTER_CONFIG_H})
target_link_libraries(
    loggerd 
    PRIVATE 
        stdc++fs
        fmt::fmt
        ${CEREAL_LIBRARIES}
        zmq
        CapnProto::capnp
)

add_library(serial src/realtime/serial.cc)
target_link_libraries(serial PUBLIC fmt::fmt)

add_library(msgvec src/msgvec/msgvec.cc)
set_property(TARGET msgvec PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(
    msgvec 
    PUBLIC 
        ${CEREAL_LIBRARIES}
        nlohmann_json::nlohmann_json
        CapnProto::capnp
        CapnProto::capnp-rpc
    )

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/msgvec/_pymsgvec.cpp
    COMMAND ${Python3_EXECUTABLE} -m cython --cplus ${CMAKE_CURRENT_SOURCE_DIR}/src/msgvec/pymsgvec.pyx -o ${CMAKE_CURRENT_SOURCE_DIR}/src/msgvec/_pymsgvec.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/msgvec
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/src/msgvec/msgvec.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/src/msgvec/msgvec.h
      ${CMAKE_CURRENT_SOURCE_DIR}/src/msgvec/pymsgvec.pyx
      ${CMAKE_CURRENT_SOURCE_DIR}/src/msgvec/pymsgvec.pxd
  )

Python3_add_library(pymsgvec MODULE WITH_SOABI src/msgvec/_pymsgvec.cpp)
target_include_directories(pymsgvec PRIVATE ${Python3_SITELIB})
target_link_libraries(pymsgvec PRIVATE msgvec)
# Cython has a bug where it produces code with shadow warnings, so we need to supress them here
target_compile_options(pymsgvec PRIVATE -Wno-shadow -Wno-unused)
set_target_properties(pymsgvec PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/msgvec
)

add_executable(msgvec_tests src/tests/ctest_msgvec.cc)
target_link_libraries(
    msgvec_tests
    PRIVATE 
        msgvec
        ${CEREAL_LIBRARIES}
        nlohmann_json::nlohmann_json
        CapnProto::capnp
        Catch2::Catch2WithMain
)
catch_discover_tests(msgvec_tests)


add_executable(odrived src/realtime/odrived.cc ${MASTER_CONFIG_H})
target_link_libraries(
    odrived 
    PRIVATE
        serial 
        stdc++fs
        fmt::fmt
        ${CEREAL_LIBRARIES}
        zmq
        CapnProto::capnp
)

add_executable(simplebgcd src/realtime/simplebgcd.cc ${MASTER_CONFIG_H})
target_link_libraries(
    simplebgcd 
    PRIVATE
        serial 
        stdc++fs
        fmt::fmt
        ${CEREAL_LIBRARIES}
        zmq
        CapnProto::capnp
)

# Mike D with the master plan
if(${ALSA_FOUND})
    add_executable(micd src/realtime/micd.cc ${MASTER_CONFIG_H})
    target_link_libraries(
        micd
        PRIVATE
            ALSA::ALSA
            fmt::fmt   
            ${CEREAL_LIBRARIES}
            zmq
            CapnProto::capnp
    )
endif()

# TODO Build the brain on host device too, for now we are excluding it due to some weird linker errors
if(EXISTS ${JETSON_MULTIMEDIA_API_INCLUDE_DIRS})
    add_executable(braind src/realtime/braind/braind.cc ${MASTER_CONFIG_H})
    target_include_directories(braind PRIVATE ${CUDAToolkit_INCLUDE_DIRS} src/realtime)
    target_link_libraries(
        braind 
        PRIVATE
            serial 
            stdc++fs
            fmt::fmt
            nvinfer
            argparse::argparse
            npy::npy
            CUDA::cudart
            ${CEREAL_LIBRARIES}
            zmq
            CapnProto::capnp
    )
endif()

if(EXISTS ${JETSON_MULTIMEDIA_API_INCLUDE_DIRS})
    add_executable(encoderd 
        src/realtime/encoderd.cc 
        src/realtime/nvencoder.cc
        src/realtime/nvvisionbuf.cc
        ${MASTER_CONFIG_H}
        )
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
        src/tests/ctest_nvencoder.cc 
        src/realtime/nvencoder.cc 
        src/realtime/nvvisionbuf.cc
        ${MASTER_CONFIG_H}
        )
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
            PkgConfig::LIBAV
    )

    catch_discover_tests(encoderd_tests)
endif()

if(${realsense2_FOUND})
    add_executable(camerad src/realtime/camerad.cc ${MASTER_CONFIG_H})
    target_link_libraries(
        camerad
        PRIVATE
            fmt::fmt
            ${CEREAL_LIBRARIES}
            zmq
            realsense2::realsense2
            CapnProto::capnp)
endif()
