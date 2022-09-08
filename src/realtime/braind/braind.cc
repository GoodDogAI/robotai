#include <string>
#include <fstream>
#include <memory>
#include <utility>
#include <chrono>
#include <experimental/filesystem>

#include <argparse/argparse.hpp>
#include <fmt/core.h>
#include <fmt/chrono.h>
#include <fmt/ranges.h>

#include <npy/npy.h>
#include <npy/tensor.h>

#include <NvInfer.h>

#include "braind/trtwrapper.hpp"

#include "util.h"
#include "config.h"

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"

namespace fs = std::experimental::filesystem;

const fs::path model_path{MODEL_STORAGE_PATH};

ExitHandler do_exit;

std::unique_ptr<TrtEngine> prepare_engine(const std::string &model_full_name)
{
  const fs::path vision_onnx_path {model_path / fmt::format("{}.engine", model_full_name)};
  auto engine = std::make_unique<TrtEngine>(vision_onnx_path.string());

  for (auto &input_name : engine->get_input_names())
  {
    const fs::path npy_path {model_path / fmt::format("{}-input-{}.npy", model_full_name, input_name)};
    auto tensor = npy::load<float, npy::tensor>(npy_path.string());
    fmt::print("Loaded input tensor {} with shape {}\n", input_name, tensor.shape());

    std::copy(tensor.begin(), tensor.end(), static_cast<float *>(engine->get_host_buffer(input_name)));
  }

  engine->copy_input_to_device();
  engine->infer();
  engine->copy_output_to_host();

  fmt::print("Finished sample inference run\n");

  for (auto &output_name : engine->get_output_names())
  {
    const fs::path npy_path {model_path / fmt::format("{}-output-{}.npy", model_full_name, output_name)};
    auto ref_tensor = npy::load<float, npy::tensor>(npy_path.string());
    fmt::print("Loaded output tensor {} with shape {}\n", output_name, ref_tensor.shape());

    auto output_tensor = npy::tensor<float>(ref_tensor.shape());
    output_tensor.copy_from(static_cast<float *>(engine->get_host_buffer(output_name)), output_tensor.size());

    // Compare output and ref tensors
    bool matches = true;
    for (auto i = std::make_pair(ref_tensor.begin(), output_tensor.begin());
        i.first != ref_tensor.end();
        ++i.first, ++i.second)
    {
        if (std::abs(*i.first - *i.second) > 1e-4 + 1e-4 * std::abs(*i.first)) {
          matches = false;
          break;
        }
    }

    fmt::print("Output tensor {} matches: {}\n", output_name, matches);

    if (!matches) {
      throw std::runtime_error("Output tensor does not match reference");
    }
  }

  return engine;
}

int main(int argc, char *argv[])
{
  argparse::ArgumentParser args("braind");

  args.add_argument("--vision_model")
      .help("fullname of the vision model to use")
      .required();

  try
  {
    args.parse_args(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    fmt::print(stderr, "Error parsing arguments {}\n", err.what());
    return EXIT_FAILURE;
  }

  VisionIpcClient vipc_client { "camerad", VISION_STREAM_HEAD_COLOR, false };
  size_t last_10_sec_msgs { 0 };
  //auto last_10_sec_time { std::chrono::steady_clock::now() };
  auto vision_engine = prepare_engine(args.get<std::string>("vision_model"));
  

  // Connect to the visionipc server
  while (!do_exit) {
    if (!vipc_client.connect(false)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
    }
    else {
        std::cout << "Connected to visionipc" << std::endl;
        break;
    }
  }

  // Perform the brain function
  while (!do_exit) {
    VisionIpcBufExtra extra;
    VisionBuf* buf = vipc_client.recv(&extra);
    if (buf == nullptr)
        continue;

    //TODO Copy the image to the engine's input buffer and convert from YUV to RGB
    const auto cur_time = std::chrono::steady_clock::now();
    vision_engine->copy_input_to_device();
    vision_engine->infer();

    // TODO Synchronize the cuda stream?
    fmt::print("infer took {}\n", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - cur_time));

    // const auto cur_time = std::chrono::steady_clock::now();
    // if (cur_time - last_10_sec_time > std::chrono::seconds(10)) {
    //     fmt::print("braind {:1.1f} frames/sec\n", last_10_sec_msgs / 10.0f);
    //     last_10_sec_msgs = 0;
    //     last_10_sec_time = cur_time;
    // }
    
    last_10_sec_msgs++;
  }


  return EXIT_SUCCESS;
}