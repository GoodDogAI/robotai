#include <string>
#include <fstream>
#include <memory>
#include <utility>
#include <experimental/filesystem>

#include <argparse/argparse.hpp>
#include <fmt/core.h>
#include <fmt/chrono.h>
#include <fmt/ranges.h>

#include <npy/npy.h>
#include <npy/tensor.h>

#include <NvInfer.h>

#include "braind/trtwrapper.hpp"

#include "config.h"

namespace fs = std::experimental::filesystem;

const fs::path model_path{MODEL_STORAGE_PATH};


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

  auto vision_engine = prepare_engine(args.get<std::string>("vision_model"));

  return EXIT_SUCCESS;
}