#include <string>
#include <fstream>
#include <memory>
#include <experimental/filesystem>

#include <argparse/argparse.hpp>
#include <fmt/core.h>
#include <fmt/chrono.h>

#include <NvInfer.h>

#include "braind/trtwrapper.hpp"

#include "config.h"

namespace fs = std::experimental::filesystem;

const fs::path model_path{MODEL_STORAGE_PATH};


std::unique_ptr<TrtEngine> prepare_engine(const std::string &model_full_name)
{
  const fs::path vision_onnx_path{model_path / fmt::format("{}.engine", model_full_name)};
  auto engine = std::make_unique<TrtEngine>(vision_onnx_path.string());

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