#include <string>
#include <fstream>
#include <memory>
#include <experimental/filesystem>

#include <argparse/argparse.hpp>
#include <fmt/core.h>
#include <fmt/chrono.h>

#include <NvInfer.h>

#include "braind/buffers.h"
#include "braind/logger.h"

#include "config.h"

namespace fs = std::experimental::filesystem;

const fs::path model_path{MODEL_STORAGE_PATH};

Logger m_logger;

std::unique_ptr<nvinfer1::ICudaEngine> load_engine(const std::string &engine_path)
{
  std::ifstream engine_file(engine_path, std::ios::binary);
  if (!engine_file)
  {
    throw std::runtime_error(fmt::format("Failed to open engine file: {}", engine_path));
  }

  engine_file.seekg(0, engine_file.end);
  const auto engine_size = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);

  std::vector<char> engine_data(engine_size);
  engine_file.read(engine_data.data(), engine_size);

  std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(m_logger)};
  return std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), engine_size));
}

std::unique_ptr<nvinfer1::ICudaEngine> prepare_engine(const std::string &model_full_name)
{
  const fs::path vision_onnx_path{model_path / fmt::format("{}.engine", model_full_name)};
  auto engine = load_engine(vision_onnx_path.string());

  // Verify that inputs and outputs match the references


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