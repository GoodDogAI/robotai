#include <string>
#include <fstream>
#include <memory>
#include <utility>
#include <chrono>
#include <thread>
#include <unordered_set>
#include <experimental/filesystem>

#include <argparse/argparse.hpp>
#include <fmt/core.h>
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <npy/npy.h>
#include <npy/tensor.h>

#include <NvInfer.h>

#include "braind/trtwrapper.hpp"

#include "util.h"
#include "config.h"

#include "msgvec.h"

#include "cereal/services.h"
#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"

namespace fs = std::experimental::filesystem;

const fs::path model_path{MODEL_STORAGE_PATH};
const char *validation_service_name = "brainValidation";

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
  engine->sync();

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

static void close_message(Message *m) {
    m->close();
}

static void msgvec_reader(MsgVec &msgvec) {
  // Connect to all of the message queues to receive data to input into the RL model
  std::unique_ptr<Context> ctx{ Context::create() };
  std::unique_ptr<Poller> poller{ Poller::create() };

   // Register all sockets
  std::unordered_set<std::unique_ptr<SubSocket>> socks;

  for (const auto& it : services) {
    if (!it.should_log || strcmp(it.name, "braind") == 0)
        continue;

    auto sock = std::unique_ptr<SubSocket> { SubSocket::create(ctx.get(), it.name) };
    assert(sock != NULL);

    fmt::print("brain {} (on port {})\n", it.name, it.port);

    poller->registerSocket(sock.get());
    socks.insert(std::move(sock));
  }

  while (!do_exit) {
    for (auto sock : poller->poll(1000)) {
      auto msg = std::unique_ptr<Message, std::function<void(Message*)>>(sock->receive(true), close_message);

      capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize()));
      auto event = cmsg.getRoot<cereal::Event>();

      msgvec.input(event);
    }
  }
}

int main(int argc, char *argv[])
{
  argparse::ArgumentParser args("braind");

  args.add_argument("--config")
      .help("path to config json file")
      .required();

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

  std::ifstream config_ifs { args.get<std::string>("config") };
  json config { json::parse(config_ifs) };
  MsgVec msgvec { config["msgvec"].dump() };
  VisionIpcClient vipc_client { "camerad", VISION_STREAM_HEAD_COLOR, false };
  std::thread msgvec_thread { &msgvec_reader, std::ref(msgvec) };
  PubMaster pm { {validation_service_name} };
  size_t last_10_sec_msgs { 0 };
  auto last_10_sec_time { std::chrono::steady_clock::now() };
  auto vision_engine = prepare_engine(args.get<std::string>("vision_model"));
  MsgVec::TimeoutResult msgvec_obs_result { MsgVec::TimeoutResult::MESSAGES_NOT_READY };

  // Make sure the vision engine inputs and outputs are setup as we expect them
  if (vision_engine->get_tensor_dtype("y") != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error("Vision model output tensor y is not of type float");
  }
  if (vision_engine->get_tensor_dtype("uv") != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error("Vision model output tensor y is not of type float");
  }
  if (vision_engine->get_tensor_shape("y") != std::vector{1, 1, CAMERA_HEIGHT, CAMERA_WIDTH}) {
    throw std::runtime_error("Vision model output tensor y is not of expected size");
  }
  if (vision_engine->get_tensor_shape("uv") != std::vector{1, 1, CAMERA_HEIGHT / 2, CAMERA_WIDTH}) {
    throw std::runtime_error("Vision model output tensor y is not of expected size");
  }

  // Connect to the visionipc server
  while (!do_exit) {
    if (!vipc_client.connect(false)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
    }
    else {
        fmt::print("Connected to visionipc\n");
        break;
    }
  }
  


  // Receive all stale frames from visionipc, and wait for the msgvec obs vector to become ready
  bool vision_ready = false, msgvec_ready = false;
  while (!vision_ready || !msgvec_ready) {
    VisionIpcBufExtra extra;

    //half a frame timeout, so if there are no pending frames, we can exit
    VisionBuf* buf = vipc_client.recv(&extra, (1000/CAMERA_FPS) / 2);
    if (buf == nullptr) {
        vision_ready = true;
    }

    auto timeout_res = msgvec.get_obs_vector(nullptr);
    msgvec_ready = timeout_res != MsgVec::TimeoutResult::MESSAGES_NOT_READY;
    msgvec_obs_result = timeout_res;
  }


  float *host_y = static_cast<float*>(vision_engine->get_host_buffer("y"));
  float *host_uv = static_cast<float*>(vision_engine->get_host_buffer("uv"));
  float *host_intermediate = static_cast<float*>(vision_engine->get_host_buffer("intermediate"));
  auto intermediate_shape = vision_engine->get_tensor_shape("intermediate");

  // Perform the brain function
  while (!do_exit) {
    VisionIpcBufExtra extra;
    VisionBuf* buf = vipc_client.recv(&extra);
    if (buf == nullptr)
        continue;

    if (buf->width != CAMERA_WIDTH|| buf->stride != CAMERA_WIDTH || buf->height != CAMERA_HEIGHT) {
        std::cout << "Invalid frame size" << std::endl;
        fmt::print(stderr, "Invalid frame size\n");
        return EXIT_FAILURE;
    }

    const auto cur_time = std::chrono::steady_clock::now();

    // Copy and convert from vision ipc to float inputs in range of [16.0, 235.0]
    for (size_t i = 0; i < buf->width * buf->height; i++) {
        host_y[i] = buf->y[i];
    }

    for (size_t i = 0; i < buf->width * buf->height / 2; i++) {
        host_uv[i] = buf->uv[i];
    }

    std::vector<float> obs(msgvec.obs_size());
    auto timeout_res = msgvec.get_obs_vector(obs.data());

    if (timeout_res == MsgVec::TimeoutResult::MESSAGES_NOT_READY) {
      throw std::runtime_error("msgvec lost ready state");
    }
    else if (timeout_res == MsgVec::TimeoutResult::MESSAGES_PARTIALLY_READY && msgvec_obs_result == MsgVec::TimeoutResult::MESSAGES_ALL_READY) {
      throw std::runtime_error("msgvec partially lost ready state");
    }
    msgvec_obs_result = timeout_res;

    vision_engine->copy_input_to_device();
    vision_engine->infer();
    vision_engine->copy_output_to_host();
    vision_engine->sync();

    const auto inference_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - cur_time);

    // Log every N frames with a model validation message
    if (extra.frame_id % 60 == 0) {
        MessageBuilder msg;
        auto event = msg.initEvent(true);
        auto mdat = event.initModelValidation();
        mdat.setModelType(cereal::ModelValidation::ModelType::VISION_INTERMEDIATE);
        mdat.setModelFullName(args.get<std::string>("vision_model"));
        mdat.setFrameId(extra.frame_id);
        mdat.setTensorName("intermediate");
        mdat.setShape(kj::arrayPtr(intermediate_shape.data(), intermediate_shape.size()));
        mdat.setData(kj::ArrayPtr<float>(host_intermediate, host_intermediate + vision_engine->get_tensor_size("intermediate")));
        
        auto words = capnp::messageToFlatArray(msg);
        auto bytes = words.asBytes();
        pm.send(validation_service_name, bytes.begin(), bytes.size());

        // Log the actual input frame too, because there are some issues
        event = msg.initEvent(true);
        mdat = event.initModelValidation();
        mdat.setModelType(cereal::ModelValidation::ModelType::VISION_INPUT);
        mdat.setModelFullName(args.get<std::string>("vision_model"));
        mdat.setFrameId(extra.frame_id);
        mdat.setTensorName("y_slice");
        auto yshape = std::vector<int32_t>{1, 1, 2, CAMERA_WIDTH};
        mdat.setShape(kj::arrayPtr(yshape.data(), yshape.size()));

        mdat.setData(kj::ArrayPtr<float>(host_y, host_y + CAMERA_WIDTH * 2));
        
        words = capnp::messageToFlatArray(msg);
        bytes = words.asBytes();
        pm.send(validation_service_name, bytes.begin(), bytes.size());
    }

    // Basic status log every 10 seconds
    if (cur_time - last_10_sec_time > std::chrono::seconds(10)) {
        fmt::print("braind {:1.1f} frames/sec, inference time {}\n", last_10_sec_msgs / 10.0f, inference_elapsed);
        last_10_sec_msgs = 0;
        last_10_sec_time = cur_time;
    }
    
    last_10_sec_msgs++;
  }

  msgvec_thread.join();
  return EXIT_SUCCESS;
}