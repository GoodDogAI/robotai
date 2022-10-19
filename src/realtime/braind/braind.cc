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
#include "braind/formatters.hpp"

#include "util.h"
#include "config.h"

#include "msgvec.h"

#include "cereal/services.h"
#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_client.h"

namespace fs = std::experimental::filesystem;
using json = nlohmann::json;

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


static void msgvec_reader(kj::MutexGuarded<MsgVec> &msgvec_guard) {
  // Connect to all of the message queues to receive data to input into the RL model
  std::unique_ptr<Context> ctx{ Context::create() };
  std::unique_ptr<Poller> poller{ Poller::create() };

  // Register all sockets
  std::unordered_map<std::unique_ptr<SubSocket>, std::string> socks;

  for (const auto& it : services) {
    if (!it.should_log || strcmp(it.name, "brainCommands") == 0 || strcmp(it.name, "brainValidation") == 0)
        continue;

    fmt::print("braind {} (on port {})\n", it.name, it.port);

    auto sock = std::unique_ptr<SubSocket> {SubSocket::create(ctx.get(), it.name)};
    KJ_ASSERT(sock != NULL);

    poller->registerSocket(sock.get());
    socks.insert(std::make_pair(std::move(sock), it.name));
  }

  while (!do_exit) {
    for (auto sock : poller->poll(1000)) {
      auto msg = std::unique_ptr<Message>(sock->receive(true));
      if (msg == nullptr) {
          continue;
      }

      //std::cout <<  socks[sock] << " size" << msg->getSize() << std::endl;

      capnp::FlatArrayMessageReader cmsg(kj::ArrayPtr<capnp::word>((capnp::word *)msg->getData(), msg->getSize() / sizeof(capnp::word)));
      auto event = cmsg.getRoot<cereal::Event>();

      auto msgvec = msgvec_guard.lockExclusive();
      msgvec->input(event);
    }
  }

}

void send_model_inference_msg(PubMaster &pm, int32_t frame_id) {
    MessageBuilder msg;
    auto event = msg.initEvent(true);
    auto mdat = event.initModelInference();

    //TODO enable it
    //mdat.setModelFullName(args.get<std::string>("brain_model"));
    mdat.setFrameId(frame_id);

    auto words = capnp::messageToFlatArray(msg);
    auto bytes = words.asBytes();
    pm.send("brainValidation", bytes.begin(), bytes.size());
}

int main(int argc, char *argv[])
{
  argparse::ArgumentParser args("braind");

  args.add_argument("--config")
      .help("path to config json file")
      .required();

  args.add_argument("--vision_model")
      .help("fullname of the intermediate vision model to use")
      .required();  

  args.add_argument("--brain_model")
      .help("fullname of the brain actor model to use")
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
  if (!config_ifs.is_open()) {
    fmt::print(stderr, "Error opening config file {}\n", args.get<std::string>("config"));
    return EXIT_FAILURE;
  }

  json brain_config { json::parse(config_ifs) };
  KJ_ASSERT(brain_config.is_object());
  KJ_ASSERT(brain_config["type"] == "brain");

  kj::MutexGuarded<MsgVec> msgvec_guard { brain_config["msgvec"].dump(), MsgVec::MessageTimingMode::REALTIME };

  VisionIpcClient vipc_client { "camerad", VISION_STREAM_HEAD_COLOR, false };
  std::thread msgvec_thread { &msgvec_reader, std::ref(msgvec_guard) };
  PubMaster pm { {"brainValidation", "brainCommands"} };
  size_t last_10_sec_msgs { 0 };
  auto last_10_sec_time { std::chrono::steady_clock::now() };
  auto vision_engine = prepare_engine(args.get<std::string>("vision_model"));
  auto brain_engine = prepare_engine(args.get<std::string>("brain_model"));
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
  const auto start_sync_time = std::chrono::steady_clock::now();

  while (!vision_ready || !msgvec_ready) {
    VisionIpcBufExtra extra;

    //half a frame timeout, so if there are no pending frames, we can exit
    VisionBuf* buf = vipc_client.recv(&extra, (1000/CAMERA_FPS) / 2);
    if (buf == nullptr) {
        vision_ready = true;
    }

    auto msgvec = msgvec_guard.lockExclusive();
    std::vector<float> obs(msgvec->obs_size());
    auto timeout_res = msgvec->get_obs_vector(obs.data());
    msgvec_ready = timeout_res != MsgVec::TimeoutResult::MESSAGES_NOT_READY;
    msgvec_obs_result = timeout_res;

    if (std::chrono::steady_clock::now() - start_sync_time > std::chrono::seconds(5)) {
        fmt::print(stderr, "Failed to sync vision and msgvec, check timeout values in MsgVec configuration\n");
        return EXIT_FAILURE;
    }
  }

  fmt::print("Vision and msgvec ready, starting inference\n");
  fmt::print("-------------------------------------------\n");

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

    // Once you get a frame, we lock msgvec until that frame is completed
    auto msgvec = msgvec_guard.lockExclusive();
    const auto cur_time = std::chrono::steady_clock::now();

    // Copy and convert from vision ipc to float inputs in range of [16.0, 235.0]
    for (size_t i = 0; i < buf->width * buf->height; i++) {
        host_y[i] = buf->y[i];
    }

    for (size_t i = 0; i < buf->width * buf->height / 2; i++) {
        host_uv[i] = buf->uv[i];
    }

    vision_engine->copy_input_to_device();
    vision_engine->infer();
    vision_engine->copy_output_to_host();
    vision_engine->sync();

    const auto vision_inference_completed_time = std::chrono::steady_clock::now();

    // Send a model inference message to indicate that inference was performed and this moment should be included in the training data
    send_model_inference_msg(pm, extra.frame_id);

    msgvec->input_vision(static_cast<const float*>(vision_engine->get_host_buffer("intermediate")), extra.frame_id);
    auto timeout_res = msgvec->get_obs_vector(static_cast<float*>(brain_engine->get_host_buffer("observation")));

    if (timeout_res == MsgVec::TimeoutResult::MESSAGES_NOT_READY) {
      throw std::runtime_error("msgvec completely lost ready state");
    }
    else if (timeout_res == MsgVec::TimeoutResult::MESSAGES_PARTIALLY_READY && msgvec_obs_result == MsgVec::TimeoutResult::MESSAGES_ALL_READY) {
      throw std::runtime_error("msgvec partially lost ready state");
    }
    msgvec_obs_result = timeout_res;

    brain_engine->copy_input_to_device();
    brain_engine->infer();
    brain_engine->copy_output_to_host();
    brain_engine->sync();
    
    auto messages = msgvec->get_action_command(static_cast<const float*>(brain_engine->get_host_buffer("action")));

    for (auto &msgdata : messages) {
        auto bytes = msgdata.asBytes();
        pm.send("brainCommands", bytes.begin(), bytes.size());
    }

    const auto brain_inference_completed_time = std::chrono::steady_clock::now();

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
        pm.send("brainValidation", bytes.begin(), bytes.size());

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
        pm.send("brainValidation", bytes.begin(), bytes.size());
    }

    // Basic status log every 10 seconds
    if (cur_time - last_10_sec_time > std::chrono::seconds(10)) {
        const auto vision_inference_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(vision_inference_completed_time - cur_time);
        const auto brain_inference_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(brain_inference_completed_time - vision_inference_completed_time);

        fmt::print("braind {:1.1f} frames/sec, inference time {}v + {}b\n", last_10_sec_msgs / 10.0f, vision_inference_elapsed, brain_inference_elapsed);
        last_10_sec_msgs = 0;
        last_10_sec_time = cur_time;
    }
    
    last_10_sec_msgs++;
  }

  msgvec_thread.join();
  return EXIT_SUCCESS;
}