#pragma once

#include <NvInfer.h>

#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

#include "braind/logger.hpp"

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

template <typename A, typename B>
inline A divUp(A m, B n)
{
    return (m + n - 1) / n;
}

class TrtEngine {
    public:
    TrtEngine(const std::string &engine_path) 
        :m_logger(), m_runtime(nvinfer1::createInferRuntime(m_logger)) {
            // Load the engine from file
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

            // Deserialize the engine
            m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engine_data.data(), engine_size));
            if (!m_engine)
            {
                throw std::runtime_error("Failed to create TensorRT engine");
            }

            m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
            if (!m_context)
            {
                throw std::runtime_error("Failed to create TensorRT execution context");
            }

            // Allocate memory for the input and output tensors
    }

    private:
        Logger m_logger;
        std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
        std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

};