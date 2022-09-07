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
#include "braind/buffers.hpp"


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

            // Create host and device buffers
            for (int i = 0; i < m_engine->getNbBindings(); i++)
            {
                size_t vol = 1;
                nvinfer1::DataType type = m_engine->getBindingDataType(i);
                auto dims = m_context->getBindingDimensions(i);
                
                int vecDim = m_engine->getBindingVectorizedDim(i);
                if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
                {
                    int scalarsPerVec = m_engine->getBindingComponentsPerElement(i);
                    dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
                    vol *= scalarsPerVec;
                }
                vol *= volume(dims);
                std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
                manBuf->deviceBuffer = DeviceBuffer(vol, type);
                manBuf->hostBuffer = HostBuffer(vol, type);
                m_buffers.push_back(std::move(manBuf));
            }
    }

    void* getDeviceBuffer(const std::string& tensorName) const
    {
        int index = m_engine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return nullptr;

        return m_buffers[index]->deviceBuffer.data();
    }

    void* getHostBuffer(const std::string& tensorName) const
    {
        int index = m_engine->getBindingIndex(tensorName.c_str());
        if (index == -1)
            return nullptr;

        return m_buffers[index]->hostBuffer.data();
    }

    void copyInputToDevice()
    {
        memcpyBuffers(true, false, false);
    }

    void copyOutputToHost()
    {
        memcpyBuffers(false, true, false);
    }

    private:
        void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0)
        {
            for (int i = 0; i < m_engine->getNbBindings(); i++)
            {
                void* dstPtr
                        = deviceToHost ? m_buffers[i]->hostBuffer.data() : m_buffers[i]->deviceBuffer.data();
                const void* srcPtr
                        = deviceToHost ? m_buffers[i]->deviceBuffer.data() : m_buffers[i]->hostBuffer.data();
                const size_t byteSize = m_buffers[i]->hostBuffer.nbBytes();
                const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
                if ((copyInput && m_engine->bindingIsInput(i)) || (!copyInput && !m_engine->bindingIsInput(i)))
                {
                    if (async) {
                        auto ret = cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream);

                        if (ret != cudaSuccess) {
                            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
                            abort();
                        }
                    }
                    else {
                        auto ret = cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType);

                        if (ret != cudaSuccess) {
                            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
                            abort();
                        }
                    }
                }
            }
        }

        Logger m_logger;
        std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
        std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

        // Input and output buffers
        std::vector<std::unique_ptr<ManagedBuffer>> m_buffers;
};