/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2021-2024, Massachusetts Institute of Technology.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * * -------------------------------------------------------------------------- */

#include "model.h"

#include <config_utilities/validation.h>
#include <cuda_runtime_api.h>

#include "semantic_inference/logging.h"

namespace semantic_inference {

struct DimInfo {
  bool is_image = false;
  bool is_color = false;
  int start = 0;
};

DimInfo getDimInfo(const nvinfer1::Dims& dims) {
  //SLOG(INFO) << "getDimInfo called with dims.nbDims=" << dims.nbDims;
  if (dims.nbDims < 3) {
    SLOG(WARNING) << "dims.nbDims < 3, returning empty DimInfo";
    return {};
  }

  const auto start = dims.nbDims - 3;
  const auto end = dims.nbDims - 1;
  //SLOG(INFO) << "start=" << start << ", end=" << end;
  const bool is_color = dims.d[start] == 3 || dims.d[end] == 3;
  //SLOG(INFO) << "is_color=" << is_color;
  return {true, is_color, start};
}

bool areDimsCHWOrder(const nvinfer1::Dims& dims) {
  //SLOG(INFO) << "areDimsCHWOrder called with dims=" << toString(dims);
  const auto info = getDimInfo(dims);
  if (!info.is_image) {
    SLOG(ERROR) << "Invalid tensor for image method";
    throw std::runtime_error("invalid tensor for image method");
  }

  const int expected_channels = info.is_color ? 3 : 1;
  //SLOG(INFO) << "expected_channels=" << expected_channels << ", actual=" << dims.d[info.start];
  return dims.d[info.start] == expected_channels;
}

Shape getShapeFromDims(const nvinfer1::Dims& dims) {
  const auto info = getDimInfo(dims);
  if (dims.nbDims < 2) {
    // TODO(nathan) fix
    // SLOG(ERROR) << "invalid tensor: " << *this;
    throw std::runtime_error("unsupported layout!");
  }

  Shape shape;
  if (!info.is_image) {
    shape.height = dims.d[0];
    shape.width = dims.d[1];
    return shape;
  }

  shape.chw_order = areDimsCHWOrder(dims);
  if (shape.chw_order) {
    shape.height = dims.d[info.start + 1];
    shape.width = dims.d[info.start + 2];
    shape.channels = dims.d[info.start];
  } else {
    shape.height = dims.d[info.start];
    shape.width = dims.d[info.start + 1];
    shape.channels = dims.d[info.start + 2];
  }

  return shape;
}

bool TensorInfo::isCHWOrder() const { return areDimsCHWOrder(dims); }

bool TensorInfo::isDynamic() const {
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] == -1) {
      return true;
    }
  }
  return false;
}

nvinfer1::Dims TensorInfo::replaceDynamic(const cv::Mat& mat) const {
  nvinfer1::Dims new_dims = dims;
  const auto info = getDimInfo(dims);
  const auto chw_order = isCHWOrder();
  size_t h_index = chw_order ? info.start + 1 : info.start;
  size_t w_index = chw_order ? info.start + 2 : info.start + 1;
  new_dims.d[h_index] = new_dims.d[h_index] < 0 ? mat.rows : new_dims.d[h_index];
  new_dims.d[w_index] = new_dims.d[w_index] < 0 ? mat.cols : new_dims.d[w_index];
  return new_dims;
}

Shape TensorInfo::shape() const { return getShapeFromDims(dims); }

std::ostream& operator<<(std::ostream& out, const TensorInfo& info) {
  out << "<name=" << info.name << ", layout=" << toString(info.dims) << " ("
      << toString(info.dtype) << ")>";
  return out;
}

ModelInfo::ModelInfo() {}

ModelInfo::ModelInfo(const nvinfer1::ICudaEngine& engine) {
  auto num_tensors = engine.getNbIOTensors();
  for (int i = 0; i < num_tensors; ++i) {
    const auto tname = engine.getIOTensorName(i);
    const auto tmode = engine.getTensorIOMode(tname);
    if (tmode == nvinfer1::TensorIOMode::kNONE) {
      continue;
    }

    const auto dims = engine.getTensorShape(tname);
    const auto dtype = engine.getTensorDataType(tname);
    TensorInfo info{tname, dims, dtype};
    if (tmode == nvinfer1::TensorIOMode::kOUTPUT) {
      if (!setIfUnset(info, labels_)) {
        SLOG(ERROR) << "Multiple outputs detected! Rejecting " << tname;
      } else {
        SLOG(DEBUG) << "Set " << info << " for labels";
      }

      continue;
    }

    assert(tmode == nvinfer1::TensorIOMode::kINPUT);
    const auto dim_info = getDimInfo(dims);
    if (!dim_info.is_image) {
      SLOG(WARNING) << "Found input tensor with invalid layout: " << info;
      continue;
    }

    if (dim_info.is_color) {
      if (!setIfUnset(info, color_)) {
        SLOG(ERROR) << "Multiple color inputs detect! Rejecting " << tname;
      } else {
        SLOG(DEBUG) << "Set " << info << " for color!";
      }
    } else {
      if (!setIfUnset(info, depth_)) {
        SLOG(ERROR) << "Multiple depth inputs detect! Rejecting " << tname;
      } else {
        SLOG(DEBUG) << "Set " << info << " for depth!";
      }
    }
  }
}

bool ModelInfo::setIfUnset(const TensorInfo& info, std::optional<TensorInfo>& field) {
  if (!field) {
    field = info;
    return true;
  }

  return false;
}

std::ostream& operator<<(std::ostream& out, const ModelInfo& info) {
  out << "Model: ";
  if (!info) {
    out << "(uninitialized)";
    return out;
  }

  out << "color=" << info.color();

  out << ", depth=";
  const auto& depth = info.depth();
  if (depth) {
    out << depth.value();
  } else {
    out << "n/a";
  }

  out << ", labels=" << info.labels();
  return out;
}

Model::Model(const ModelConfig& config)
    : model(config::checkValid(config)),
      runtime_(getRuntime(model.log_severity)),
      engine_(deserializeEngine(*runtime_, model.engine_file)),
      color_conversion_(config.color),
      depth_conversion_(config.depth) {
  if (!engine_ || config.force_rebuild) {
    SLOG(WARNING) << "Engine file not found! rebuilding...";
    engine_ = buildEngineFromOnnx(
        *runtime_, model.model_file, model.engine_file, model.log_severity);
    SLOG(INFO) << "Finished building engine";
  } else {
    SLOG(INFO) << "Loaded engine file";
  }

  if (!engine_) {
    SLOG(ERROR) << "Building engine from onnx failed!";
    throw std::runtime_error("failed to load or build engine");
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    SLOG(ERROR) << "Failed to create execution context";
    throw std::runtime_error("failed to set up trt context");
  }

  SLOG(INFO) << "Execution context started";

  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    SLOG(ERROR) << "Creating cuda stream failed!";
    throw std::runtime_error("failed to set up cuda stream");
  } else {
    SLOG(INFO) << "CUDA stream started";
  }

  initialized_ = true;

  info_ = ModelInfo(*engine_);
  SLOG(INFO) << info_;
  if (!info_) {
    SLOG(ERROR) << "Invalid engine for segmentation!";
    throw std::runtime_error("invalid model");
  }

  if (info_.color().dtype != nvinfer1::DataType::kFLOAT) {
    SLOG(ERROR) << "Input type mismatch: " << info_.color() << ", must be FLOAT";
    throw std::runtime_error("invalid model");
  }

  const auto depth = info_.depth();
  if (depth && depth.value().dtype != nvinfer1::DataType::kFLOAT) {
    SLOG(ERROR) << "Depth type mismatch: " << *depth << ", must be FLOAT";
    throw std::runtime_error("invalid model");
  }
}

Model::~Model() {
  if (initialized_) {
    cudaStreamDestroy(stream_);
  }
}

void Model::initOutput(const cv::Mat& color) {
  //SLOG(INFO) << "Initializing output for color image: " << color.size();
  if (context_->inferShapes(0, nullptr)) {
    SLOG(ERROR) << "Invalid shapes!";
    throw std::runtime_error("could not infer output shape");
  }

  const auto info = info_.labels();
  const auto tensor_name = info.name.c_str();
  //SLOG(INFO) << "Output tensor name: " << tensor_name;
  if (info.dtype != nvinfer1::DataType::kINT32) {
    SLOG(FATAL) << "Unhandled type: " << toString(info.dtype);
    throw std::runtime_error("output datatype is forced to be int32_t");
  }

  const auto shape = info.shape().updateFrom(color);
  //SLOG(INFO) << "Allocating output memory with shape: " << shape.width << "x" << shape.height;
  auto size = sizeof(int32_t) * shape.width * shape.height * shape.channels.value_or(1);
  //SLOG(INFO) << "Allocating " << size << " bytes for output";
  label_memory_.reset(reinterpret_cast<int32_t*>(CudaMemoryManager::alloc(size)));
  if (!label_memory_) {
    SLOG(ERROR) << "Failed to allocate output memory!";
    throw std::runtime_error("memory allocation failed");
  }
  //SLOG(INFO) << "Output memory allocated successfully";
  context_->setTensorAddress(tensor_name, label_memory_.get());
}

bool Model::setInputs(const cv::Mat& color, const cv::Mat& depth) {
  //SLOG(INFO) << "Setting inputs - color size: " << color.size() << ", type: " << color.type() << ", channels: " << color.channels();
  
  const auto color_info = info_.color();
  //SLOG(INFO) << "Color info: " << color_info;
  
  const auto color_shape = color_info.shape().updateFrom(color);
  //SLOG(INFO) << "Updated color shape - width: " << color_shape.width << ", height: " << color_shape.height;
  
  if (!color_.updateShape(color_shape) || !color_) {
    SLOG(ERROR) << "Failed to reshape color!";
    return false;
  }
  //SLOG(INFO) << "Color shape updated successfully";

  if (color_info.isDynamic()) {
    auto new_dims = color_info.replaceDynamic(color);
    //SLOG(INFO) << "Setting dynamic input shape: " << toString(new_dims);
    context_->setInputShape(color_info.name.c_str(), new_dims);
  }

  //SLOG(INFO) << "Filling color image - host buffer size: " << color_.host_image.size();
  color_conversion_.fillImage(color, color_.host_image);
  
  //SLOG(INFO) << "Setting input tensor address";
  context_->setInputTensorAddress(color_info.name.c_str(), color_.device_image.get());
  
  //SLOG(INFO) << "Copying data to device - size: " << color_.size() << " bytes";
  auto error = cudaMemcpyAsync(color_.device_image.get(),
                               color_.host_image.data,
                               color_.size(),
                               cudaMemcpyHostToDevice,
                               stream_);
  if (error != cudaSuccess) {
    SLOG(ERROR) << "Copying color input failed: " << cudaGetErrorString(error);
    return false;
  }
  //SLOG(INFO) << "Color data copied to device successfully";

  initOutput(color);

  const auto depth_info = info_.depth();
  if (!depth_info) {
    if (!depth.empty()) {
      SLOG(WARNING) << "Depth input provided but not used!";
    }
    return true;
  }

  if (depth.empty()) {
    SLOG(ERROR) << "Depth required by network!";
    return false;
  }

  const auto depth_shape = depth_info->shape().updateFrom(depth);
  if (!depth_.updateShape(depth_shape) || !depth_) {
    SLOG(ERROR) << "Failed to reshape depth!";
    return false;
  }

  depth_conversion_.fillImage(depth, depth_.host_image);
  context_->setInputTensorAddress(depth_info.value().name.c_str(),
                                  depth_.device_image.get());
  error = cudaMemcpyAsync(depth_.device_image.get(),
                          depth_.host_image.data,
                          depth_.size(),
                          cudaMemcpyHostToDevice,
                          stream_);
  if (error != cudaSuccess) {
    SLOG(ERROR) << "Copying color input failed: " << cudaGetErrorString(error);
    return false;
  }

  return error == cudaSuccess;
}

SegmentationResult Model::infer() const {
  //SLOG(INFO) << "Starting inference";
  cudaStreamSynchronize(stream_);
  //SLOG(INFO) << "Stream synchronized";
  
  bool status = context_->enqueueV3(stream_);
  if (!status) {
    SLOG(ERROR) << "Inference failed!";
    return {};
  }
  //SLOG(INFO) << "Inference enqueued successfully";

  const auto curr_dims = context_->getTensorShape(info_.labels().name.c_str());
  const auto output_shape = getShapeFromDims(curr_dims);
  //SLOG(INFO) << "Output shape: " << output_shape.width << "x" << output_shape.height;

  cv::Mat labels(output_shape.height, output_shape.width, CV_32S);
  //SLOG(INFO) << "Created output Mat with size: " << labels.size();
  
  auto error = cudaMemcpyAsync(labels.data,
                             label_memory_.get(),
                             labels.step[0] * labels.rows,
                             cudaMemcpyDeviceToHost,
                             stream_);
  if (error != cudaSuccess) {
    SLOG(ERROR) << "Copying output failed: " << cudaGetErrorString(error);
    return {};
  }
  //SLOG(INFO) << "Output copied to host successfully";

  cudaStreamSynchronize(stream_);
  //SLOG(INFO) << "Final stream synchronization done";
  return {true, labels};
}

}  // namespace semantic_inference
