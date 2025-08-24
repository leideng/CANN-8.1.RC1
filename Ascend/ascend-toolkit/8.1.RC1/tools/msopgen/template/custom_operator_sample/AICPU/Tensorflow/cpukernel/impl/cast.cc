/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "cast.h"

#include <memory.h>
#include <cfloat>
#include <ctime>

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kCast = "Cast";
}

namespace aicpu {
CastCpuKernel::CastCpuKernel() : calls_({}), x_tensor_(nullptr), y_tensor_(nullptr),
                                 x_data_type_(DT_INT64), y_data_type_(DT_INT64),
                                 x_data_size_(0), y_data_size_(0) {}

template <typename T, typename S>
uint32_t CastTask(Tensor *&x_tensor, Tensor *&y_tensor, int64_t &start,
                  int64_t &end) {
  T *inptr = static_cast<T *>(x_tensor->GetData());
  S *outptr = static_cast<S *>(y_tensor->GetData());
  Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> input_map(
      (inptr + start), 1, (end - start));
  const auto &input = Eigen::Tensor<T, 2, Eigen::RowMajor>(input_map);
  Eigen::TensorMap<Eigen::Tensor<S, 2, Eigen::RowMajor>> output(
      (outptr + start), 1, (end - start));
  output = input.template cast<S>();
  return KERNEL_STATUS_OK;
}

void CastCpuKernel::SetMap() {
  calls_[DT_INT8][DT_INT8] = CastTask<int8_t, int8_t>;
  calls_[DT_INT8][DT_INT16] = CastTask<int8_t, int16_t>;
  calls_[DT_INT8][DT_INT32] = CastTask<int8_t, int32_t>;
  calls_[DT_INT8][DT_INT64] = CastTask<int8_t, int64_t>;
  calls_[DT_INT8][DT_FLOAT16] = CastTask<int8_t, Eigen::half>;
  calls_[DT_INT8][DT_FLOAT] = CastTask<int8_t, float>;
  calls_[DT_INT8][DT_DOUBLE] = CastTask<int8_t, double>;
  calls_[DT_INT8][DT_UINT8] = CastTask<int8_t, uint8_t>;
  calls_[DT_INT8][DT_UINT16] = CastTask<int8_t, uint16_t>;
  calls_[DT_INT8][DT_UINT32] = CastTask<int8_t, uint32_t>;
  calls_[DT_INT8][DT_UINT64] = CastTask<int8_t, uint64_t>;
  calls_[DT_INT8][DT_BOOL] = CastTask<int8_t, bool>;
  calls_[DT_INT16][DT_INT8] = CastTask<int16_t, int8_t>;
  calls_[DT_INT16][DT_INT16] = CastTask<int16_t, int16_t>;
  calls_[DT_INT16][DT_INT32] = CastTask<int16_t, int32_t>;
  calls_[DT_INT16][DT_INT64] = CastTask<int16_t, int64_t>;
  calls_[DT_INT16][DT_FLOAT16] = CastTask<int16_t, Eigen::half>;
  calls_[DT_INT16][DT_FLOAT] = CastTask<int16_t, float>;
  calls_[DT_INT16][DT_DOUBLE] = CastTask<int16_t, double>;
  calls_[DT_INT16][DT_UINT8] = CastTask<int16_t, uint8_t>;
  calls_[DT_INT16][DT_UINT16] = CastTask<int16_t, uint16_t>;
  calls_[DT_INT16][DT_UINT32] = CastTask<int16_t, uint32_t>;
  calls_[DT_INT16][DT_UINT64] = CastTask<int16_t, uint64_t>;
  calls_[DT_INT16][DT_BOOL] = CastTask<int16_t, bool>;
  calls_[DT_INT32][DT_INT8] = CastTask<int32_t, int8_t>;
  calls_[DT_INT32][DT_INT16] = CastTask<int32_t, int16_t>;
  calls_[DT_INT32][DT_INT32] = CastTask<int32_t, int32_t>;
  calls_[DT_INT32][DT_INT64] = CastTask<int32_t, int64_t>;
  calls_[DT_INT32][DT_FLOAT16] = CastTask<int32_t, Eigen::half>;
  calls_[DT_INT32][DT_FLOAT] = CastTask<int32_t, float>;
  calls_[DT_INT32][DT_DOUBLE] = CastTask<int32_t, double>;
  calls_[DT_INT32][DT_UINT8] = CastTask<int32_t, uint8_t>;
  calls_[DT_INT32][DT_UINT16] = CastTask<int32_t, uint16_t>;
  calls_[DT_INT32][DT_UINT32] = CastTask<int32_t, uint32_t>;
  calls_[DT_INT32][DT_UINT64] = CastTask<int32_t, uint64_t>;
  calls_[DT_INT32][DT_BOOL] = CastTask<int32_t, bool>;
  calls_[DT_INT64][DT_INT8] = CastTask<int64_t, int8_t>;
  calls_[DT_INT64][DT_INT16] = CastTask<int64_t, int16_t>;
  calls_[DT_INT64][DT_INT32] = CastTask<int64_t, int32_t>;
  calls_[DT_INT64][DT_INT64] = CastTask<int64_t, int64_t>;
  calls_[DT_INT64][DT_FLOAT16] = CastTask<int64_t, Eigen::half>;
  calls_[DT_INT64][DT_FLOAT] = CastTask<int64_t, float>;
  calls_[DT_INT64][DT_DOUBLE] = CastTask<int64_t, double>;
  calls_[DT_INT64][DT_UINT8] = CastTask<int64_t, uint8_t>;
  calls_[DT_INT64][DT_UINT16] = CastTask<int64_t, uint16_t>;
  calls_[DT_INT64][DT_UINT32] = CastTask<int64_t, uint32_t>;
  calls_[DT_INT64][DT_UINT64] = CastTask<int64_t, uint64_t>;
  calls_[DT_INT64][DT_BOOL] = CastTask<int64_t, bool>;
  calls_[DT_FLOAT16][DT_INT8] = CastTask<Eigen::half, int8_t>;
  calls_[DT_FLOAT16][DT_INT16] = CastTask<Eigen::half, int16_t>;
  calls_[DT_FLOAT16][DT_INT32] = CastTask<Eigen::half, int32_t>;
  calls_[DT_FLOAT16][DT_INT64] = CastTask<Eigen::half, int64_t>;
  calls_[DT_FLOAT16][DT_FLOAT16] = CastTask<Eigen::half, Eigen::half>;
  calls_[DT_FLOAT16][DT_FLOAT] = CastTask<Eigen::half, float>;
  calls_[DT_FLOAT16][DT_DOUBLE] = CastTask<Eigen::half, double>;
  calls_[DT_FLOAT16][DT_UINT8] = CastTask<Eigen::half, uint8_t>;
  calls_[DT_FLOAT16][DT_UINT16] = CastTask<Eigen::half, uint16_t>;
  calls_[DT_FLOAT16][DT_UINT32] = CastTask<Eigen::half, uint32_t>;
  calls_[DT_FLOAT16][DT_UINT64] = CastTask<Eigen::half, uint64_t>;
  calls_[DT_FLOAT16][DT_BOOL] = CastTask<Eigen::half, bool>;
  calls_[DT_FLOAT][DT_INT8] = CastTask<float, int8_t>;
  calls_[DT_FLOAT][DT_INT16] = CastTask<float, int16_t>;
  calls_[DT_FLOAT][DT_INT32] = CastTask<float, int32_t>;
  calls_[DT_FLOAT][DT_INT64] = CastTask<float, int64_t>;
  calls_[DT_FLOAT][DT_FLOAT16] = CastTask<float, Eigen::half>;
  calls_[DT_FLOAT][DT_FLOAT] = CastTask<float, float>;
  calls_[DT_FLOAT][DT_DOUBLE] = CastTask<float, double>;
  calls_[DT_FLOAT][DT_UINT8] = CastTask<float, uint8_t>;
  calls_[DT_FLOAT][DT_UINT16] = CastTask<float, uint16_t>;
  calls_[DT_FLOAT][DT_UINT32] = CastTask<float, uint32_t>;
  calls_[DT_FLOAT][DT_UINT64] = CastTask<float, uint64_t>;
  calls_[DT_FLOAT][DT_BOOL] = CastTask<float, bool>;
  calls_[DT_DOUBLE][DT_INT8] = CastTask<double, int8_t>;
  calls_[DT_DOUBLE][DT_INT16] = CastTask<double, int16_t>;
  calls_[DT_DOUBLE][DT_INT32] = CastTask<double, int32_t>;
  calls_[DT_DOUBLE][DT_INT64] = CastTask<double, int64_t>;
  calls_[DT_DOUBLE][DT_FLOAT16] = CastTask<double, Eigen::half>;
  calls_[DT_DOUBLE][DT_FLOAT] = CastTask<double, float>;
  calls_[DT_DOUBLE][DT_DOUBLE] = CastTask<double, double>;
  calls_[DT_DOUBLE][DT_UINT8] = CastTask<double, uint8_t>;
  calls_[DT_DOUBLE][DT_UINT16] = CastTask<double, uint16_t>;
  calls_[DT_DOUBLE][DT_UINT32] = CastTask<double, uint32_t>;
  calls_[DT_DOUBLE][DT_UINT64] = CastTask<double, uint64_t>;
  calls_[DT_DOUBLE][DT_BOOL] = CastTask<double, bool>;
  calls_[DT_UINT8][DT_INT8] = CastTask<uint8_t, int8_t>;
  calls_[DT_UINT8][DT_INT16] = CastTask<uint8_t, int16_t>;
  calls_[DT_UINT8][DT_INT32] = CastTask<uint8_t, int32_t>;
  calls_[DT_UINT8][DT_INT64] = CastTask<uint8_t, int64_t>;
  calls_[DT_UINT8][DT_FLOAT16] = CastTask<uint8_t, Eigen::half>;
  calls_[DT_UINT8][DT_FLOAT] = CastTask<uint8_t, float>;
  calls_[DT_UINT8][DT_DOUBLE] = CastTask<uint8_t, double>;
  calls_[DT_UINT8][DT_UINT8] = CastTask<uint8_t, uint8_t>;
  calls_[DT_UINT8][DT_UINT16] = CastTask<uint8_t, uint16_t>;
  calls_[DT_UINT8][DT_UINT32] = CastTask<uint8_t, uint32_t>;
  calls_[DT_UINT8][DT_UINT64] = CastTask<uint8_t, uint64_t>;
  calls_[DT_UINT8][DT_BOOL] = CastTask<uint8_t, bool>;
  calls_[DT_UINT16][DT_INT8] = CastTask<uint16_t, int8_t>;
  calls_[DT_UINT16][DT_INT16] = CastTask<uint16_t, int16_t>;
  calls_[DT_UINT16][DT_INT32] = CastTask<uint16_t, int32_t>;
  calls_[DT_UINT16][DT_INT64] = CastTask<uint16_t, int64_t>;
  calls_[DT_UINT16][DT_FLOAT16] = CastTask<uint16_t, Eigen::half>;
  calls_[DT_UINT16][DT_FLOAT] = CastTask<uint16_t, float>;
  calls_[DT_UINT16][DT_DOUBLE] = CastTask<uint16_t, double>;
  calls_[DT_UINT16][DT_UINT8] = CastTask<uint16_t, uint8_t>;
  calls_[DT_UINT16][DT_UINT16] = CastTask<uint16_t, uint16_t>;
  calls_[DT_UINT16][DT_UINT32] = CastTask<uint16_t, uint32_t>;
  calls_[DT_UINT16][DT_UINT64] = CastTask<uint16_t, uint64_t>;
  calls_[DT_UINT16][DT_BOOL] = CastTask<uint16_t, bool>;
  calls_[DT_UINT32][DT_INT8] = CastTask<uint32_t, int8_t>;
  calls_[DT_UINT32][DT_INT16] = CastTask<uint32_t, int16_t>;
  calls_[DT_UINT32][DT_INT32] = CastTask<uint32_t, int32_t>;
  calls_[DT_UINT32][DT_INT64] = CastTask<uint32_t, int64_t>;
  calls_[DT_UINT32][DT_FLOAT16] = CastTask<uint32_t, Eigen::half>;
  calls_[DT_UINT32][DT_FLOAT] = CastTask<uint32_t, float>;
  calls_[DT_UINT32][DT_DOUBLE] = CastTask<uint32_t, double>;
  calls_[DT_UINT32][DT_UINT8] = CastTask<uint32_t, uint8_t>;
  calls_[DT_UINT32][DT_UINT16] = CastTask<uint32_t, uint16_t>;
  calls_[DT_UINT32][DT_UINT32] = CastTask<uint32_t, uint32_t>;
  calls_[DT_UINT32][DT_UINT64] = CastTask<uint32_t, uint64_t>;
  calls_[DT_UINT32][DT_BOOL] = CastTask<uint32_t, bool>;
  calls_[DT_UINT64][DT_INT8] = CastTask<uint64_t, int8_t>;
  calls_[DT_UINT64][DT_INT16] = CastTask<uint64_t, int16_t>;
  calls_[DT_UINT64][DT_INT32] = CastTask<uint64_t, int32_t>;
  calls_[DT_UINT64][DT_INT64] = CastTask<uint64_t, int64_t>;
  calls_[DT_UINT64][DT_FLOAT16] = CastTask<uint64_t, Eigen::half>;
  calls_[DT_UINT64][DT_FLOAT] = CastTask<uint64_t, float>;
  calls_[DT_UINT64][DT_DOUBLE] = CastTask<uint64_t, double>;
  calls_[DT_UINT64][DT_UINT8] = CastTask<uint64_t, uint8_t>;
  calls_[DT_UINT64][DT_UINT16] = CastTask<uint64_t, uint16_t>;
  calls_[DT_UINT64][DT_UINT32] = CastTask<uint64_t, uint32_t>;
  calls_[DT_UINT64][DT_UINT64] = CastTask<uint64_t, uint64_t>;
  calls_[DT_UINT64][DT_BOOL] = CastTask<uint64_t, bool>;
  calls_[DT_BOOL][DT_INT8] = CastTask<bool, int8_t>;
  calls_[DT_BOOL][DT_INT16] = CastTask<bool, int16_t>;
  calls_[DT_BOOL][DT_INT32] = CastTask<bool, int32_t>;
  calls_[DT_BOOL][DT_INT64] = CastTask<bool, int64_t>;
  calls_[DT_BOOL][DT_FLOAT16] = CastTask<bool, Eigen::half>;
  calls_[DT_BOOL][DT_FLOAT] = CastTask<bool, float>;
  calls_[DT_BOOL][DT_DOUBLE] = CastTask<bool, double>;
  calls_[DT_BOOL][DT_UINT8] = CastTask<bool, uint8_t>;
  calls_[DT_BOOL][DT_UINT16] = CastTask<bool, uint16_t>;
  calls_[DT_BOOL][DT_UINT32] = CastTask<bool, uint32_t>;
  calls_[DT_BOOL][DT_UINT64] = CastTask<bool, uint64_t>;
  calls_[DT_BOOL][DT_BOOL] = CastTask<bool, bool>;
}
//transfer data
uint32_t CastCpuKernel::TransferType(int64_t start, int64_t end) {
  if (calls_.find(x_data_type_) == calls_.end()) {
    KERNEL_LOG_ERROR(
        "Cast kernel input data types: [%s] not support",
        typeid(x_data_type_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  } else if (calls_[x_data_type_].find(y_data_type_) == calls_[x_data_type_].end()) {
    KERNEL_LOG_ERROR(
        "Cast kernel output data types: [%s] not support",
        typeid(y_data_type_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return calls_[x_data_type_][y_data_type_](x_tensor_, y_tensor_, start, end);
}

uint32_t CastCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("CastCpuKernel start.");
  x_tensor_ = ctx.Input(0);
  if (x_tensor_ == nullptr) {
    KERNEL_LOG_ERROR("Get input tensor failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  y_tensor_ = ctx.Output(0);
  if (y_tensor_ == nullptr) {
    KERNEL_LOG_ERROR("Get output tensor failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  x_data_size_ = x_tensor_->GetDataSize();
  y_data_size_ = y_tensor_->GetDataSize();

  if (x_data_size_ == 0) {
    KERNEL_LOG_INFO("Input data is empty, input size: [%llu]",
                     x_data_size_);
    return KERNEL_STATUS_OK;
  }
  KERNEL_LOG_INFO("Cast input size: [%llu], output size: [%llu]",
                  x_data_size_, y_data_size_);
  x_data_type_ = DataType(x_tensor_->GetDataType());
  y_data_type_ = DataType(y_tensor_->GetDataType());
  KERNEL_LOG_INFO("Cast input type: [%s], output type: [%s]",
                  DTypeStr(x_data_type_).c_str(), DTypeStr(y_data_type_).c_str());
  int x_type_size = GetSizeByDataType(x_data_type_);
  int y_type_size = GetSizeByDataType(y_data_type_);
  KERNEL_LOG_INFO("Cast input type size: [%d], output type size: [%d]",
                  x_type_size, y_type_size);
  if (x_type_size <= 0 || y_type_size <= 0) {
    KERNEL_LOG_ERROR("Input type size and output type size should greater than 0, "
                     "input data type: [%s], input data size: [%d], "
                     "output data type: [%s], output data size: [%d]",
                     DTypeStr(x_data_type_).c_str(), x_type_size,
                     DTypeStr(y_data_type_).c_str(), y_type_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  x_data_size_ = x_data_size_ / static_cast<uint64_t>(x_type_size);
  y_data_size_ = y_data_size_ / static_cast<uint64_t>(y_type_size);
  KERNEL_LOG_INFO("Cast input data length: [%llu], output data length: [%llu]",
                  x_data_size_, y_data_size_);
  if (x_data_size_ > y_data_size_) {
    x_data_size_ = y_data_size_;
  }

  /*
Procedure for multi-core concurrent computing:
1. Call the CpuKernelUtils::GetCPUNum function to obtain the number of AI CPUs (max_core_num).
2. Calculate the computing data size on each AI CPU (per_unit_size) by dividing the total data size by the number of AI CPUs.
3. Implement the working process function shard of each compute unit, and compile the computing logic that needs to be
   concurrently executed in the function.
4. Call the CpuKernelUtils::ParallelFor function and input parameters such as the CpuKernelContext object (ctx), total
   data size (data_num), computing data size on each AI CPU (per_unit_size), and working process function shard of each
   compute unit. Then execute multi-core concurrent computing.
For example:
uint32_t min_core_num = 1;
int64_t max_core_num =
      std::max(min_core_num, CpuKernelUtils::GetCPUNum(ctx));
per_unit_size = data_num / max_core_num;
auto shard = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
    // Execution process
     ... ...
    }
};
CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size, shard);
*/

  uint32_t min_core_num = 1;
  uint64_t max_core_num = std::max(
    min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
  if (max_core_num > x_data_size_) {
    max_core_num = x_data_size_;
  }
  SetMap();
  aicpu::CpuKernelUtils::ParallelFor(
      ctx, x_data_size_, x_data_size_ / max_core_num,
      [&](int64_t start, int64_t end) {
        uint32_t result = TransferType(start, end);
        if (result == KERNEL_STATUS_PARAM_INVALID) {
          KERNEL_LOG_ERROR("Cast TransferType failed");
          return KERNEL_STATUS_PARAM_INVALID;
        }
        return KERNEL_STATUS_OK;
      });
  calls_.clear();
  y_data_size_ = y_tensor_->GetDataSize();
  KERNEL_LOG_INFO("Cast output size: [%llu].", y_data_size_);
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kCast, CastCpuKernel);
}  // namespace aicpu
