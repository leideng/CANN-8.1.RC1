/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

/*!
 * \file three_bcast.cc
 * \brief
 */
#include "three_bcast.h"

#include <algorithm>
#include <unordered_set>

#include "log.h"
#include "status.h"

namespace {
const int64_t kNoBroadcastValue = 1;
const int64_t kNoRepeatElements = 2;
}  // namespace

namespace aicpu {

uint32_t ThreeBcast::ThreeBcastInfo(size_t max_size) {
  // genarate broarcast info
  x_bcast_.resize(max_size, kNoBroadcastValue);
  y_bcast_.resize(max_size, kNoBroadcastValue);
  z_bcast_.resize(max_size, kNoBroadcastValue);
  for (size_t i = 0; i < max_size; ++i) {
    // no need broadcast
    if ((x_reshape_[i] == y_reshape_[i]) && (y_reshape_[i] == z_reshape_[i])) {
      continue;
    }

    if (BcastCheck(x_reshape_[i], y_reshape_[i], z_reshape_[i]) !=
        KERNEL_STATUS_OK) {
      KERNEL_LOG_ERROR("Broadcast not support, dim_x[%zu]=%ld, dim_y[%zu]=%ld.",
                       i, x_reshape_[i], i, y_reshape_[i]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (x_reshape_[i] == kNoBroadcastValue) {
      x_bcast_[i] = std::max(y_reshape_[i], z_reshape_[i]);
    }
    if (y_reshape_[i] == kNoBroadcastValue) {
      y_bcast_[i] = std::max(x_reshape_[i], z_reshape_[i]);
    }
    if (z_reshape_[i] == kNoBroadcastValue) {
      z_bcast_[i] = std::max(x_reshape_[i], y_reshape_[i]);
    }
  }
  return KERNEL_STATUS_OK;
}

void ThreeBcast::GenerateBcastStrides() {
  // generate strides, just for row major
  int32_t size = static_cast<int32_t>(shape_out_.size());
  x_input_strides_.resize(size, 0);
  y_input_strides_.resize(size, 0);
  z_input_strides_.resize(size, 0);
  x_output_strides_.resize(size, 0);
  y_output_strides_.resize(size, 0);
  z_output_strides_.resize(size, 0);
  x_input_strides_[size - 1] = 1;
  y_input_strides_[size - 1] = 1;
  z_input_strides_[size - 1] = 1;
  x_output_strides_[size - 1] = 1;
  y_output_strides_[size - 1] = 1;
  z_output_strides_[size - 1] = 1;
  for (int32_t i = size - 2; i >= 0; --i) {
    x_input_strides_[i] = x_input_strides_[i + 1] * x_reshape_[i + 1];
    y_input_strides_[i] = y_input_strides_[i + 1] * y_reshape_[i + 1];
    z_input_strides_[i] = z_input_strides_[i + 1] * z_reshape_[i + 1];
    x_output_strides_[i] = x_output_strides_[i + 1] * shape_out_[i + 1];
    y_output_strides_[i] = y_output_strides_[i + 1] * shape_out_[i + 1];
    z_output_strides_[i] = z_output_strides_[i + 1] * shape_out_[i + 1];
  }
}

uint32_t ThreeBcast::GenerateBcastInfo(const ThreeBCalcInfo &calcInfo) {
  x_reshape_ = calcInfo.input_0->GetTensorShape()->GetDimSizes();
  y_reshape_ = calcInfo.input_1->GetTensorShape()->GetDimSizes();
  z_reshape_ = calcInfo.input_2->GetTensorShape()->GetDimSizes();
  shape_out_ = calcInfo.output->GetTensorShape()->GetDimSizes();

  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());
  std::reverse(z_reshape_.begin(), z_reshape_.end());

  size_t dim_num_x = x_reshape_.size();
  size_t dim_num_y = y_reshape_.size();
  size_t dim_num_z = z_reshape_.size();

  size_t max_size = std::max({dim_num_x, dim_num_y, dim_num_z});
  if (dim_num_x != max_size) {
    x_reshape_.resize(max_size, kNoBroadcastValue);
  }
  if (dim_num_y != max_size) {
    y_reshape_.resize(max_size, kNoBroadcastValue);
  }
  if (dim_num_z != max_size) {
    z_reshape_.resize(max_size, kNoBroadcastValue);
  }
  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());
  std::reverse(z_reshape_.begin(), z_reshape_.end());
  // Check if shape match
  if (shape_out_.size() != max_size) {
    KERNEL_LOG_ERROR("shape mismatch, max_dim_in=%zu, dim_out=%zu.", max_size,
                     shape_out_.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t i = 0; i < max_size; ++i) {
    if (shape_out_[i] !=
        std::max({x_reshape_[i], y_reshape_[i], z_reshape_[i]})) {
      KERNEL_LOG_ERROR(
          "shape mismatch, index=%zu, dim_x=%ld, dim_y=%ld, dim_z=%ld, dim_out=%ld.",
          i, x_reshape_[i], y_reshape_[i], z_reshape_[i], shape_out_[i]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  // genarate broarcast info
  if (ThreeBcastInfo(max_size) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Genarate broarcast info failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // generate broadcast strides
  GenerateBcastStrides();
  return KERNEL_STATUS_OK;
}

uint32_t ThreeBcast::BcastCheck(const int64_t x, const int64_t y,
                                const int64_t z) const {
  std::unordered_set<int64_t> set_tmp{x, y, z};
  if (set_tmp.size() != kNoRepeatElements) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (x != 1 && y != 1 && z != 1) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

void ThreeBcast::GetBcastVec(ThreeBCalcInfo &calcInfo) {
  calcInfo.reshape_0 = std::move(x_reshape_);
  calcInfo.reshape_1 = std::move(y_reshape_);
  calcInfo.reshape_2 = std::move(z_reshape_);
  calcInfo.shape_out = std::move(shape_out_);
  calcInfo.bcast_0 = std::move(x_bcast_);
  calcInfo.bcast_1 = std::move(y_bcast_);
  calcInfo.bcast_2 = std::move(z_bcast_);
}

int64_t ThreeBcast::GetBroadcastXIndex(int64_t index) const {
  int64_t input_index = 0;
  const size_t num_dims = shape_out_.size();
  for (size_t i = 0; i < num_dims - 1; ++i) {
    const int64_t idx = index / x_output_strides_[i];
    if (x_bcast_[i] == kNoBroadcastValue) {
      input_index += idx * x_input_strides_[i];
    } else {
      if (x_reshape_[i] != kNoBroadcastValue) {
        input_index += (idx % x_reshape_[i]) * x_input_strides_[i];
      }
    }
    index -= idx * x_output_strides_[i];
  }
  if (x_bcast_[num_dims - 1] == kNoBroadcastValue) {
    input_index += index;
  } else {
    if (x_reshape_[num_dims - 1] != kNoBroadcastValue) {
      input_index += (index % x_reshape_[num_dims - 1]);
    }
  }
  return input_index;
}

int64_t ThreeBcast::GetBroadcastYIndex(int64_t index) const {
  int64_t input_index = 0;
  const size_t num_dims = shape_out_.size();
  for (size_t i = 0; i < num_dims - 1; ++i) {
    const int64_t idx = index / y_output_strides_[i];
    if (y_bcast_[i] == kNoBroadcastValue) {
      input_index += idx * y_input_strides_[i];
    } else {
      if (y_reshape_[i] != kNoBroadcastValue) {
        input_index += (idx % y_reshape_[i]) * y_input_strides_[i];
      }
    }
    index -= idx * y_output_strides_[i];
  }
  if (y_bcast_[num_dims - 1] == kNoBroadcastValue) {
    input_index += index;
  } else {
    if (y_reshape_[num_dims - 1] != kNoBroadcastValue) {
      input_index += (index % y_reshape_[num_dims - 1]);
    }
  }
  return input_index;
}

int64_t ThreeBcast::GetBroadcastZIndex(int64_t index) const {
  int64_t input_index = 0;
  const size_t num_dims = shape_out_.size();
  for (size_t i = 0; i < num_dims - 1; ++i) {
    const int64_t idx = index / y_output_strides_[i];
    if (z_bcast_[i] == kNoBroadcastValue) {
      input_index += idx * z_input_strides_[i];
    } else {
      if (z_reshape_[i] != kNoBroadcastValue) {
        input_index += (idx % z_reshape_[i]) * z_input_strides_[i];
      }
    }
    index -= idx * z_output_strides_[i];
  }
  if (z_bcast_[num_dims - 1] == kNoBroadcastValue) {
    input_index += index;
  } else {
    if (z_reshape_[num_dims - 1] != kNoBroadcastValue) {
      input_index += (index % z_reshape_[num_dims - 1]);
    }
  }
  return input_index;
}
}  // namespace aicpu
