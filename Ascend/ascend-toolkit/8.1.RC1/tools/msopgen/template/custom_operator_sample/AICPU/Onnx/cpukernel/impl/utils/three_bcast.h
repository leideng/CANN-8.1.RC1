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
#ifndef _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_THREE_BCAST_H_
#define _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_THREE_BCAST_H_

#include "cpu_context.h"

namespace aicpu {
enum class ThreeBcastShapeType {
  SAME_SHAPE = 0,
  X_ONE_ELEMENT = 1,
  Y_ONE_ELEMENT = 2,
  Z_ONE_ELEMENT = 3,
  XY_ONE_ELEMENT = 4,
  XZ_ONE_ELEMENT = 5,
  YZ_ONE_ELEMENT = 6,
  DIFF_SHAPE = 7
};

struct ThreeBCalcInfo {
  ThreeBCalcInfo()
      : input_0(nullptr), input_1(nullptr), input_2(nullptr), output(nullptr) {}
  Tensor *input_0;
  Tensor *input_1;
  Tensor *input_2;
  Tensor *output;
  std::vector<int64_t> reshape_0;
  std::vector<int64_t> reshape_1;
  std::vector<int64_t> reshape_2;
  std::vector<int64_t> shape_out;
  std::vector<int64_t> bcast_0;
  std::vector<int64_t> bcast_1;
  std::vector<int64_t> bcast_2;
};

class ThreeBcast {
 public:
  ThreeBcast() = default;
  ~ThreeBcast() = default;

  uint32_t GenerateBcastInfo(const ThreeBCalcInfo &calcInfo);
  void GenerateBcastStrides();
  void GetBcastVec(ThreeBCalcInfo &calcInfo);
  uint32_t ThreeBcastInfo(size_t max_size);
  uint32_t BcastCheck(const int64_t x, const int64_t y, const int64_t z) const;
  int64_t GetBroadcastXIndex(int64_t index) const;
  int64_t GetBroadcastYIndex(int64_t index) const;
  int64_t GetBroadcastZIndex(int64_t index) const;

 private:
  std::vector<int64_t> x_reshape_;
  std::vector<int64_t> y_reshape_;
  std::vector<int64_t> z_reshape_;
  std::vector<int64_t> shape_out_;
  std::vector<int64_t> x_bcast_;
  std::vector<int64_t> y_bcast_;
  std::vector<int64_t> z_bcast_;
  std::vector<int64_t> x_input_strides_;
  std::vector<int64_t> y_input_strides_;
  std::vector<int64_t> z_input_strides_;
  std::vector<int64_t> x_output_strides_;
  std::vector<int64_t> y_output_strides_;
  std::vector<int64_t> z_output_strides_;
};
}  // namespace aicpu
#endif  // _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_
