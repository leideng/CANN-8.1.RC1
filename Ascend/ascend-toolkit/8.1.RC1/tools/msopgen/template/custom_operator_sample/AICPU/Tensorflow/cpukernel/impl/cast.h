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
#ifndef _AICPU_KERNELS_NORMALIZED_CAST_H_
#define _AICPU_KERNELS_NORMALIZED_CAST_H_

#include "cpu_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class CastCpuKernel : public CpuKernel {
 public:
  CastCpuKernel();
  ~CastCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t TransferType(int64_t start, int64_t end);
  void SetMap();
  std::map<int, std::map<int, std::function<uint32_t(Tensor *&, Tensor *&,
                                                     int64_t &, int64_t &)>>>
      calls_;
  Tensor *x_tensor_;
  Tensor *y_tensor_;
  DataType x_data_type_;
  DataType y_data_type_;
  uint64_t x_data_size_;
  uint64_t y_data_size_;
};
}  // namespace aicpu
#endif // _AICPU_KERNELS_NORMALIZED_CAST_H_
