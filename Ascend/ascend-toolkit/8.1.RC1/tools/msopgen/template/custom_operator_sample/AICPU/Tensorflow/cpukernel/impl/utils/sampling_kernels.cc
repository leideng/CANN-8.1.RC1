/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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

#include "sampling_kernels.h"
#include <map>
using namespace std;

namespace aicpu {
SamplingKernelType SamplingKernelTypeFromString(const std::string &str) {
  // Define map for different types of sampling kernels
  static const std::map<std::string, SamplingKernelType> SamplingTypesInfo {
    {"lanczos1",      Lanczos1Kernel},
    {"lanczos3",      Lanczos3Kernel},
    {"lanczos5",      Lanczos5Kernel},
    {"gaussian",      GaussianKernel},
    {"box",           BoxKernel},
    {"triangle",      TriangleKernel},
    {"keyscubic",     KeysCubicKernel},
    {"mitchellcubic", MitchellCubicKernel},
  };

  std::map<std::string, SamplingKernelType>::const_iterator iter = SamplingTypesInfo.find(str);
  if (iter != SamplingTypesInfo.end()) {
    return iter->second;
  }

  return SamplingKernelTypeEnd;
}
}  // namespace aicpu