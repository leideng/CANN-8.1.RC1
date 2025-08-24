/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#ifndef _AICPU_UTILS_GUARDED_PHILOX_RANDOM_
#define _AICPU_UTILS_GUARDED_PHILOX_RANDOM_

#include "utils/kernel_util.h"
#include "philox_random.h"

namespace aicpu {
class GuardedPhiloxRandom {
 public:
  // Must call Init to finish initialization
  GuardedPhiloxRandom() : initialized_(false) {}

  // Initialize the generator from attributes "seed" and "seed2".
  void Init(const CpuKernelContext &ctx);

  // Initialize with given seeds.
  void Init(int64_t seed, int64_t seed2);

  // Reserve a certain number of 128-bit samples.
  PhiloxRandom ReserveSamples128(int64_t samples);

  // Reserve a certain number of 32-bit samples.
  PhiloxRandom ReserveSamples32(int64_t samples) {
    const static int64_t up_num = 3;
    const static int64_t mod_num = 4;
    return ReserveSamples128((samples + up_num) / mod_num);
  }

  // Reserve enough random samples in the generator for the given output count.
  PhiloxRandom ReserveRandomOutputs(int64_t output_count, int multiplier) {
    int64_t conservative_sample_count = output_count * multiplier;
    return ReserveSamples128(conservative_sample_count);
  }

 private:
  PhiloxRandom generator_;
  bool initialized_;
};
}  // namespace aicpu

#endif  // _AICPU_UTILS_GUARDED_PHILOX_RANDOM_
