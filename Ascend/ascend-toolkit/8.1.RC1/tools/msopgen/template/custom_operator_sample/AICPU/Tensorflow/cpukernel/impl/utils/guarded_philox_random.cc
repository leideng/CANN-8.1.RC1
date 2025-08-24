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

#include "guarded_philox_random.h"
#include "kernels/normalized/random/utils.h"
#include "utils/kernel_util.h"
#include "utils/philox_random.h"

namespace aicpu {
void GuardedPhiloxRandom::Init(const CpuKernelContext &ctx) {
  // Grab seed Attrs.
  int64_t seed = 0;
  int64_t seed2 = 0;

  auto attr_seed = ctx.GetAttr("seed");
  if (attr_seed != nullptr) {
    seed = attr_seed->GetInt();
  }
  auto attr_seed2 = ctx.GetAttr("seed2");
  if (attr_seed2 != nullptr) {
    seed2 = attr_seed2->GetInt();
  }

  // Initialize with the given seeds
  Init(seed, seed2);
  return;
}

void GuardedPhiloxRandom::Init(int64_t seed, int64_t seed2) {
  if (initialized_) {
    KERNEL_LOG_INFO("GuardedPhiloxRandom has initialized!");
    return;
  }
  if (seed == 0 && seed2 == 0) {
    // If both seeds are unspecified, use completely random seeds.
    seed = aicpu::random::New64();
    seed2 = aicpu::random::New64();
  }
  generator_ = PhiloxRandom(seed, seed2);
  initialized_ = true;
}

PhiloxRandom GuardedPhiloxRandom::ReserveSamples128(int64_t samples) {
  if (!initialized_) {
    KERNEL_LOG_ERROR("GuardedPhiloxRandom has not initialized!");
  }
  auto local = generator_;
  generator_.Skip(samples);
  return local;
}
}  // namespace aicpu
