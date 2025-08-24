/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#ifndef OPS_BUILT_IN_AICPU_CONTEXT_STUB_AICPU_SHARDER_H_
#define OPS_BUILT_IN_AICPU_CONTEXT_STUB_AICPU_SHARDER_H_

#include <functional>
#include <cstdint>

namespace aicpu {
using Closure = std::function<void()>;
using ClosureBool = std::function<bool()>;
using RunnerBool = std::function<bool(Closure)>;
using SharderWork = std::function<void(int64_t, int64_t)>;

class SharderNonBlock {
 public:
  static SharderNonBlock &GetInstance();

  void Register(const RunnerBool &schedule, const ClosureBool &doTask,
                uint32_t cpuCoreNum);
  void ParallelFor(int64_t total, int64_t perUnitSize, const SharderWork &work);

  void ParallelForHash(int64_t total, int64_t cpuNums, const SharderWork &work);

  void Schedule(const Closure &closure);

  uint32_t GetCPUNum();

 private:
  SharderNonBlock();
  ~SharderNonBlock() = default;

  SharderNonBlock(const SharderNonBlock &) = delete;
  SharderNonBlock &operator=(const SharderNonBlock &) = delete;
  SharderNonBlock(SharderNonBlock &&) = delete;
  SharderNonBlock &operator=(SharderNonBlock &&) = delete;

  bool Enqueue(const Closure &closure);
  inline int64_t CeilMultiple(int64_t x, int64_t base);

  RunnerBool schedule_;
  ClosureBool doTask_;
  uint32_t cpuCoreNum_;
};
}  // namespace aicpu
#endif