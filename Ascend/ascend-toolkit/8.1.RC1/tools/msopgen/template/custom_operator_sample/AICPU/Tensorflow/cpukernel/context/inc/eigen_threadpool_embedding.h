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
#ifndef AICPU_CONTEXT_COMMON_EIGEN_THREAD_POOL_EMBEDDING_H
#define AICPU_CONTEXT_COMMON_EIGEN_THREAD_POOL_EMBEDDING_H

#include <functional>
#include <memory>
#include <mutex>
#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

namespace aicpu {
using SharderWork = std::function<void(int64_t, int64_t)>;

class EigenThreadPoolEmbedding {
 public:
  static EigenThreadPoolEmbedding *GetInstance();

  /*
   * ParallelFor shards the "total" units of work.
   */
  void ParallelFor(int64_t total, int64_t per_unit_size,
                   const SharderWork &work) const;

 private:
  EigenThreadPoolEmbedding() = default;
  ~EigenThreadPoolEmbedding() = default;

  EigenThreadPoolEmbedding(const EigenThreadPoolEmbedding &) = delete;
  EigenThreadPoolEmbedding(EigenThreadPoolEmbedding &&) = delete;
  EigenThreadPoolEmbedding &operator=(const EigenThreadPoolEmbedding &) = delete;
  EigenThreadPoolEmbedding &operator=(EigenThreadPoolEmbedding &&) = delete;

  static std::mutex mutex_;  // protect init_flag_
  static bool init_flag_;    // true means initialized
  static int32_t
      core_num_;  // the number of CPU cores that can be used by users
  static std::unique_ptr<Eigen::ThreadPool> eigen_threadpool_;
  static std::unique_ptr<Eigen::ThreadPoolDevice> threadpool_device_;
};
};      // namespace aicpu
#endif  // AICPU_CONTEXT_COMMON_EIGEN_THREAD_POOL_EMBEDDING_H
