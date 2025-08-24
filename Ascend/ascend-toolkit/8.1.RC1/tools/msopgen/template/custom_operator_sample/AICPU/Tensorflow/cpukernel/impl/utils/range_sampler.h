/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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
#ifndef AICPU_UTILS_RANGE_SAMPLER_H_
#define AICPU_UTILS_RANGE_SAMPLER_H_

#include <cstdint>
#include <random>
#include <vector>
#include "distinct_uniform_int_distribution.h"
#include "log.h"
#include "status.h"

namespace aicpu {
namespace cpu {
template <typename T>
using ArraySlice = std::vector<T>;

template <typename T>
using MutableArraySlice = std::vector<T>;

// Abstract subclass for sampling from the set of non-negative integers
// [0, range)
class RangeSampler {
 public:
  explicit RangeSampler(int64_t range)
      : range_(range) {
  }
  virtual ~RangeSampler();

  // Sample a single value
  virtual int64_t Sample() const = 0;

  // The probability that a single call to Sample() returns the given value.
  // Assumes that value is in [0, range).  No range checking is done.
  virtual float Probability(int64_t value) const = 0;

  uint32_t SampleBatchGetExpectedCount(
      bool unique, aicpu::cpu::MutableArraySlice<int64_t> &batch,
      aicpu::cpu::MutableArraySlice<float> &batch_expected_count,
      const aicpu::cpu::ArraySlice<int64_t> &extras,
      aicpu::cpu::MutableArraySlice<float> &extras_expected_count) const;

  virtual uint32_t SampleBatchGetExpectedCountAvoid(
      bool unique, aicpu::cpu::MutableArraySlice<int64_t> &batch,
      aicpu::cpu::MutableArraySlice<float> &batch_expected_count,
      const aicpu::cpu::ArraySlice<int64_t> &extras,
      aicpu::cpu::MutableArraySlice<float> &extras_expected_count,
      const aicpu::cpu::ArraySlice<int64_t> &avoided_values) const;

  uint32_t ComputeExpectedCount(
      size_t kBatchSize, int num_tries,
      aicpu::cpu::MutableArraySlice<int64_t> &batch,
      aicpu::cpu::MutableArraySlice<float> &batch_expected_count,
      const aicpu::cpu::ArraySlice<int64_t> &extras,
      aicpu::cpu::MutableArraySlice<float> &extras_expected_count) const;

  int64_t range() const {
    return range_;
  }

 protected:
  const int64_t range_;
};

class UniformSampler : public RangeSampler {
 public:
  explicit UniformSampler(int64_t range);

  ~UniformSampler() override {
  }

  int64_t Sample() const override;

  float Probability(int64_t value) const override;

 private:
  const float inv_range_;
};

class LogUniformSampler : public RangeSampler {
 public:
  explicit LogUniformSampler(int64_t range);

  ~LogUniformSampler() override {
  }

  int64_t Sample() const override;

  float Probability(int64_t value) const override;

 private:
  const double log_range_;
};

}  // namespace cpu
}  // namespace aicpu

#endif  // AICPU_UTILS_RANGE_SAMPLER_H_
