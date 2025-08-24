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
#include "range_sampler.h"
#include <cmath>
#include <unordered_set>
#include <vector>

namespace aicpu {
namespace cpu {
RangeSampler::~RangeSampler() {
}

uint32_t RangeSampler::SampleBatchGetExpectedCount(
    bool unique, aicpu::cpu::MutableArraySlice<int64_t> &batch,
    aicpu::cpu::MutableArraySlice<float> &batch_expected_count,
    const aicpu::cpu::ArraySlice<int64_t> &extras,
    aicpu::cpu::MutableArraySlice<float> &extras_expected_count) const {
  return SampleBatchGetExpectedCountAvoid(unique, batch, batch_expected_count,
                                          extras, extras_expected_count,
                                          aicpu::cpu::ArraySlice<int64_t>());
}

namespace {
float ExpectedCountHelper(float p, int batch_size, int number_tries) {
  if (number_tries == batch_size) {
    // This shortcut will always be taken if unique=false
    return p * batch_size;
  }
  // numerically stable version of (1 - (1-p)^number_tries)
  return -std::expm1(number_tries * std::log1p(-p));
}

template <class Collection>
bool InsertIfNotPresent(Collection *const collection,
                        const typename Collection::value_type &vt) {
  return collection->insert(vt).second;
}

const int32_t kInt32Max = static_cast<int32_t>(0x7FFFFFFF);
}  // namespace

uint32_t RangeSampler::SampleBatchGetExpectedCountAvoid(
    bool unique, aicpu::cpu::MutableArraySlice<int64_t> &batch,
    aicpu::cpu::MutableArraySlice<float> &batch_expected_count,
    const aicpu::cpu::ArraySlice<int64_t> &extras,
    aicpu::cpu::MutableArraySlice<float> &extras_expected_count,
    const aicpu::cpu::ArraySlice<int64_t> &avoided_values) const {
  const size_t kBatchSize = batch.size();
  if (range_ <= 0) {
    KERNEL_LOG_ERROR("The value of range_:[%ld] must be greater than 0!",
                     range_);
    return KERNEL_STATUS_INNER_ERROR;
  }

  int num_tries = 0;
  if (unique) {
    if (kBatchSize + avoided_values.size() > static_cast<size_t>(range_)) {
      KERNEL_LOG_ERROR(
          "The value should be less than range_:[%ld], but got [%zu]", range_,
          kBatchSize + avoided_values.size());
      return KERNEL_STATUS_INNER_ERROR;
    }
    std::unordered_set<int64_t> used(kBatchSize);
    used.insert(avoided_values.begin(), avoided_values.end());
    size_t num_picked = 0u;
    num_tries = 0;
    while (num_picked < kBatchSize) {
      num_tries++;
      if (num_tries >= kInt32Max) {
        KERNEL_LOG_ERROR(
            "The num_tries: [%d] should be less than kInt32Max: [%d]!",
            num_tries, kInt32Max);
        return KERNEL_STATUS_INNER_ERROR;
      }
      int64_t value = Sample();
      if (InsertIfNotPresent(&used, value)) {
        batch[num_picked++] = value;
      }
    }
  } else {
    if (avoided_values.size() != size_t{0}) {
      KERNEL_LOG_ERROR("The avoided_values only supported with unique=true");
      return KERNEL_STATUS_INNER_ERROR;
    }
    for (size_t i = 0; i < kBatchSize; i++) {
      batch[i] = Sample();
    }
    num_tries = static_cast<int>(kBatchSize);
  }
  // Compute the expected counts of the batch and the extra values
  return ComputeExpectedCount(kBatchSize, num_tries, batch, batch_expected_count,
                              extras, extras_expected_count);
}

uint32_t RangeSampler::ComputeExpectedCount(
    size_t kBatchSize, int num_tries,
    aicpu::cpu::MutableArraySlice<int64_t> &batch,
    aicpu::cpu::MutableArraySlice<float> &batch_expected_count,
    const aicpu::cpu::ArraySlice<int64_t> &extras,
    aicpu::cpu::MutableArraySlice<float> &extras_expected_count) const {
  if (!batch_expected_count.empty()) {
    if (kBatchSize != batch_expected_count.size()) {
      KERNEL_LOG_ERROR(
          "The size of extras_expected_count:[%zu] should be equal "
          "to batch_size:[%zu]!",
          batch_expected_count.size(), kBatchSize);
      return KERNEL_STATUS_INNER_ERROR;
    }
    for (size_t i = 0; i < kBatchSize; i++) {
      batch_expected_count[i] =
          ExpectedCountHelper(Probability(batch[i]), static_cast<int>(kBatchSize), num_tries);
    }
  }
  if (extras.size() != extras_expected_count.size()) {
    KERNEL_LOG_ERROR(
        "The size of extras:[%zu] and extras_expected_count:[%zu] should be "
        "equal!",
        extras.size(), extras_expected_count.size());
    return KERNEL_STATUS_INNER_ERROR;
  }
  for (size_t i = 0; i < extras.size(); i++) {
    extras_expected_count[i] =
        ExpectedCountHelper(Probability(extras[i]), static_cast<int>(kBatchSize), num_tries);
  }
  return KERNEL_STATUS_OK;
}

UniformSampler::UniformSampler(int64_t range)
    : RangeSampler(range), inv_range_(1.0f / range) {
}

int64_t UniformSampler::Sample() const {
  std::random_device rd;
  uint64_t RNG_seed = rd();
  std::mt19937 gen(RNG_seed);
  aicpu::DistinctUniformIntDistribution<> dis(0, range_ - 1);
  return dis.exec(gen);
}

float UniformSampler::Probability(int64_t value) const {
  (void)value;
  return inv_range_;
}

LogUniformSampler::LogUniformSampler(int64_t range)
    : RangeSampler(range), log_range_(log1p(range)) {
}

int64_t LogUniformSampler::Sample() const {
  std::random_device rd;
  uint64_t RNG_seed = rd();
  std::mt19937 gen(RNG_seed);
  std::uniform_real_distribution<float> uni_real(0.0, 1.0);

  const int64_t value =
      static_cast<int64_t>(exp(uni_real(gen) * log_range_)) - 1;
  if (value < 0) {
    KERNEL_LOG_ERROR("The value: [%ld] should be >= 0", value);
    return 0;
  }
  return value % range_;
}

float LogUniformSampler::Probability(int64_t value) const {
  // value is returned iff the call to UniformDouble(log_range_) in the
  // Sample() function returns a value between log(value + 1)
  // and log(value + 2).   The probability of this is:
  // (log(value + 2) - log(value + 1)) / log_range
  // To avoid two calls to log(), we compute this as follows:
  return static_cast<float>((log((value + 2.0) / (value + 1.0))) / log_range_);
}
}  // namespace cpu
}  // namespace aicpu
