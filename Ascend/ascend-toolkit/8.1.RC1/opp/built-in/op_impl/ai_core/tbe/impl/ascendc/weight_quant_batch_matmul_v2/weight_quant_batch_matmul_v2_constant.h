/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file weight_quant_batch_matmul_v2_constant.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_CONSTANT_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_CONSTANT_H
namespace WeightQuantBatchMatmulV2 {
using HighPreciseType = int32_t;
using HighPerformanceType = half;
enum class QuantType {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
};

enum class PrecisionType {
    NONE = 0,
    HIGH_PRECISION = 1,
};

} // namespace WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_CONSTANT_H