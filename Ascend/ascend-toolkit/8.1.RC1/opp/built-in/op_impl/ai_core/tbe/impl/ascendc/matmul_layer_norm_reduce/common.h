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
 * \file common.h
 * \brief
 */

namespace MatmulLayerNormReduceND {
constexpr uint32_t FRACTAL_M = 16;
constexpr uint32_t FRACTAL_N = 16;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t REPEAT_MAX_BLOCK_NUM = 8;
constexpr uint32_t IN_PING_INDEX = 4;
constexpr uint32_t IN_PONG_INDEX = 5;
constexpr uint32_t OUT_PING_INDEX = 6;
constexpr uint32_t OUT_PONG_INDEX = 7;
constexpr uint32_t PING_PONG_NUM = 2;
constexpr uint32_t UB_SIZE = 256 * 1024;
}  // namespace MatmulLayerNormReduceND
