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
 * \file apply_came_part3_common.h
 * \brief
 */
#ifndef _APPLY_CAME_PART3_COMMON_H_
#define _APPLY_CAME_PART3_COMMON_H_

#include "kernel_operator.h"

constexpr int64_t SCALAR_INPUT_SIZE = 8; // 32B / 4B
constexpr int64_t DEFAULT_QUEUE_BUFFE_SIZE = 2;
constexpr int64_t ONE_VECTOR_BLOCK_SIZE = 256;
constexpr int64_t ONE_BLOCK_SIZE = 32;
constexpr int64_t REP_BLOCK_STRIDE = 8;
constexpr int64_t FP16_ONE_BLOCK_COUNT = 16;
constexpr int64_t FP32_ONE_BLOCK_COUNT = 8;
constexpr int64_t BUFFER_SIZE = 3;
constexpr int64_t ONE_BLOCK_INT32_COUNT = 8;
constexpr int64_t CORE_NUM = 48;
constexpr int64_t SPLIT_PART = 2;
constexpr int64_t ONE_VECTOR_FP32_SIZE = 64;
constexpr int64_t MAX_REPEAT_TIME = 255;
constexpr int64_t INT64_ONE_BLOCK_COUNT = 4;
constexpr uint32_t DET_WORKSPACE_SIZE = 392; // 8 * 49
constexpr uint32_t DET_WORKSPACE_BYTE = 1568; // 32B * 49

struct CamePart3InOut {
  GM_ADDR u;
  GM_ADDR mIn;
  GM_ADDR eps;
  GM_ADDR beta1;
  GM_ADDR clipThreshold;
  GM_ADDR sumSquareU;
  GM_ADDR globalShape;
  GM_ADDR mOut;
  GM_ADDR sumUR;
  GM_ADDR sumUC;
  GM_ADDR sumURC;
};

#endif
