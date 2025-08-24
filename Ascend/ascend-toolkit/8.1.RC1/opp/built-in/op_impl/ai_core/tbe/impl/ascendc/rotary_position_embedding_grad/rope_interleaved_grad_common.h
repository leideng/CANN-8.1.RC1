/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file rope_interleaved_grad_common.h
 * \brief
 */
#ifndef ROPE_INTERLEAVED_GRAD_COMMON_H
#define ROPE_INTERLEAVED_GRAD_COMMON_H
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t LOG_FP32_SIZE = 2;
constexpr int32_t FP32_DIVIDE_FP16 = 2;
constexpr int32_t LOG_BLOCK_FP32_NUM = 3;
constexpr int32_t BLOCK_FP32_NUM = 8;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t MINI_HEADIM_NUM = 32;
constexpr int32_t MASK_FP32 = 64;
constexpr int64_t MASK_INT32 = 64;
constexpr int32_t MASK_FP16 = 128;

#endif  // ROTATE_INTERLEAVED_GRAD_COMMON_H