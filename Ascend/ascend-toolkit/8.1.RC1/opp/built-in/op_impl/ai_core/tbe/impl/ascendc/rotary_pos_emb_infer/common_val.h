/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file common_val.h
 * \brief
 */

#ifndef COMMON_VAL_H
#define COMMON_VAL_H
const constexpr uint32_t NUM_TWO = 2; // 2
const constexpr uint32_t BLK_SIZE = 32; // 一个block字节数
const constexpr uint32_t ELE_NUM_FP16 = 16; // 一个block fp16元素个数
const constexpr uint32_t ELE_NUM_FP32 = 8; // 一个block字节数 fp32元素个数
const constexpr uint32_t MAX_LEN_FP16 = 8192; // 非fp16情况下最大长度（hiddensize）
const constexpr uint8_t DEFAULT_REPEAT_STRIDE = 8; // 默认stride, 8 * 32 = 256
const constexpr int64_t REG_910B = 48; // 饱和模式寄存器位置
const constexpr int64_t REG_310P = 53; // 饱和模式寄存器位置
const constexpr int64_t SLICE_SIZE = 4096; // 切片大小
const constexpr int64_t SLICE_SIZE_FP16 = 12288;
#endif