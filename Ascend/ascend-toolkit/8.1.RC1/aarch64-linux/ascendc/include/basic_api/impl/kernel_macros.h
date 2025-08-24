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

/*!
 * \file kernel_macros.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_MACROS_H
#define ASCENDC_KERNEL_MACROS_H

#include <cstdint>
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#define ASSERT(x) assert(x)
#define DEBUG_CODE(T) T

#else

#ifndef ASSERT
#define ASSERT(x)
#endif
#define DEBUG_CODE(T)

#ifndef __aicore__
#define __aicore__ [aicore]
#endif // __aicore__
#if __CCE_AICORE__ == 200
#ifndef __BLOCK_LOCAL__
#define __BLOCK_LOCAL__ [[block_local]]
#endif // __BLOCK_LOCAL__
#else
#ifndef __WORKGROUP_LOCAL__
#define __WORKGROUP_LOCAL__ [[workgroup_local]]
#endif // __WORKGROUP_LOCAL__
#endif

#ifndef __BLOCK_LOCAL__
#define __BLOCK_LOCAL__ [[block_local]]
#endif // __BLOCK_LOCAL__

#ifndef inline
#define inline __inline__ __attribute__((always_inline))
#endif

#endif // ASCENDC_CPU_DEBUG

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 8
#endif

#ifndef QBUF_MAX_LEN
#define QBUF_MAX_LEN 64
#endif

#ifndef QBUFPOOL_MAX_LEN
#define QBUFPOOL_MAX_LEN 16
#endif

#ifndef MAX_MSG_COUNT
#define MAX_MSG_COUNT 64
#endif

#ifndef QBUF_L0A_RESERVED_LEN
#define QBUF_L0A_RESERVED_LEN 2
#endif

#ifndef QBUF_L0B_RESERVED_LEN
#define QBUF_L0B_RESERVED_LEN 2
#endif
#ifndef QBUF_TOTAL_RESERVED_LEN
#define QBUF_TOTAL_RESERVED_LEN 4
#endif

#ifndef TPIPE_MAX_TYPE
#define TPIPE_MAX_TYPE 4
#endif

#if (__CCE_AICORE__ == 100) || (__CCE_AICORE__ == 200) || (__CCE_AICORE__ == 220) ||  (__CCE_AICORE__ == 300)
#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif

namespace AscendC {
#if __CCE_AICORE__ >= 200 // Available for V200 and V210
constexpr int32_t QUE_MAX_EVENT = 8;
#else
constexpr int32_t QUE_MAX_EVENT = 4;
#endif
constexpr int32_t HF32_MODE_BIT = 46;
constexpr int32_t HF32_TRANS_MODE_BIT = 47;
constexpr int32_t MM_LAYOUT_MODE_BIT = 51;
constexpr int32_t LEAKY_RELU_MODE_BIT = 50;
constexpr int32_t CAST_MODE_BIT = 59;
} // namespace AscendC

#endif // ASCENDC_KERNEL_MACROS_H