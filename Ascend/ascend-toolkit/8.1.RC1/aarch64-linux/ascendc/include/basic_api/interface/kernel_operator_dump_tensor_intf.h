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
 * \file kernel_operator_dump_tensor_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_DUMP_TENSOR_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_DUMP_TENSOR_INTERFACE_H
#include "kernel_tensor.h"

namespace AscendC {
template <typename T>
__aicore__ inline void DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize);
template <typename T>
__aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize);
template <typename T>
__aicore__ inline void DumpTensor(const LocalTensor<T>& tensor, uint32_t desc,
    uint32_t dumpSize, const ShapeInfo& shapeInfo);
template <typename T>
__aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc,
    uint32_t dumpSize, const ShapeInfo& shapeInfo);
template <typename T>
__aicore__ inline void DumpAccChkPoint(const LocalTensor<T> &tensor,
    uint32_t index, uint32_t countOff, uint32_t dumpSize);
template <typename T>
__aicore__ inline void DumpAccChkPoint(const GlobalTensor<T> &tensor,
    uint32_t index, uint32_t countOff, uint32_t dumpSize);
#ifndef ASCENDC_CPU_DEBUG
template <class... Args>
__aicore__ inline void PRINTF(__gm__ const char* fmt, Args&&... args);
template <class... Args>
__aicore__ inline void printf(__gm__ const char* fmt, Args&&... args);
#endif

// assert define
#undef assert
#ifdef ASCENDC_DUMP
#define assert(...) ASCENDC_DEBUG_ASSERT_IMPL(__VA_ARGS__)
#else
#define assert(...)
#endif

/***************内部定义time stamp id**************************
定义值范围: 0x000 - 0xfff

time stamp id按块分组，快说明如下:
TIME_STAMP_WRAP: NPU套壳函数中的时间戳打点
TIME_STAMP_TPIPE/BUFFER: TPIPE、BUFFER中的时间戳打点
TIME_STAMP_MATMUL: MATMUL相关时间戳打点
TIME_STAMP_TILING_DATA: TILING DATA模块时间戳打点
TIME_STAMP_MC2_START/END: MC2模块使用打点id范围

TimeStampId更新原则：每个分组新增ID不可改变原有定义的ID值！

***************************************************************/
enum class TimeStampId : uint32_t {
    TIME_STAMP_WRAP_FIRST = 0x000,
    TIME_STAMP_WRAP_MC2_CTX,
    TIME_STAMP_WRAP_WK_SPACE,
    TIME_STAMP_WRAP_INIT_DUMP,
    TIME_STAMP_WRAP_FFTS_ADDR,
    TIME_STAMP_WRAP_CLEAR_WK_SPAC,

    TIME_STAMP_TPIPE = 0x030,
    TIME_STAMP_BUFFER,

    TIME_STAMP_MATMUL_SERVER = 0x060,
    TIME_STAMP_MATMUL_SERVER_INIT,
    TIME_STAMP_MATMUL_SERVER_OBJ,
    TIME_STAMP_MATMUL_MATRIX_KFC,
    TIME_STAMP_MATMUL_CLIENT_KFC,
    TIME_STAMP_MATMUL_WAIT_EVE,
    TIME_STAMP_MATMUL_OBJ,

    TIME_STAMP_TILING_DATA = 0x090,
    TIME_STAMP_TILING_DATA_STRUCT,
    TIME_STAMP_TILING_DATA_MEMBER,

    // MC2 :0x1000-0x1fff
    TIME_STAMP_MC2_START = 0x1000,
    TIME_STAMP_MC2_END = 0x1fff,
 
    TIME_STAMP_MAX = 0xffff,
};

__aicore__ inline void PrintTimeStamp(uint32_t descId);
}  // namespace AscendC
#endif  // END OF ASCENDC_MODULE_OPERATOR_DUMP_TENSOR_INTERFACE_H

