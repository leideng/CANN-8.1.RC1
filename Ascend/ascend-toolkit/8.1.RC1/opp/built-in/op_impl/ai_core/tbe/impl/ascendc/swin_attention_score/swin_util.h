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
 * \file swin_util.h
 * \brief
 */
#ifndef __SWIN_UTIL_H__
#define __SWIN_UTIL_H__

#include "kernel_operator.h"
#include "lib/matrix/matmul/tiling.h"
#include "kernel_tpipe.h"
#include "kernel_tensor.h"
#include "kernel_type.h"
#include "kernel_operator_intf.h"

template <typename T>
struct GetDstType {
    using Type = T;
};

template <>
struct GetDstType<float> {
    using Type = float;
};

template <>
struct GetDstType<half> {
    using Type = float;
};

template <>
struct GetDstType<int8_t> {
    using Type = int32_t;
};

#if __CCE_AICORE__ == 220
template <>
struct GetDstType<bfloat16_t> {
    using Type = float;
};
#endif

#define halfQuantParams (QuantMode_t)1
#define bf16QuantParams (QuantMode_t)16

namespace matmul {
using namespace AscendC;

constexpr int32_t QUEUE_DEPTH = 1;
constexpr int32_t NZ_MASK_VAlUE = 2;

constexpr int32_t FLOAT_FACTOR = 2;
constexpr int32_t B32_C0SIZE = 8;
constexpr int32_t B16_C0SIZE = 16;

template <typename SrcT> constexpr int32_t AuxGetFactor()
{
    if (sizeof(SrcT) == sizeof(float)) {
        return FLOAT_FACTOR;
    }
    return 1;
}

template <typename SrcT> constexpr int32_t AuxGetC0Size()
{
    if (sizeof(SrcT) == sizeof(float)) {
        return B32_C0SIZE;
    }
    return B16_C0SIZE;
}

__aicore__ inline void clearWorkspace(__gm__ uint8_t* workspace) {
    set_atomic_none();
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        set_mask_norm();
        set_l1_3d_size((uint64_t)0);
        set_padding((uint64_t)0);
    } else {
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        set_mask_norm();
    }
#endif

#ifdef __DAV_C220_CUBE__

    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    ffts_cross_core_sync(PIPE_MTE3, 0x321);
#endif
}


int32_t constexpr GetNdNzMask(CubeFormat dstFormat, CubeFormat srcFormat)
{
    if ((srcFormat == CubeFormat::ND) && (dstFormat == CubeFormat::NZ)) {
        return 1;
    } else if ((srcFormat == CubeFormat::NZ) && (dstFormat == CubeFormat::ND)) {
        return NZ_MASK_VAlUE;
    }
    return 0;
}

template <TPosition POSITION, CubeFormat FORMAT, typename TYPE> struct BatchMatmulType {
    constexpr static TPosition pos = POSITION;
    constexpr static CubeFormat format = FORMAT;
    using T = TYPE;
};
} // namespace matmul

#define N_SYNC 4

#endif