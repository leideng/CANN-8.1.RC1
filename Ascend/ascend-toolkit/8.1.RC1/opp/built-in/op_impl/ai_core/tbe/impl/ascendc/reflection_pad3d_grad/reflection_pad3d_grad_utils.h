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
 * \file reflection_pad3d_grad_utils.h
 * \brief
 */
#ifndef REFLECTION_PAD3D_GRAD_UTILS_H
#define REFLECTION_PAD3D_GRAD_UTILS_H
#include <typeinfo>
#include "kernel_operator.h"

class CopyOutParam {
public:
    int64_t dstOffset;
    int64_t srcOffset;
    int64_t calH;
    int64_t calW;
    int64_t offsetWidth;
    bool isAtomicAdd;
    int64_t srcStride;
    __aicore__ inline CopyOutParam(int64_t dstOffset_tmp, int64_t srcOffset_tmp, int64_t calH_tmp, int64_t calW_tmp, int64_t offsetWidth_tmp, bool isAtomicAdd_tmp, int64_t srcStride_tmp = 0) {
        dstOffset = dstOffset_tmp;
        srcOffset = srcOffset_tmp;
        calH = calH_tmp;
        calW = calW_tmp;
        offsetWidth = offsetWidth_tmp;
        isAtomicAdd = isAtomicAdd_tmp;
        srcStride = srcStride_tmp;
    }
};

class CopyInParam {
public:
    int64_t dstOffset;
    int64_t srcOffset;
    int64_t calH;
    int64_t calW;
    __aicore__ inline CopyInParam( int64_t dstOffset_tmp,  int64_t srcOffset_tmp,  int64_t calH_tmp,  int64_t calW_tmp) {
        dstOffset = dstOffset_tmp;
        srcOffset = srcOffset_tmp;
        calH = calH_tmp;
        calW = calW_tmp;
    }
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b) {
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
};
template <typename T1, typename T2>
__aicore__ inline T1 FloorDiv(T1 a, T2 b) {
    if (b <= 0) {
        return 0;
    }
    return (a) / (b);
};
template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b) {
    if (b <= 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
};
template <typename T1, typename T2>
__aicore__ inline T1 FloorAlign(T1 a, T2 b) {
    if (b <= 0) {
        return 0;
    }
    return (a) / b * b;
};

template <typename T>
__aicore__ inline T Mymax(T a, T b) {
    if (a > b) {
        return a;
    }
    return b;
};

#endif //REFLECTION_PAD3D_GRAD_UTILS_H