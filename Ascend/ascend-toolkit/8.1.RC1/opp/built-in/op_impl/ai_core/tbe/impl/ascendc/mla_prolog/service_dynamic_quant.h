/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file service_dynamic_quant.h
 * \brief
 */

#ifndef SERVICE_DYNAMIC_QUANT_H
#define SERVICE_DYNAMIC_QUANT_H

#include "mla_prolog_comm.h"

namespace MlaProlog {

/**
 * @brief DynamicQuant 只支持对单个batch的处理， row为1， col为512或1536
 * @param inputLocal 输入tensor，[B, S1, H]，dtype只支持fp32
 * @param smoothGm 输入smooth tensor， [H, ]，dtype只支持fp32
 * @param row 待处理的行数，对应S1
 * @param col 待处理的列数，对应H
 * @param shareTmpUb 临时buffer
 * @param outputScaleValue 当前行对应的deQuantScaleValue
 * @param outputLocal 输出tensor，[B, S1, H], dtype只支持int8
 */

// 只支持对单个batch的处理 row为1
//  T1 = rmsNormCqOutputType;
//  T2 = dynamicQuantCqSmoothType;
//  T3 = dynamicQuantCqComputeType;
//  T4 = dynamicQuantCqScaleType;
//  T5 = dynamicQuantCqOutputType;
template <typename T1, typename T2, typename C, typename S, typename O>
__aicore__ inline void DynamicQuant(const LocalTensor<T1>& inputLocal, const GlobalTensor<T2>& smoothGm,
                                    int64_t row, int64_t col, LocalTensor<uint8_t>& shareTmpUb,
                                    float &outputScaleValue, LocalTensor<O>& outputLocal) {
    int64_t cnt = row * col;
    int64_t smoothLocalOffset = 2 * col + 8;

    // load smooth input [1, col]
    LocalTensor<C> smoothLocal = shareTmpUb.ReinterpretCast<C>()[smoothLocalOffset];
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(col * sizeof(T2)), 0, 0, 0};
    DataCopyPadExtParams<T2> padParams{false, 0, 0, 0};
    DataCopyPad(smoothLocal, smoothGm, copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    LocalTensor<C> xSmoothLocal = shareTmpUb.ReinterpretCast<C>();
    LocalTensor<C> xScaleLocal = xSmoothLocal[col];
    // Calc: xSmoothLocal = inputLocal * smoothLocal, xSmoothLocal shape is [1, col]
    Mul(xSmoothLocal, inputLocal, smoothLocal, cnt);
    pipe_barrier(PIPE_V);
    // Calc: xScaleLocal = Abs(xSmoothLocal), xSmoothLocal shape is [1, col]
    Abs(xScaleLocal, xSmoothLocal, cnt);
    pipe_barrier(PIPE_V);
    // Do Inplace ReduceMax: xScaleLocal[0] stores maxRowValue(xScaleLocal)
    // 64 = fp32 element nums per repeat; calcNum >> 6 = calcNum / 64;
    uint64_t repeatTimes = static_cast<uint64_t>(cnt) >> 6;
    Max(xScaleLocal, xScaleLocal[64], xScaleLocal, 64, repeatTimes - 1, {1, 1, 1, 0, 8, 0});
    pipe_barrier(PIPE_V);
    WholeReduceMax(xScaleLocal, xScaleLocal, 64, 1, 8, 1, 8);
    pipe_barrier(PIPE_V);
    // Construct: maxInt8Tensor = [127.0, 127.0, ..., 127.0], maxInt8Tensor shape is [1, 8]
    LocalTensor<C> maxInt8Tensor = shareTmpUb.ReinterpretCast<C>()[2 * col];
    Duplicate<C>(maxInt8Tensor, static_cast<C>(127.0), 8);
    pipe_barrier(PIPE_V);
    // Calc: xScaleLocal[0] = 127.0 / xScaleLocal[0]
    Div(xScaleLocal, maxInt8Tensor, xScaleLocal, 8);
    SetFlag<HardEvent::V_S>(EVENT_ID1);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    // Calc: deQuantScaleLocal[rowIdx] = 1 / scaleRecip
    C scaleRecip = xScaleLocal.GetValue(0);
    outputScaleValue = 1 / scaleRecip;
    // Calc: xSmoothLocal = xSmoothLocal * scaleRecip
    Muls(xSmoothLocal, xSmoothLocal, scaleRecip, cnt);
    pipe_barrier(PIPE_V);
    // Calc: outputLocal = CastFloatToInt8(smoothLocal) [CastPath: float->int32->half->int8]
    LocalTensor<int32_t> tempInt32 = shareTmpUb.ReinterpretCast<int32_t>();
    LocalTensor<half> tempHalf = shareTmpUb.ReinterpretCast<half>();
    Cast(tempInt32, xSmoothLocal, RoundMode::CAST_RINT, cnt);
    pipe_barrier(PIPE_V);
    SetDeqScale(static_cast<half>(1.0));
    pipe_barrier(PIPE_V);
    Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, cnt);
    pipe_barrier(PIPE_V);
    Cast(outputLocal, tempHalf, RoundMode::CAST_TRUNC, cnt);
    pipe_barrier(PIPE_V);
}

}

#endif
