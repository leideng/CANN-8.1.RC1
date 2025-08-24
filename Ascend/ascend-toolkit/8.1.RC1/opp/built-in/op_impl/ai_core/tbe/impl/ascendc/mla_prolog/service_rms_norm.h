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
 * \file service_rms_norm.h
 * \brief
 */

#ifndef SERVICE_RMS_NORM_H
#define SERVICE_RMS_NORM_H

#include "mla_prolog_comm.h"
#include "service_dynamic_quant.h"

namespace MlaProlog {

/**
 * @brief RmsNorm 只支持对单个batch的处理，row为1，col为512或1536
 * @param inputGm 输入tensor，[B, S1, H]，dtype支持bf16或fp32
 * @param gammaGm 输入tensor，[H, ]，dtype只支持bf16
 * @param reciprocal
 * @param epsilon
 * @param row 待处理的行数，对应S1
 * @param col 待处理的列数，对应H
 * @param shareTmpUb 临时buffer，input为fp32时，buffer大小为(2*row*col+8)*sizeof(fp32)+row*col*sizeof(bf16)；input为bf16时，buffer大小为(2*row*col+8)*sizeof(fp32)+2*row*col*sizeof(bf16)
 * @param outputLocal 输出tensor，[B, S1, H], dtype只支持bf16
 */

// 只支持对单个batch的处理 row为1
template <typename T1, typename T2, typename T3, typename C, typename O>
__aicore__ inline void RmsNormDynamicQuant(const GlobalTensor<T1>& inputGm, 
                               const GlobalTensor<T2>& gammaGm, const GlobalTensor<T3>& smoothGm,
                               float reciprocal, float epsilon, float &outputScaleValue, int64_t row, int64_t col,
                               LocalTensor<uint8_t>& shareTmpUb, LocalTensor<O>& outputLocal) {
    int64_t cnt = row * col;
    uint32_t gammaLocalOffset = (2 * col + 8) * 2;
    uint32_t smoothLocalOffset = 3 * col + 8;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();

    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1); // wait for vector operations to finish
    // load input [1, col]
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(col * sizeof(T1)), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    if constexpr (std::is_same<T1, float>::value) {
        DataCopyPad(xFp32Local, inputGm, copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    } else {
        LocalTensor<T1> inputLocal = shareTmpUb.ReinterpretCast<T1>()[gammaLocalOffset + col];
        DataCopyPad(inputLocal, inputGm, copyParams, padParams);

        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // Cast input to fp32 [1, col]
        Cast(xFp32Local, inputLocal, RoundMode::CAST_NONE, cnt);
        pipe_barrier(PIPE_V);
    }

    // load gamma [col]
    LocalTensor<T2> gammaLocal = shareTmpUb.ReinterpretCast<T2>()[gammaLocalOffset];
    DataCopy(gammaLocal, gammaGm, cnt);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

    // load smooth input [1, col]
    LocalTensor<C> smoothLocal = shareTmpUb.ReinterpretCast<C>()[smoothLocalOffset];
    DataCopyExtParams copyParamsSmooth{1, static_cast<uint32_t>(col * sizeof(T3)), 0, 0, 0};
    DataCopyPadExtParams<T3> padParamsSmooth{false, 0, 0, 0};
    DataCopyPad(smoothLocal, smoothGm, copyParamsSmooth, padParamsSmooth);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);

    // Calc: xSquare = xFp32 * xFp32, xSquare shape is [1, col]
    if constexpr (std::is_same<T1, float>::value) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    LocalTensor<C> xSquareLocal = xFp32Local[col];
    pipe_barrier(PIPE_V);
    Mul(xSquareLocal, xFp32Local, xFp32Local, cnt);
    pipe_barrier(PIPE_V);

    uint64_t repeatTimesAdd = static_cast<uint64_t>(cnt) >> 6;
    Add(xSquareLocal, xSquareLocal[64], xSquareLocal, 64, repeatTimesAdd - 1, {1, 1, 1, 0, 8, 0});
    pipe_barrier(PIPE_V);
    LocalTensor<C> xSumLocal = xFp32Local[col * 2];
    pipe_barrier(PIPE_V);
    WholeReduceSum(xSumLocal, xSquareLocal, 1 * 64, 1, 8, 1, 8);

    // Calc: xSum = xSum * reciprocal
    pipe_barrier(PIPE_V);
    Muls<C>(xSumLocal, xSumLocal, reciprocal, 1);
    pipe_barrier(PIPE_V);

    // Calc: xSum = xSum + epsilon
    pipe_barrier(PIPE_V);
    Adds<C>(xSumLocal, xSumLocal, epsilon, 1);
    pipe_barrier(PIPE_V);

    // Calc: xSum = sqrt(xSum)
    pipe_barrier(PIPE_V);
    Sqrt(xSumLocal, xSumLocal, 1);
    pipe_barrier(PIPE_V);

    // Calc: xSquare[1, 8] = brc(xSum[1,1])
    pipe_barrier(PIPE_V);
    Brcb(xSquareLocal, xSumLocal, 1, {1, 1});
    pipe_barrier(PIPE_V);

    // Calc: xFp32Local = xFp32Local / xSquareLocal
    pipe_barrier(PIPE_V);
    uint64_t mask[2] = {UINT64_MAX, UINT64_MAX};
    Div(xFp32Local, xFp32Local, xSquareLocal, mask, col / 64, {1, 1, 0, 8, 8, 0});
    pipe_barrier(PIPE_V);

    // Cast gammaLocal to xSquareLocal (bf16 -> fp32) [col]
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    pipe_barrier(PIPE_V);
    Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, col);
    pipe_barrier(PIPE_V);

    // Calc: xFp32Local = xFp32Local * xSquareLocal [1, col] * [col]
    Mul(xFp32Local, xFp32Local, xSquareLocal, col);
    pipe_barrier(PIPE_V);

    // calc DynamicQuant
    // Calc: xFp32Local = xFp32Local * smoothLocal, xFp32Local shape is [1, col]
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    LocalTensor<C> xScaleLocal = xFp32Local[col];
    Mul(xFp32Local, xFp32Local, smoothLocal, cnt);
    pipe_barrier(PIPE_V);

    Abs(xScaleLocal, xFp32Local, cnt);
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
    // Calc: xFp32Local = xFp32Local * scaleRecip
    Muls(xFp32Local, xFp32Local, scaleRecip, cnt);
    pipe_barrier(PIPE_V);
    // Calc: outputLocal = CastFloatToInt8(smoothLocal) [CastPath: float->int32->half->int8]
    LocalTensor<int32_t> tempInt32 = shareTmpUb.ReinterpretCast<int32_t>();
    LocalTensor<half> tempHalf = shareTmpUb.ReinterpretCast<half>();
    Cast(tempInt32, xFp32Local, RoundMode::CAST_RINT, cnt);
    pipe_barrier(PIPE_V);
    SetDeqScale(static_cast<half>(1.0));
    pipe_barrier(PIPE_V);
    Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, cnt);
    pipe_barrier(PIPE_V);
    Cast(outputLocal, tempHalf, RoundMode::CAST_TRUNC, cnt);
    pipe_barrier(PIPE_V);
}

// 只支持对单个batch的处理 row为1
template <typename T1, typename T2, typename C, typename O>
__aicore__ inline void RmsNorm(const GlobalTensor<T1>& inputGm, const GlobalTensor<T2>& gammaGm, 
                               float reciprocal, float epsilon, float quantScale, int64_t row, int64_t col,
                               LocalTensor<uint8_t>& shareTmpUb, LocalTensor<O>& outputLocal) {
    int64_t cnt = row * col;
    uint32_t gammaLocalOffset = (2 * col + 8) * 2;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();

    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1); // wait for vector operations to finish
    // load input [1, col]
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(col * sizeof(T1)), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    if constexpr (std::is_same<T1, float>::value) {
        DataCopyPad(xFp32Local, inputGm, copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    } else {
        LocalTensor<T1> inputLocal = shareTmpUb.ReinterpretCast<T1>()[gammaLocalOffset + col];
        DataCopyPad(inputLocal, inputGm, copyParams, padParams);

        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // Cast input to fp32 [1, col]
        Cast(xFp32Local, inputLocal, RoundMode::CAST_NONE, cnt);
        pipe_barrier(PIPE_V);
    }

    // load gamma [col]
    LocalTensor<T2> gammaLocal = shareTmpUb.ReinterpretCast<T2>()[gammaLocalOffset];
    DataCopy(gammaLocal, gammaGm, cnt);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

    // Calc: xSquare = xFp32 * xFp32, xSquare shape is [1, col]
    if constexpr (std::is_same<T1, float>::value) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    LocalTensor<C> xSquareLocal = xFp32Local[col];
    pipe_barrier(PIPE_V);
    Mul(xSquareLocal, xFp32Local, xFp32Local, cnt);
    pipe_barrier(PIPE_V);

    uint64_t mask[2] = {UINT64_MAX, UINT64_MAX};
    if (col == 512) {
        // Calc: xSquare[1, 512] -> [1, 256] -> [1, 128] -> [1, 64]
        pipe_barrier(PIPE_V);
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[64], mask, 4, {1, 1, 1, 8, 16, 16}); // [1, 256]
        pipe_barrier(PIPE_V);
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[64], mask, 2, {1, 1, 1, 8, 16, 16}); // [1, 128]
        pipe_barrier(PIPE_V);
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[64], mask, 1, {1, 1, 1, 8, 16, 16}); // [1, 64]
        pipe_barrier(PIPE_V);
    } else {
        // Calc: xSquare[1, 1536] -> [1, 768] -> [1, 384] -> [1, 192]
        pipe_barrier(PIPE_V);
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[64], mask, 12, {1, 1, 1, 8, 16, 16}); // [1, 768]
        pipe_barrier(PIPE_V);
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[64], mask, 6, {1, 1, 1, 8, 16, 16}); // [1, 384]
        pipe_barrier(PIPE_V);
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[64], mask, 3, {1, 1, 1, 8, 16, 16}); // [1, 192]
        pipe_barrier(PIPE_V);
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[64], mask, 1, {1, 1, 1, 8, 16, 16});
        pipe_barrier(PIPE_V);
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[128], mask, 1, {1, 1, 1, 8, 16, 16});
        pipe_barrier(PIPE_V);
    }

    // Calc: xSum = [1, 1]
    LocalTensor<C> xSumLocal = xFp32Local[col * 2];
    pipe_barrier(PIPE_V);
    WholeReduceSum(xSumLocal, xSquareLocal, 1 * 64, 1, 1, 1, 8);
    pipe_barrier(PIPE_V);

    // Calc: xSum = xSum * reciprocal
    pipe_barrier(PIPE_V);
    Muls<C>(xSumLocal, xSumLocal, reciprocal, 1);
    pipe_barrier(PIPE_V);

    // Calc: xSum = xSum + epsilon
    pipe_barrier(PIPE_V);
    Adds<C>(xSumLocal, xSumLocal, epsilon, 1);
    pipe_barrier(PIPE_V);

    // Calc: xSum = sqrt(xSum)
    pipe_barrier(PIPE_V);
    Sqrt(xSumLocal, xSumLocal, 1);
    pipe_barrier(PIPE_V);

    // Calc: xSquare[1, 8] = brc(xSum[1,1])
    pipe_barrier(PIPE_V);
    Brcb(xSquareLocal, xSumLocal, 1, {1, 1});
    pipe_barrier(PIPE_V);

    // Calc: xFp32Local = xFp32Local / xSquareLocal
    pipe_barrier(PIPE_V);
    Div(xFp32Local, xFp32Local, xSquareLocal, mask, col / 64, {1, 1, 0, 8, 8, 0});
    pipe_barrier(PIPE_V);

    // Cast gammaLocal to xSquareLocal (bf16 -> fp32) [col]
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    pipe_barrier(PIPE_V);
    Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, col);
    pipe_barrier(PIPE_V);

    // Calc: xFp32Local = xFp32Local * xSquareLocal [1, col] * [col]
    pipe_barrier(PIPE_V);
    if constexpr (std::is_same<C, O>::value) {
        Mul(outputLocal, xFp32Local, xSquareLocal, col);
        pipe_barrier(PIPE_V);
    } else {
        Mul(xFp32Local, xFp32Local, xSquareLocal, col);
        pipe_barrier(PIPE_V);
    
        // Cast xFp32 to outputLocal
        pipe_barrier(PIPE_V);
        Cast(outputLocal, xFp32Local, RoundMode::CAST_RINT, cnt);
        pipe_barrier(PIPE_V);
    }
}

template <typename T, typename C, typename O>
__aicore__ inline void RmsNormPostQuantPerChannel(const LocalTensor<T> &inputLocal, const GlobalTensor<C> &quantScaleGm,
                                                  int64_t row, int64_t col, const LocalTensor<uint8_t> &shareTmpUb,
                                                  const LocalTensor<O> outLocal)
{
    int64_t cnt = row * col;
    LocalTensor<C> quantScaleLocal = shareTmpUb.ReinterpretCast<C>();
    DataCopy(quantScaleLocal, quantScaleGm, cnt);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

    LocalTensor<C> inFp32;
    if constexpr (std::is_same<T, float>::value){
        inFp32 = inputLocal;
    } else {
        inFp32 = shareTmpUb.ReinterpretCast<C>()[cnt];
        Cast(inFp32, inputLocal, RoundMode::CAST_NONE, cnt);
        pipe_barrier(PIPE_V);
    }

    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Mul(inFp32, inFp32, quantScaleLocal, cnt);
    pipe_barrier(PIPE_V);

    LocalTensor<int32_t> tmpInt32 = shareTmpUb.ReinterpretCast<int32_t>();
    LocalTensor<half> tmpHalf = shareTmpUb.ReinterpretCast<half>()[cnt];
    Cast(tmpInt32, inFp32, RoundMode::CAST_RINT, cnt);
    pipe_barrier(PIPE_V);
    SetDeqScale(static_cast<half>(1.0));
    pipe_barrier(PIPE_V);
    Cast(tmpHalf, tmpInt32, RoundMode::CAST_ROUND, cnt);
    pipe_barrier(PIPE_V);

    Cast(outLocal, tmpHalf, RoundMode::CAST_TRUNC, cnt);
    pipe_barrier(PIPE_V);
}
// bf16, bf16, float, bf16, int8
// 只支持对单个batch的处理 row为1
template <typename T1, typename T2, typename C, typename O1, typename O2>
__aicore__ inline void RmsNormWithPostQuant(const GlobalTensor<T1> &inputGm, const GlobalTensor<T2> &gammaGm,
                                            float reciprocal, float epsilon, float quantScale, int64_t row, int64_t col,
                                            LocalTensor<uint8_t> &shareTmpUb, LocalTensor<O2> &outputLocal,
                                            GlobalTensor<float> smoothGm, GlobalTensor<float> &quantScaleGm,
                                            float &dequantScale, bool isDynamic = false)
{
    //dynamic quant 接口约束，rmsnorm 结果必须为 float
    LocalTensor<float> tmpOut = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<uint8_t> sharedBuf = shareTmpUb.ReinterpretCast<uint8_t>()[row * col * sizeof(float)];
    
    if (isDynamic) {
        RmsNormDynamicQuant<T1, T2, float, C, O2>(inputGm, gammaGm, smoothGm, reciprocal, epsilon, dequantScale, row, col, sharedBuf, outputLocal);
    } else {
        RmsNorm<T1, T2, C, float>(inputGm, gammaGm, reciprocal, epsilon, quantScale, row, col, sharedBuf, tmpOut);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        RmsNormPostQuantPerChannel(tmpOut, quantScaleGm, row, col, sharedBuf, outputLocal);
    }
}


} // namespace MlaProlog

#endif