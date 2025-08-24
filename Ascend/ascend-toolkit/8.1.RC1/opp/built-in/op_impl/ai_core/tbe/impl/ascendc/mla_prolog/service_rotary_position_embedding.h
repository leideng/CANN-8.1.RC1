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
 * \file service_rotary_position_embedding.h
 * \brief
 */

#ifndef SERVICE_ROTARY_POSITION_EMBEDDING_H
#define SERVICE_ROTARY_POSITION_EMBEDDING_H

#include "mla_prolog_comm.h"

namespace MlaProlog {

// 只支持对单个batch的处理 row为1
template <typename T, typename C, typename O>
__aicore__ inline void RotaryPosEmb(const GlobalTensor<T>& inputGm, const LocalTensor<C>& cosLocal,
                                    const LocalTensor<C>& sinLocal, int64_t stride, int64_t row, int64_t col, 
                                    LocalTensor<uint8_t>& shareTmpUb, LocalTensor<O>& outputLocal,
                                    GlobalTensor<C> channelDeqScaleGm = GlobalTensor<C>(), float deQuantScale = -1) {

    int64_t cnt = row * col;

    // load kLocal [row, col]
    LocalTensor<T> kLocal = shareTmpUb.ReinterpretCast<T>();
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(row),
        static_cast<uint32_t>(col * sizeof(T)),
        static_cast<uint32_t>((stride - col) * sizeof(T)),
        0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(kLocal, inputGm, copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    int64_t baseOffset;
    if constexpr (std::is_same<T, int32_t>::value){
        baseOffset = cnt;
    } else {
        baseOffset = cnt / 2;
    }
    // cast input to fp32
    LocalTensor<C> kFp32Local = shareTmpUb.ReinterpretCast<C>()[baseOffset];
    Cast(kFp32Local, kLocal, RoundMode::CAST_NONE, cnt);
    pipe_barrier(PIPE_V);


    LocalTensor<C> kFp32ReArrLocal = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt];
    LocalTensor<C> kFp32OutputLocal = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 2];
    uint64_t rsvdCnt = 0;
    if constexpr (std::is_same<T, int32_t>::value) { // 反量化
        DataCopyExtParams copyParams1{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(C)),
                                      static_cast<uint32_t>((stride - col) * sizeof(C)), 0, 0};
        DataCopyPadExtParams<C> padParams1{false, 0, 0, 0};
        DataCopyPad(kFp32ReArrLocal, channelDeqScaleGm, copyParams1, padParams1); // 复用内存
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        Muls(kFp32Local, kFp32Local, deQuantScale, cnt);
        pipe_barrier(PIPE_V);
        Mul(kFp32Local, kFp32Local, kFp32ReArrLocal, cnt);
        pipe_barrier(PIPE_V);
    }
    LocalTensor<C> kFp32OutputLocalSinTmp = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 3];
    GatherMask(kFp32ReArrLocal, kFp32Local, 1, true, 
                col * row, {1, 1, 0, 0}, rsvdCnt);
    GatherMask(kFp32ReArrLocal[cnt >> 1], kFp32Local, 2, true, 
                col * row, {1, 1, 0, 0}, rsvdCnt);
    pipe_barrier(PIPE_V);
    uint8_t blockNumPerRow = col / (32 / sizeof(C));
    uint8_t blockNumPerRowHalf = blockNumPerRow >> 1;
    Mul<C, true>(kFp32OutputLocal, kFp32ReArrLocal, cosLocal, col >> 1, row,
                 {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    Mul<C, true>(kFp32OutputLocal[col >> 1], kFp32ReArrLocal[cnt >> 1], cosLocal[col >> 1], 
                 col >> 1, row, {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    Mul<C, true>(kFp32OutputLocalSinTmp, kFp32ReArrLocal[cnt >> 1], sinLocal, 
                 col >> 1, row, {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    Mul<C, true>(kFp32OutputLocalSinTmp[col >> 1], kFp32ReArrLocal, sinLocal[col >> 1], 
                 col >> 1, row, {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    pipe_barrier(PIPE_V);
    Add(kFp32OutputLocal, kFp32OutputLocal, kFp32OutputLocalSinTmp, cnt);
    pipe_barrier(PIPE_V);

    if constexpr (std::is_same<O,C>::value) {
        DataCopy(outputLocal, kFp32OutputLocal, cnt);
        PipeBarrier<PIPE_V>();
    } else {
        Cast(outputLocal, kFp32OutputLocal, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, typename C, typename O>
__aicore__ inline void RopePostQuantPerChannel(LocalTensor<T> &inputLocal, GlobalTensor<C> &quantScaleGm,
                                               int64_t stride, int64_t row, int64_t col,
                                               LocalTensor<uint8_t> &shareTmpUb, LocalTensor<O> &outputLocal)
{
    int64_t cnt = row * col;
    LocalTensor<C> quantScaleLocal = shareTmpUb.ReinterpretCast<C>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(C)),
                                 static_cast<uint32_t>((stride - col) * sizeof(C)), 0, 0};
    DataCopyPadExtParams<C> padParams{false, 0, 0, 0};
    DataCopyPad(quantScaleLocal, quantScaleGm, copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    LocalTensor<C> inFp32;
    if constexpr (std::is_same<T,float>::value) {
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
    pipe_barrier(PIPE_V);
    Cast(tmpInt32, inFp32, RoundMode::CAST_RINT, cnt);
    pipe_barrier(PIPE_V);
    SetDeqScale(static_cast<half>(1.0));
    pipe_barrier(PIPE_V);
    Cast(tmpHalf, tmpInt32, RoundMode::CAST_ROUND, cnt);
    pipe_barrier(PIPE_V);
    Cast(outputLocal, tmpHalf, RoundMode::CAST_TRUNC, cnt);
    pipe_barrier(PIPE_V);
}

}

#endif