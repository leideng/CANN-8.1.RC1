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
 * \file dynamic_quant_moe_large_shape.h
 * \brief
 */
#ifndef DYNAMIC_MOE_QUANT_H
#define DYNAMIC_MOE_QUANT_H

#include "dynamic_quant_base.h"

namespace DynamicQuantNDOpt {
using namespace AscendC;

template <typename T1, typename T2, typename T3, typename yDtype>
class DynamicQuantMoeLargeShape : public DynamicQuantBase {
 public:
  __aicore__ inline DynamicQuantMoeLargeShape(TPipe* pipe) {
    pPipe = pipe;
  }

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_indexs, GM_ADDR y, GM_ADDR scale,
                              GM_ADDR offset, GM_ADDR workSpace, const DynamicQuantTilingData* __restrict tilingData) {
    ParseTilingData(tilingData);
    InitLargeShapeParams(offset);
    InitLargeShapeBaseBuffer();
    InitAndSetBuffer(x, smooth_scales, group_indexs, y, scale, offset);
  }

  __aicore__ inline void Process() {
    T3 baseGroupNum = 0;

    GroupCopyIn();
    groupLocal = groupQueue.DeQue<T3>();
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    baseGroupNum = groupLocal.GetValue(0);

    if (isAsymmetrical) {
      ProcessAsymmetrical(baseGroupNum);
    } else {
      ProcessSymmetrical(baseGroupNum);
    }
  }

  __aicore__ inline void ProcessSymmetrical(T3 baseGroupNum) {
    uint32_t smoothIndex = 0;
    uint32_t smoothBaseOffset = 0;
    uint32_t offsetBase = 0;
    uint32_t srcOffset = 0;
    float scale = 0.0;
    T3 realRowNum = 0;
    T3 groupNum = baseGroupNum;

    LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
    // 一次处理一行，每个核循环处理的次数
    for (uint32_t i = 0; i < loopCnt; i++) {
      offsetBase = i * tilingData_.rowLen;
      // 获取符合条件的smooth_scales的index
      realRowNum = baseRow + i + 1;
      if (groupNum < realRowNum) {
        smoothIndex = GetSmoothIndex(realRowNum, groupNum, smoothIndex + 1);
        smoothBaseOffset = smoothIndex * tilingData_.rowLen;
      }
      LoopProcess(offsetBase, smoothBaseOffset, scale);
      scaleLocal.SetValue(i, 1 / scale);
    }
    scaleQueue.EnQue<float>(scaleLocal);
    CopyScalesOut(loopCnt);
    groupQueue.FreeTensor(groupLocal);
  }

  __aicore__ inline void ProcessAsymmetrical(T3 baseGroupNum) {
    uint32_t smoothIndex = 0;
    uint32_t smoothBaseOffset = 0;
    uint32_t offsetBase = 0;
    uint32_t srcOffset = 0;
    float scale = 0.0;
    float offset = 0.0;
    T3 realRowNum = 0;
    T3 groupNum = baseGroupNum;

    LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
    LocalTensor<float> offsetLocal = offsetQueue.AllocTensor<float>();
    // 一次处理一行，每个核循环处理的次数
    for (uint32_t i = 0; i < loopCnt; i++) {
      offsetBase = i * tilingData_.rowLen;
      // 获取符合条件的smooth_scales的index
      realRowNum = baseRow + i + 1;
      if (groupNum < realRowNum) {
        smoothIndex = GetSmoothIndex(realRowNum, groupNum, smoothIndex + 1);
        smoothBaseOffset = smoothIndex * tilingData_.rowLen;
      }
      LoopProcessAsymmetrical(offsetBase, smoothBaseOffset, scale, offset);
      scaleLocal.SetValue(i, scale);
      offsetLocal.SetValue(i, offset);
    }
    scaleQueue.EnQue<float>(scaleLocal);
    CopyScalesOut(loopCnt);
    offsetQueue.EnQue<float>(offsetLocal);
    CopyOffsetOut(loopCnt);
    groupQueue.FreeTensor(groupLocal);
  }

 private:
  __aicore__ inline void InitAndSetBuffer(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_indexs, GM_ADDR y,
                                          GM_ADDR scale, GM_ADDR offset) {
    smoothGm.SetGlobalBuffer((__gm__ T2*)smooth_scales);
    pPipe->InitBuffer(smoothQueue, BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T2));
    groupGm.SetGlobalBuffer((__gm__ T3*)group_indexs);
    pPipe->InitBuffer(groupQueue, BUFFER_NUM, tilingData_.alignGroupNum * sizeof(T3));
    if (blockIdx < tilingData_.headCoreNum) {
      inGm.SetGlobalBuffer((__gm__ T1*)x + blockIdx * lenHead, lenHead);
      baseRow = blockIdx * rowPerHeadCore;
      outGm.SetGlobalBuffer((__gm__ yDtype*)y + blockIdx * outLenHead, outLenHead);
      scaleGm.SetGlobalBuffer((__gm__ float*)scale + baseRow, rowPerHeadCore);
      if (isAsymmetrical) {
        offsetGm.SetGlobalBuffer((__gm__ float*)offset + baseRow, rowPerHeadCore);
      }
    } else {
      inGm.SetGlobalBuffer(
          (__gm__ T1*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail, lenTail);
      baseRow = tilingData_.headCoreNum * rowPerHeadCore + (blockIdx - tilingData_.headCoreNum) * rowPerTailCore;
      outGm.SetGlobalBuffer(
          (__gm__ yDtype*)y + tilingData_.headCoreNum * outLenHead + (blockIdx - tilingData_.headCoreNum) * outLenTail,
          outLenTail);
      scaleGm.SetGlobalBuffer((__gm__ float*)scale + baseRow, rowPerTailCore);
      if (isAsymmetrical) {
        offsetGm.SetGlobalBuffer((__gm__ float*)offset + baseRow, rowPerTailCore);
      }
    }
    if (isAsymmetrical) {
      pPipe->InitBuffer(offsetQueue, BUFFER_NUM, sizeFloatLen * sizeof(float));
    }
    // innerLoopEle已经保证了32Bytes对齐
    pPipe->InitBuffer(inQueue, BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T1));
    pPipe->InitBuffer(outQueue, BUFFER_NUM, tilingData_.innerLoopEle * sizeof(yDtype));
    pPipe->InitBuffer(scaleQueue, BUFFER_NUM, sizeFloatLen * sizeof(float));
  }

  __aicore__ inline void LoopProcess(uint32_t offsetBase, uint32_t smoothBaseOffset, float& scale) {
    float maxUpdateValue = MIN_FLOAT_VALUE;
    uint32_t srcOffset = 0;
    uint32_t smoothOffset = 0;
    // 计算每一行的最大值
    for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
      srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
      smoothOffset = smoothBaseOffset + innerLoopIndex * tilingData_.innerLoopEle;
      ComputeMaxValue(srcOffset, smoothOffset, tilingData_.innerLoopEle, 0, maxUpdateValue);
    }
    // 如果核内切的有尾块
    if (tilingData_.innerLoopTail > 0) {
      srcOffset = offsetBase + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
      smoothOffset = smoothBaseOffset + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
      CopyInByEle(srcOffset, smoothOffset, tilingData_.innerLoopTail, rightPadding);
      ComputeTail(tilingData_.innerLoopTail, maxUpdateValue);
      CopyOut(srcOffset, tilingData_.innerLoopTail);
    }
    GetScale(maxUpdateValue, scale);
    // 量化计算
    for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
      srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
      smoothOffset = smoothBaseOffset + innerLoopIndex * tilingData_.innerLoopEle;
      QuantCompute(srcOffset, smoothOffset, scale, 0.0, tilingData_.innerLoopEle, 0);
    }
  }

  __aicore__ inline void LoopProcessAsymmetrical(uint32_t offsetBase, uint32_t smoothBaseOffset, float& scale,
                                                 float& offset) {
    float maxUpdateValue = MIN_FLOAT_VALUE;
    float minUpdateValue = MAX_FLOAT_VALUE;
    uint32_t srcOffset = 0;
    uint32_t smoothOffset = 0;
    // 计算每一行的最大值
    for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
      srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
      smoothOffset = smoothBaseOffset + innerLoopIndex * tilingData_.innerLoopEle;
      ComputeMaxAndMinValue(srcOffset, smoothOffset, tilingData_.innerLoopEle, 0, maxUpdateValue, minUpdateValue);
    }
    // 如果核内切的有尾块
    if (tilingData_.innerLoopTail > 0) {
      srcOffset = offsetBase + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
      smoothOffset = smoothBaseOffset + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
      ComputeMaxAndMinValue(srcOffset, smoothOffset, tilingData_.innerLoopTail, rightPadding, maxUpdateValue,
                            minUpdateValue);
    }

    GetScaleAndOffset(maxUpdateValue, minUpdateValue, scale, offset);

    // 量化计算
    for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
      srcOffset = offsetBase + innerLoopIndex * tilingData_.innerLoopEle;
      smoothOffset = smoothBaseOffset + innerLoopIndex * tilingData_.innerLoopEle;
      QuantCompute(srcOffset, smoothOffset, scale, offset, tilingData_.innerLoopEle, 0);
    }
    // 如果核内切的有尾块
    if (tilingData_.innerLoopTail > 0) {
      srcOffset = offsetBase + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
      smoothOffset = smoothBaseOffset + tilingData_.innerLoopTimes * tilingData_.innerLoopEle;
      QuantCompute(srcOffset, smoothOffset, scale, offset, tilingData_.innerLoopTail, rightPadding);
    }
  }

  __aicore__ inline uint32_t GetSmoothIndex(T3 realRowNum, T3& groupNum, uint32_t smoothIndex) {
    // 获取符合条件的smooth_scales的index
    for (size_t index = smoothIndex; index < tilingData_.groupNum; index++) {
      groupNum = groupLocal.GetValue(index);
      if (groupNum >= realRowNum) {
        return index;
      }
    }
    return smoothIndex;
  }

  __aicore__ inline void GroupCopyIn() {
    LocalTensor<T3> groupLocal = groupQueue.AllocTensor<T3>();
    uint8_t rightPadding = tilingData_.alignGroupNum - tilingData_.groupNum;
    DataCopyParams copyParams{1, (uint16_t)(tilingData_.groupNum * sizeof(T3)), 0, 0};
    DataCopyPadParams padParams{true, 0, rightPadding, 0};
    DataCopyPad(groupLocal, groupGm, copyParams, padParams);
    groupQueue.EnQue(groupLocal);
  }

  __aicore__ inline void CopyInByEle(uint32_t offset, uint32_t scaleOffset, uint32_t elementNum, uint8_t rightPadding) {
    LocalTensor<T1> inLocal = inQueue.AllocTensor<T1>();
    DataCopyParams copyParams{1, (uint16_t)(elementNum * sizeof(T1)), 0, 0};
    DataCopyPadParams padParams{true, 0, rightPadding, 0};
    DataCopyPad(inLocal, inGm[offset], copyParams, padParams);
    inQueue.EnQue(inLocal);

    LocalTensor<T2> smoothLocal = smoothQueue.AllocTensor<T2>();
    // 一次拷贝不完
    DataCopyParams copyParams1{1, (uint16_t)(elementNum * sizeof(T2)), 0, 0};
    DataCopyPadParams padParams1{true, 0, rightPadding, 0};
    DataCopyPad(smoothLocal, smoothGm[scaleOffset], copyParams1, padParams1);
    smoothQueue.EnQue(smoothLocal);
  }

  __aicore__ inline void CopyScalesOut(uint32_t element) {
    LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();
    DataCopyParams copyParams{1, (uint16_t)(element * sizeof(float)), 0, 0};
    DataCopyPad(scaleGm, scaleLocal, copyParams);
    scaleQueue.FreeTensor(scaleLocal);
  }

  __aicore__ inline void CopyOffsetOut(uint32_t element) {
    LocalTensor<float> offsetLocal = offsetQueue.DeQue<float>();
    DataCopyParams copyParams{1, (uint16_t)(element * sizeof(float)), 0, 0};
    DataCopyPad(offsetGm, offsetLocal, copyParams);
    offsetQueue.FreeTensor(offsetLocal);
  }

  __aicore__ inline void QuantCompute(uint32_t offset, uint32_t scaleOffset, float scale, float offsetOut,
                                      uint32_t elementNum, uint8_t rightPadding) {
    CopyInByEle(offset, scaleOffset, elementNum, rightPadding);
    Compute(scale, offsetOut, elementNum);
    CopyOut(offset, elementNum);
  }

  __aicore__ inline void ComputeMaxValue(uint32_t offset, uint32_t scaleOffset, uint32_t elementNum,
                                         uint8_t rightPadding, float& maxUpdateValue) {
    CopyInByEle(offset, scaleOffset, elementNum, rightPadding);
    ComputeGetMaxValue(elementNum, maxUpdateValue);
  }

  __aicore__ inline void ComputeTail(uint32_t elementNum, float& maxUpdateValue) {
    float scale = 0.0f;
    LocalTensor<T1> smoothLocal;
    LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
    LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
    LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
    LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();
    Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
    pipe_barrier(PIPE_V);
    if (tilingData_.hasSmooth) {
      smoothLocal = smoothQueue.DeQue<T1>();
      Cast(temp, smoothLocal, RoundMode::CAST_NONE, elementNum);
      pipe_barrier(PIPE_V);
      Mul(tempFp32, tempFp32, temp, elementNum);
      pipe_barrier(PIPE_V);
      smoothQueue.FreeTensor(smoothLocal);
    }
    Abs(temp, tempFp32, elementNum);
    pipe_barrier(PIPE_V);
    ReduceMaxInplace(temp, elementNum);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float maxValue = temp.GetValue(0);
    maxUpdateValue = (maxUpdateValue > maxValue) ? maxUpdateValue : maxValue;
    GetScale(maxUpdateValue, scale);
    Muls(tempFp32, tempFp32, scale, elementNum);
    pipe_barrier(PIPE_V);
    Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, elementNum);
    pipe_barrier(PIPE_V);
    AscendC::LocalTensor<half> tempHalfCast = temp.ReinterpretCast<half>();
    SetDeqScale(static_cast<half>(1.0));
    pipe_barrier(PIPE_V);
    Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, elementNum);
    pipe_barrier(PIPE_V);
    Cast(outLocal, tempHalfCast, RoundMode::CAST_TRUNC, elementNum);
    outQueue.EnQue<yDtype>(outLocal);
    inQueue.FreeTensor(inLocal);
  }

  __aicore__ inline void ComputeGetMaxValue(uint32_t elementNum, float& maxUpdateValue) {
    LocalTensor<T2> smoothLocal;
    LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
    LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();

    Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
    pipe_barrier(PIPE_V);
    smoothLocal = smoothQueue.DeQue<T2>();
    Cast(temp, smoothLocal, RoundMode::CAST_NONE, elementNum);
    pipe_barrier(PIPE_V);
    Mul(tempFp32, tempFp32, temp, elementNum);
    pipe_barrier(PIPE_V);
    smoothQueue.FreeTensor(smoothLocal);
    Abs(temp, tempFp32, elementNum);
    pipe_barrier(PIPE_V);
    ReduceMaxInplace(temp, elementNum);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float maxValue = temp.GetValue(0);
    maxUpdateValue = (maxUpdateValue > maxValue) ? maxUpdateValue : maxValue;
    inQueue.FreeTensor(inLocal);
  }

  __aicore__ inline void ComputeMaxAndMinValue(uint32_t offset, uint32_t scaleOffset, uint32_t elementNum,
                                               uint8_t rightPadding, float& maxUpdateValue, float& minUpdateValue) {
    CopyInByEle(offset, scaleOffset, elementNum, rightPadding);
    ComputeGetMaxAndMinValue(elementNum, maxUpdateValue, minUpdateValue);
  }

  __aicore__ inline void ComputeGetMaxAndMinValue(uint32_t elementNum, float& maxUpdateValue, float& minUpdateValue) {
    LocalTensor<T2> smoothLocal;
    LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
    LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();

    Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
    pipe_barrier(PIPE_V);
    smoothLocal = smoothQueue.DeQue<T2>();
    Cast(temp, smoothLocal, RoundMode::CAST_NONE, elementNum);
    pipe_barrier(PIPE_V);
    Mul(tempFp32, tempFp32, temp, elementNum);
    pipe_barrier(PIPE_V);
    smoothQueue.FreeTensor(smoothLocal);
    ReduceMax(temp, tempFp32, temp, elementNum, false);
    pipe_barrier(PIPE_V);
    float maxValue = temp.GetValue(0);
    maxUpdateValue = (maxUpdateValue > maxValue) ? maxUpdateValue : maxValue;
    ReduceMin(temp, tempFp32, temp, elementNum, false);
    pipe_barrier(PIPE_V);
    float minValue = temp.GetValue(0);
    minUpdateValue = (minUpdateValue > minValue) ? minValue : minUpdateValue;
    inQueue.FreeTensor(inLocal);
  }

  __aicore__ inline void Compute(float scale, float offset, uint32_t elementNum) {
    LocalTensor<T2> smoothScalsLocal;
    LocalTensor<T1> inLocal = inQueue.DeQue<T1>();
    LocalTensor<float> tempFp32 = tempCastUb.Get<float>();
    LocalTensor<yDtype> outLocal = outQueue.AllocTensor<yDtype>();
    AscendC::LocalTensor<float> temp = fp32_buf_.Get<float>();
    LocalTensor<int32_t> tempInt32 = fp32_buf_.Get<int32_t>();

    Cast(tempFp32, inLocal, RoundMode::CAST_NONE, elementNum);
    pipe_barrier(PIPE_V);
    smoothScalsLocal = smoothQueue.DeQue<T2>();
    Cast(temp, smoothScalsLocal, RoundMode::CAST_NONE, elementNum);
    pipe_barrier(PIPE_V);
    Mul(tempFp32, tempFp32, temp, elementNum);
    pipe_barrier(PIPE_V);
    smoothQueue.FreeTensor(smoothScalsLocal);
    if (isAsymmetrical) {
      Muls(tempFp32, tempFp32, 1 / scale, elementNum);
      pipe_barrier(PIPE_V);
      Adds(tempFp32, tempFp32, offset, elementNum);
    } else {
      Muls(tempFp32, tempFp32, scale, elementNum);
    }
    pipe_barrier(PIPE_V);
    Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, elementNum);
    pipe_barrier(PIPE_V);
    AscendC::LocalTensor<half> tempHalfCast = temp.ReinterpretCast<half>();
    SetDeqScale(static_cast<half>(1.0));
    pipe_barrier(PIPE_V);
    Cast(tempHalfCast, tempInt32, RoundMode::CAST_ROUND, elementNum);
    pipe_barrier(PIPE_V);
    Cast(outLocal, tempHalfCast, RoundMode::CAST_TRUNC, elementNum);
    outQueue.EnQue<yDtype>(outLocal);
    inQueue.FreeTensor(inLocal);
  }

  __aicore__ inline void CopyOut(uint32_t offset, uint32_t element) {
    LocalTensor<yDtype> outLocal = outQueue.DeQue<yDtype>();

    DataCopyExtParams copyParams{(uint16_t)1, (uint16_t)(element * sizeof(yDtype)), 0, 0, 0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
      copyParams.blockLen = copyParams.blockLen >> 1;
      uint32_t index = offset;
      DataCopyPad(outGm[index], outLocal, copyParams);
    } else {
      DataCopyPad(outGm[offset], outLocal, copyParams);
    }
    outQueue.FreeTensor(outLocal);
  }

 private:
  /* ascendc variable */
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
  TQue<QuePosition::VECIN, BUFFER_NUM> smoothQueue;
  TQue<QuePosition::VECIN, BUFFER_NUM> groupQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue, scaleQueue, offsetQueue;

  /* global memory address */
  GlobalTensor<T1> inGm;
  GlobalTensor<T2> smoothGm;
  GlobalTensor<T3> groupGm;
  GlobalTensor<yDtype> outGm;
  GlobalTensor<float> scaleGm, offsetGm;

  /* local memory address */
  LocalTensor<T3> groupLocal;

  /* variable */
  T3 baseRow = 0;  // 记录开始处理的行数
};
}  // namespace DynamicQuantNDOpt
#endif  // DYNAMIC_MOE_QUANT_H