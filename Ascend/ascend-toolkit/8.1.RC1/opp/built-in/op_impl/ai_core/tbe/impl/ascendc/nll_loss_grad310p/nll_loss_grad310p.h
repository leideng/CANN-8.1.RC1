/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file nll_loss_grad310p.h
 * \brief
 */
#ifndef NLLLOSSGRAD_N_D_H
#define NLLLOSSGRAD_N_D_H

#include "kernel_operator.h"

namespace KernelNLLLossGrad
{
  using namespace AscendC;
  constexpr int32_t BUFFER_NUM = 1;
  constexpr static uint32_t NUM_PER_BLOCK_FLOAT16 = 16;
  constexpr static uint32_t NUM_PER_BLOCK_FLOAT32 = 8;
  constexpr static uint32_t UB_SIZE_CLEAR = 20 * 1024;

  template <typename T1, typename T2>
  class KernelNLLLossGradND
  {
  public:
    __aicore__ inline KernelNLLLossGradND(){};
    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR target, GM_ADDR weight,
                                GM_ADDR out, GM_ADDR totalweight,
                                const NLLLossGradTilingData *tilingData, GM_ADDR usrWorkspace);
    __aicore__ inline void Process();
    __aicore__ inline void Process_Column_Slice(uint32_t offsetLine, uint32_t lineCount);

  private:
    __aicore__ inline void CopyIn(uint32_t offsetLine, uint32_t lineCount, uint32_t offsetColumn, uint32_t columnCount);
    __aicore__ inline void Compute(uint32_t offsetLine, uint32_t lineCount, uint32_t offsetColumn, uint32_t columnCount);
    __aicore__ inline void CopyOut(uint32_t offsetLine, uint32_t lineCount, uint32_t offsetColumn, uint32_t columnCount);
    __aicore__ inline void GetTargetWeight(const LocalTensor<T1>& dstLocal, const GlobalTensor<T1>& weightGlobal, const LocalTensor<uint32_t>& targetLocal, uint32_t srcBaseAddr, uint32_t count);
    __aicore__ inline uint32_t GetNumPerBlock();
    __aicore__ inline void ClearGM(const GlobalTensor<T1>& dstGlobal, uint32_t loop, uint32_t baseN, uint32_t tailN, uint32_t tailCoreNum);

  private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradOutputQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> targetQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> weightGather;
    TBuf<TPosition::VECCALC> UbBuf;
    TBuf<TPosition::VECCALC> syncTmpBuf_;

    GlobalTensor<T1> gradOutputTensorsGM;
    GlobalTensor<T2> targetTensorsGM;
    GlobalTensor<T1> weightTensorsGM;
    GlobalTensor<T1> totalWeightTensorsGM;
    GlobalTensor<T1> outTensorsGM;
    GlobalTensor<int32_t> syncTmpSpaceGm_;

    int64_t blockIdx = 0;
    uint32_t totalLineNum = 0;
    uint32_t perCoreMaxLine = 0;
    uint32_t perCoreMaxColum = 0;
    uint32_t coreNum = 0;
    uint32_t sampleNum = 0;
    uint32_t sampleNumOld = 0;
    uint32_t blockOffset_line = 0;
    uint32_t blockOffset_colum = 0;
    uint32_t linePerCore = 0;
    uint32_t columPerCore = 0;
    int64_t reduction = 0;
    int64_t ignoreIndex = 0;
    T1 gradOutputMS = 0;
    T1 totalWeightMS = 0;

    int64_t clearBaseN;
    int64_t clearOutPutLoop;
    int64_t clearOutPutTailN;
    int64_t clearOutPutTailCoreNum;
    uint32_t syncLen = 0;
  };

  template <typename T1, typename T2>
  __aicore__ inline void KernelNLLLossGradND<T1, T2>::Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR target, GM_ADDR weight,
                                                           GM_ADDR out, GM_ADDR totalweight,
                                                           const NLLLossGradTilingData *tilingData, GM_ADDR usrWorkspace)
  {
    blockIdx = GetBlockIdx();
    reduction = tilingData->reduction;
    ignoreIndex = tilingData->ignoreIndex;
    gradOutputTensorsGM.SetGlobalBuffer((__gm__ T1 *)gradOutput);
    if (reduction == 1 || reduction == 2)
    {
      gradOutputMS = gradOutputTensorsGM.GetValue(0);
    }
    targetTensorsGM.SetGlobalBuffer((__gm__ T2 *)target);
    weightTensorsGM.SetGlobalBuffer((__gm__ T1 *)weight);
    totalWeightTensorsGM.SetGlobalBuffer((__gm__ T1 *)totalweight);
    if (reduction == 1)
    {
      totalWeightMS = totalWeightTensorsGM.GetValue(0);
    }
    outTensorsGM.SetGlobalBuffer((__gm__ T1 *)out);
    totalLineNum = tilingData->totalLineNum;
    perCoreMaxLine = tilingData->perCoreMaxLine;
    perCoreMaxColum = tilingData->perCoreMaxColum;
    // clear GM
    clearBaseN = tilingData->clearBaseN;
    clearOutPutLoop = tilingData->clearOutPutLoop;
    clearOutPutTailN = tilingData->clearOutPutTailN;
    clearOutPutTailCoreNum = tilingData->clearOutPutTailCoreNum;

    coreNum = tilingData->coreNum;
    sampleNum = tilingData->sampleNum;
    sampleNumOld = tilingData->sampleNumOld;
    linePerCore = ((totalLineNum / coreNum + (32 / sizeof(T1)) - 1) / (32 / sizeof(T1))) * (32 / sizeof(T1));
    if (sampleNum <= perCoreMaxColum) {
      perCoreMaxColum = sampleNum;
    } else {
      columPerCore = ((sampleNum / coreNum + (32 / sizeof(T1)) - 1) / (32 / sizeof(T1))) * (32 / sizeof(T1));
    }
    // soft Syncall
    syncLen = NUM_PER_BLOCK_FLOAT32 * GetBlockNum();
    syncTmpSpaceGm_.SetGlobalBuffer((__gm__ int32_t *)usrWorkspace, NUM_PER_BLOCK_FLOAT32 * GetBlockNum());
    pipe.InitBuffer(syncTmpBuf_, syncLen * sizeof(int32_t));
    pipe.InitBuffer(UbBuf, UB_SIZE_CLEAR);
    pipe.InitBuffer(gradOutputQueue, BUFFER_NUM, (perCoreMaxLine * sizeof(T1) + 31) / 32 * 32);
    pipe.InitBuffer(targetQueue, BUFFER_NUM, (perCoreMaxLine * sizeof(T2) + 31) / 32 * 32);
    pipe.InitBuffer(outQueue, BUFFER_NUM, (perCoreMaxLine * perCoreMaxColum * sizeof(T1) + 31) / 32 * 32);
    pipe.InitBuffer(weightGather, (perCoreMaxLine * sizeof(T1) + 31) / 32 * 32);
    blockOffset_line = blockIdx * linePerCore;
    if (blockIdx == coreNum - 1){
      linePerCore = totalLineNum - linePerCore * (coreNum - 1);
      columPerCore = sampleNum - columPerCore * (coreNum - 1);
    }
  }

  template <typename T1, typename T2>
  __aicore__ inline void KernelNLLLossGradND<T1, T2>::Process()
  {
    // 清空GM
    LocalTensor<int32_t> workLocal = syncTmpBuf_.Get<int32_t>();
    Duplicate<int32_t>(workLocal, 0, syncLen);
    DataCopy(syncTmpSpaceGm_, workLocal, syncLen);

    ClearGM(outTensorsGM, clearOutPutLoop, clearBaseN, clearOutPutTailN, clearOutPutTailCoreNum);

    SyncAll(syncTmpSpaceGm_, workLocal);
    if (blockIdx >= coreNum) {
        return;
    }
    uint32_t times = linePerCore / perCoreMaxLine;
    uint32_t reminder = linePerCore % perCoreMaxLine;
    for (uint32_t i = 0; i < times; i++)
    {
      Process_Column_Slice(blockOffset_line + i * perCoreMaxLine, perCoreMaxLine);
    }
    if (reminder > 0)
    {
      Process_Column_Slice(blockOffset_line + times * perCoreMaxLine, reminder);
    }
  }

  template <typename T1, typename T2>
  __aicore__ inline void KernelNLLLossGradND<T1, T2>::Process_Column_Slice(uint32_t offsetLine, uint32_t lineCount)
  {
    uint32_t lineCountGradOut = ((lineCount + (32 / sizeof(T1)) - 1) / (32 / sizeof(T1))) * (32 / sizeof(T1));
    uint32_t lineCountTarget = ((lineCount + (32 / sizeof(T2)) - 1) / (32 / sizeof(T2))) * (32 / sizeof(T2));
    if (lineCountTarget > lineCountGradOut)
    {
      lineCountGradOut = lineCountTarget;
    }
    if (sampleNum <= perCoreMaxColum) { 
      CopyIn(offsetLine, lineCountGradOut, (uint32_t)0, sampleNum);
      Compute(offsetLine,lineCountGradOut, (uint32_t)0, sampleNum);
      CopyOut(offsetLine, lineCount, (uint32_t)0, sampleNum);
    } else {
      uint32_t times_colum = sampleNum / perCoreMaxColum;
      uint32_t reminder_colum = sampleNum % perCoreMaxColum;
      for (uint32_t i = 0; i < times_colum; i++){
        CopyIn(offsetLine, lineCountGradOut, i * perCoreMaxColum, perCoreMaxColum);
        Compute(offsetLine,lineCountGradOut, i * perCoreMaxColum, perCoreMaxColum);
        CopyOut(offsetLine, lineCount, i * perCoreMaxColum, perCoreMaxColum);
      }
      if (reminder_colum > 0) {
        CopyIn(offsetLine, lineCountGradOut, times_colum * perCoreMaxColum, reminder_colum);
        Compute(offsetLine,lineCountGradOut, times_colum * perCoreMaxColum, reminder_colum);
        CopyOut(offsetLine, lineCount, times_colum * perCoreMaxColum, reminder_colum);
      }
    }
  }

  template <typename T1, typename T2>
  __aicore__ inline void KernelNLLLossGradND<T1, T2>::CopyIn(uint32_t offsetLine, uint32_t lineCount, uint32_t offsetColumn, uint32_t columnCount)
  {
    LocalTensor<T1> gradOutputLocal = gradOutputQueue.AllocTensor<T1>();
    if (reduction == 0) {
      DataCopy(gradOutputLocal, gradOutputTensorsGM[offsetLine], lineCount);
    }
    gradOutputQueue.EnQue(gradOutputLocal);
    LocalTensor<T2> targetLocal = targetQueue.AllocTensor<T2>();
    DataCopy(targetLocal, targetTensorsGM[offsetLine], lineCount);
    targetQueue.EnQue(targetLocal);
  }

  template <typename T1, typename T2>
  __aicore__ inline void KernelNLLLossGradND<T1, T2>::Compute(uint32_t offsetLine, uint32_t lineCount, uint32_t offsetColumn, uint32_t columnCount)
  {
    LocalTensor<T2> targetLocal = targetQueue.DeQue<T2>();
    LocalTensor<T1> weightGatherLocal = weightGather.Get<T1>();
    LocalTensor<T1> outLocal = outQueue.AllocTensor<T1>();
    LocalTensor<T1> gradOutputLocal = gradOutputQueue.DeQue<T1>();

    Muls(targetLocal, targetLocal, (int32_t)sizeof(T1), lineCount);
    LocalTensor<uint32_t> targetLocalu = targetLocal.template ReinterpretCast<uint32_t>();

    event_t eventIdVToS_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));  // 下面有scalar计算，要等vector计算结束, Get, SetValue
    SetFlag<HardEvent::V_S>(eventIdVToS_1);
    WaitFlag<HardEvent::V_S>(eventIdVToS_1);

    GetTargetWeight(/*dst =*/weightGatherLocal, /*src =*/weightTensorsGM[0], /*index =*/targetLocalu, /*src_offset =*/(uint32_t)0, /*count =*/lineCount);

    event_t eventIdSToV_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV_1);
    WaitFlag<HardEvent::S_V>(eventIdSToV_1);

    targetLocal = targetLocalu.ReinterpretCast<T2>();

    if (reduction == 0){
      Mul(gradOutputLocal, gradOutputLocal, weightGatherLocal, lineCount);
      PipeBarrier<PIPE_V>();
      Muls(gradOutputLocal, gradOutputLocal, (T1)(-1), lineCount);
      PipeBarrier<PIPE_V>();
      Duplicate<T1>(outLocal, 0, lineCount * columnCount);
    }
    if (reduction == 1 || reduction == 2){
      Muls(gradOutputLocal, weightGatherLocal, gradOutputMS, lineCount);
      PipeBarrier<PIPE_V>();
      Muls(gradOutputLocal, gradOutputLocal, (T1)(-1), lineCount);
      if (reduction == 1){
        PipeBarrier<PIPE_V>();
        Muls(gradOutputLocal, gradOutputLocal, (T1)((float)1 / (float)totalWeightMS), lineCount);
      }
      PipeBarrier<PIPE_V>();
      Duplicate<T1>(outLocal, 0, lineCount * columnCount);
    }

    event_t eventIdVToS_2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S)); 
    SetFlag<HardEvent::V_S>(eventIdVToS_2);
    WaitFlag<HardEvent::V_S>(eventIdVToS_2);

    for (uint32_t i = 0; i < lineCount; i++){
      uint32_t target_index = targetLocal.GetValue(i) / (int32_t)sizeof(T1);
      if (target_index < offsetColumn || target_index >= offsetColumn + columnCount) {
        continue;
      }
      if ((target_index) != ignoreIndex && (offsetLine + i) < totalLineNum){
        outLocal.SetValue(((i * columnCount) + (target_index - offsetColumn)), gradOutputLocal.GetValue(i));
      }
    }

    event_t eventS_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventS_MTE3);
    WaitFlag<HardEvent::S_MTE3>(eventS_MTE3);

    gradOutputQueue.FreeTensor(gradOutputLocal);
    targetQueue.FreeTensor(targetLocal);
    outQueue.EnQue<T1>(outLocal);
  }

  template <typename T1, typename T2>
  __aicore__ inline void KernelNLLLossGradND<T1, T2>::CopyOut(uint32_t offsetLine, uint32_t lineCount, uint32_t offsetColumn, uint32_t columnCount)
  {
    LocalTensor<T1> outLocal = outQueue.DeQue<T1>();

    SetAtomicAdd<T1>();
    for (uint32_t i = 0; i < lineCount; i++) {
      DataCopy(outTensorsGM[(offsetLine + i) * sampleNumOld + offsetColumn], outLocal[i * columnCount], columnCount);
    }
    SetAtomicNone();

    outQueue.FreeTensor(outLocal);
  }

  template <typename T1, typename T2>
  __aicore__ inline void KernelNLLLossGradND<T1, T2>::GetTargetWeight(const LocalTensor<T1>&dstLocal, const GlobalTensor<T1> &weightGlobal, const LocalTensor<uint32_t>&targetLocal, uint32_t srcBaseAddr, uint32_t count)
  {
    for (uint32_t i = 0; i < count; i++){
      dstLocal.SetValue(i, weightGlobal.GetValue(srcBaseAddr + (targetLocal.GetValue(i) / (int32_t)sizeof(T1))));
    }
  }

template <typename T1, typename T2>
__aicore__ inline void KernelNLLLossGradND<T1, T2>::ClearGM(const GlobalTensor<T1> &dstGlobal, uint32_t loop,
                                                        uint32_t baseN, uint32_t tailN, uint32_t tailCoreNum) {
  uint32_t offset = (loop * baseN + tailN) * blockIdx 
                  + (blockIdx < tailCoreNum ? blockIdx : tailCoreNum) * GetNumPerBlock();
  uint32_t tail = tailN + (blockIdx < tailCoreNum ? GetNumPerBlock() : 0);
  LocalTensor<uint8_t> clearUBLc = UbBuf.Get<uint8_t>();
  
  LocalTensor<T1> clearUb = clearUBLc.ReinterpretCast<T1>();

  SetMaskCount();
  if(loop > 0) {
    SetVectorMask<T1, MaskMode::COUNTER>(0, baseN);
  } else if(tail > 0) {
    SetVectorMask<T1, MaskMode::COUNTER>(0, tail);
  } else {
    SetMaskNorm();
    ResetMask();
    return;
  }
  Duplicate<T1, false>(clearUb, static_cast<T1>(0), MASK_PLACEHOLDER, 1,
                  DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);

  event_t eventIdVToMTE3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
  SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
  WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
  SetMaskNorm();
  ResetMask();

  for(int i = 0; i < loop; i++) {
    DataCopy<T1>(dstGlobal[offset], clearUb, baseN);
    offset += baseN;
  }
  if(tail > 0){
    DataCopy<T1>(dstGlobal[offset], clearUb, tail);
  }
}

template <typename T1, typename T2>
__aicore__ inline uint32_t KernelNLLLossGradND<T1, T2>::GetNumPerBlock(){
  if(std::is_same<T1, float>::value) {
    return NUM_PER_BLOCK_FLOAT32;
  }
  return NUM_PER_BLOCK_FLOAT16;
}

} // namespace KernelNLLLossGrad

#endif
