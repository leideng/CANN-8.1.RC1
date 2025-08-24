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
 * \file quant_matmul_dequant_normal.h
 * \brief
 */
#ifndef _ASCENDC_QUANT_MATMUL_DEQUANT_NORMAL_H_
#define _ASCENDC_QUANT_MATMUL_DEQUANT_NORMAL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_matmul_dequant_gemv.h"
using namespace AscendC;

namespace AscendC {
class QuantMatmulDequantNormal : public QuantMatmulDequantGemv {
 public:
  __aicore__ inline QuantMatmulDequantNormal() {}

  __aicore__ inline void Process() {
    if(tilingData->dynamicQuant){
      ProcessXScale();
      MySyncAll();
    }
    //x quantize and ub->L1->L0A
    ProcessX();
    MySyncAll();
    // mm
    ProcessMM();
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale, 
                              GM_ADDR y, GM_ADDR usrWorkspace, const QuantMatmulDequantTilingData* __restrict qmmTiling) {
    // block_id                                              
    block_id = GetBlockIdx();
    tilingData = qmmTiling;

    TilingInKernel();

    // alloc eventID
    InitEventId();

    InitGlobalTensors(x, quantized_weight, weight_scale, bias, x_scale, x_offset, smooth_scale, y, usrWorkspace);

    InitLocalTensors();

    InitSyncWs();

    InitTransLocalLists();
  }
 protected:
  __aicore__ inline void InitGlobalTensors(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR bias, 
                                           GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale, GM_ADDR y, GM_ADDR usrWorkspace) {
    xZNGM.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(usrWorkspace), (tilingData->originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8 * NM_FRACTAL_INT8 * tilingData->originKAligned32);
    usrWorkspace += (tilingData->originM + NM_FRACTAL_INT8 - 1) / NM_FRACTAL_INT8 * NM_FRACTAL_INT8 * tilingData->originKAligned32;
    InitCommonGlobalTensors(x, quantized_weight, weight_scale, bias, x_scale, x_offset, smooth_scale, y, usrWorkspace);
  }

  __aicore__ inline void InitLocalTensors() {
    pipe.InitBuffer(UbBuf, UB_SIZE);
    LocalTensor<uint8_t> tmp = UbBuf.Get<uint8_t>();

    InitCommonLocalTensors(tmp);

    ubXHalfND = tmp.ReinterpretCast<half>();
    ubXHalfZN = ubXHalfND[tilingData->processXKBaseNMax * NM_FRACTAL_INT8];
    ubXInt8ZN = ubXHalfND[tilingData->processXKBaseNMax * NM_FRACTAL_INT8].ReinterpretCast<int8_t>();
    ubXFloatND = ubXHalfZN[tilingData->processXKBaseNMax * NM_FRACTAL_INT8].ReinterpretCast<float>();
    if(tilingData->smoothScale) {
      ubSmoothScale = ubXFloatND[tilingData->processXKBaseNMax * NM_FRACTAL_INT8].ReinterpretCast<half>();
      ubSmoothScaleFloat = ubSmoothScale[tilingData->processXKBaseNMax].ReinterpretCast<float>();
      if(tilingData->perToken) {
        ubXScaleFloat = ubSmoothScaleFloat[tilingData->processXKBaseNMax].ReinterpretCast<float>();
      }
    } else {
      if(tilingData->perToken) {
        ubXScaleFloat = ubXFloatND[tilingData->processXKBaseNMax * NM_FRACTAL_INT8].ReinterpretCast<float>();
      }
    }

    if(tilingData->perToken) {
      ubPertokenScaleRaw = tmp.ReinterpretCast<float>();
      ubPertokenScale = ubPertokenScaleRaw[baseM * L0C_FRACTAL * L0C_FRACTAL].ReinterpretCast<float>();
      if(tilingData->dynamicQuant) ubPertokenScaleRaw_ = ubPertokenScaleRaw[baseM * NUMBER_128].ReinterpretCast<float>();
      ubYInt32 = ubPertokenScale[baseM * L0C_FRACTAL * L0C_FRACTAL].ReinterpretCast<int32_t>();
    } else {
      ubYInt32 = tmp.ReinterpretCast<int32_t>();
    }
    ubYFloat = ubYInt32.ReinterpretCast<float>();
    ubYHalfNZ = ubYInt32.ReinterpretCast<half>();
    ubYHalfND = ubYHalfNZ[baseM * baseN * L0C_FRACTAL * L0C_FRACTAL].ReinterpretCast<half>();
    ubWScale = ubYHalfND[baseM * baseN * L0C_FRACTAL * L0C_FRACTAL].ReinterpretCast<float>();

    pipe.InitBuffer(L1Buf, L1_SIZE);
    l1W = L1Buf.Get<int8_t>();
    l1X = l1W[L0A_SIZE];
    pipe.InitBuffer(L0ABuf, L0A_SIZE);
    l0aX = L0ABuf.Get<int8_t>();
    pipe.InitBuffer(L0BBuf, L0B_SIZE);
    l0bW = L0BBuf.Get<int8_t>();
    pipe.InitBuffer(L0CBuf, L0C_SIZE);
    l0cY = L0CBuf.Get<int32_t>();
  }

  __aicore__ inline void InitTransLocalLists() {
    NDLocalList[0] = reinterpret_cast<uint64_t>(ubXHalfND.GetPhyAddr());
    ZNLocalList[0] = reinterpret_cast<uint64_t>(ubXHalfZN.GetPhyAddr());
    for(int32_t i = 1; i < NUMBER_16; i++) {
      NDLocalList[i]= NDLocalList[i - 1] + processXKBaseN * sizeof(half);
      ZNLocalList[i]= ZNLocalList[i - 1] + BLOCK_SIZE;
    }

    if(tilingData->perToken) {
      pertokenRawLocalList[0] = reinterpret_cast<uint64_t>(ubPertokenScaleRaw.GetPhyAddr());
      pertokenLocalList[0] = reinterpret_cast<uint64_t>(ubPertokenScale.GetPhyAddr());
      for(int32_t i = 1; i < NUMBER_16; i++) {
        pertokenRawLocalList[i]= pertokenRawLocalList[i - 1];
        pertokenLocalList[i]= pertokenLocalList[i - 1] + BLOCK_SIZE;
      }
      if(tilingData->dynamicQuant){
        pertokenDynamicRawLocalList[0] = reinterpret_cast<uint64_t>(ubPertokenScaleRaw_.GetPhyAddr());
        pertokenDynamicLocalList[0] = reinterpret_cast<uint64_t>(ubPertokenScaleRaw.GetPhyAddr());
        pertokenDynamicLocalList[1] = pertokenDynamicLocalList[0] + BLOCK_SIZE;
        pertokenDynamicLocalList[NUMBER_2] = reinterpret_cast<uint64_t>(ubPertokenScale.GetPhyAddr());
        for(int32_t i = 1; i < NUMBER_16; i++) {
          pertokenDynamicRawLocalList[i]= pertokenDynamicRawLocalList[i - 1] + BLOCK_SIZE;
        }
        for(int32_t i = NUMBER_3; i < NUMBER_16; i++) {
          pertokenDynamicLocalList[i]= pertokenDynamicLocalList[i - 1] + BLOCK_SIZE;
        }
      }
    }
  }

  __aicore__ inline void TilingInKernel() {
    //x:quantize and ND->ZN
    realM = tilingData->originM;
    fracM = tilingData->fracM;
    tailM = tilingData->tailM;

    processXKloop = tilingData->processXKloop;
    processXKloopPerfracM = tilingData->processXKloopPerfracM;
    processXKBaseN = tilingData->processXKBaseN;
    processXKTailN = tilingData->processXKTailN;
    // mm
    MCoreNum = tilingData->MCoreNum;
    NCoreNum = tilingData->NCoreNum;
    int32_t tmpM = ((int32_t)block_id / tilingData->NCoreNum) - tilingData->singleCoreMTail;
    int32_t tmpN = ((int32_t)block_id % tilingData->NCoreNum) - tilingData->singleCoreNTail;
    realSingleCoreM = tilingData->singleCoreM - (tmpM < 0 ? 0 : 1);
    realSingleCoreN = tilingData->singleCoreN - (tmpN < 0 ? 0 : 1);
    globalOffsetM = tilingData->singleCoreM * ((int32_t)block_id / tilingData->NCoreNum) - (tmpM > 0 ? tmpM : 0);
    globalOffsetN = tilingData->singleCoreN * ((int32_t)block_id % tilingData->NCoreNum) - (tmpN > 0 ? tmpN : 0);

    baseMNum = tilingData->baseMNum;
    baseNNum = tilingData->baseNNum;
    baseKNum = tilingData->baseKNum;
    baseM = (realSingleCoreM + baseMNum - 1) / baseMNum;
    baseN = (realSingleCoreN + baseNNum - 1) / baseNNum;
    baseMTail = (realSingleCoreM-1) % baseMNum + 1;
    baseNTail = (realSingleCoreN-1) % baseNNum + 1;

    baseK = (tilingData->fracK + baseKNum - 1) / baseKNum;
    baseKTail = (tilingData->fracK-1) % baseKNum + 1;
  }

  __aicore__ inline void ProcessX() {
    if(processXKloop == 0) {
      return;
    }

    DataCopyParams repeatParamsHalf, repeatParamsHalfScale, repeatParamsInt8, repeatParamsFloat;
    UnaryRepeatParams unaryParamsH2I8, unaryParamsH2F, unaryParamsF2I, unaryParams;
    TransDataTo5HDParams transDataParamsX;
    BinaryRepeatParams binaryParams;
    int32_t fracMIdx_ = -1;
    uint64_t mulAlignMask[NUMBER_2] = {tilingData->ubKMask, 0};
    float pertokenScale[NM_FRACTAL_INT8];
    unaryParamsH2I8.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParamsH2F.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    if(tilingData->dynamicQuant) {
      repeatParamsFloat.blockLen = NM_FRACTAL_INT8;
    } else {
      repeatParamsFloat.blockLen = NM_FRACTAL_INT8 / FLOAT_PERBLOCK;
    }
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[1]);
    SetMaskCount();
    for(int32_t i = 0; i < processXKloop; i++) {
      uint32_t totalIdx = (block_id * processXKloop + i);
      if(totalIdx >= (processXKloopPerfracM * fracM)) {
        totalIdx = processXKloopPerfracM * fracM - 1;
      }
      uint32_t fracMIdx = totalIdx / processXKloopPerfracM;
      uint32_t kIdx = totalIdx % processXKloopPerfracM;

      uint32_t mOffset = fracMIdx * NM_FRACTAL_INT8 * tilingData->originK;
      uint32_t mOffsetAlignedK = fracMIdx * NM_FRACTAL_INT8 * tilingData->originKAligned32;
      uint32_t realBaseKAligned32 = processXKBaseN;
      realBaseKAligned32 -= (kIdx >= processXKTailN) ? K_FRACTAL_INT8 : 0;
      if(realBaseKAligned32 == 0) continue;
      //???
      uint32_t realBaseK = realBaseKAligned32;
      if(realBaseK == INT8_PERBLOCK) {
        if(kIdx == (processXKTailN - 1)) realBaseK -= tilingData->originKAligned32 - tilingData->originK;
      } else if(kIdx == processXKloopPerfracM-1) realBaseK -= tilingData->originKAligned32 - tilingData->originK;
      uint32_t kOffset = kIdx * processXKBaseN;
      kOffset -= (kIdx > processXKTailN) ? K_FRACTAL_INT8 * (kIdx - processXKTailN) : 0;
      uint32_t kOffsetZN = kOffset * NM_FRACTAL_INT8;
      uint32_t realBaseM = (fracMIdx == (fracM - 1)) ? tailM : NM_FRACTAL_INT8;

      if(tilingData->perToken && fracMIdx_ != (int)fracMIdx) {
        uint32_t pertoken_offset = fracMIdx * NM_FRACTAL_INT8 * (tilingData->dynamicQuant ? FLOAT_PERBLOCK : 1);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        DataCopy<float>(ubXScaleFloat, xScaleGm[pertoken_offset], repeatParamsFloat);
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS[0]);
      }

      WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
      if(tilingData->smoothScale) {
        repeatParamsHalfScale.blockLen = realBaseK / HALF_PERBLOCK;
        DataCopy<half>(ubSmoothScale, smoothScaleGm[kOffset], repeatParamsHalfScale);
      }
      repeatParamsHalf.blockCount = realBaseM;
      repeatParamsHalf.blockLen = realBaseK / HALF_PERBLOCK;
      repeatParamsHalf.srcStride = (tilingData->originK - realBaseK) / HALF_PERBLOCK;
      repeatParamsHalf.dstStride = (processXKBaseN - realBaseK) / HALF_PERBLOCK;
      DataCopy<half>(ubXHalfND, xGm[kOffset + mOffset], repeatParamsHalf);
      SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

      WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
      SetVectorMask<half, MaskMode::COUNTER>(NM_FRACTAL_INT8 * processXKBaseN);
      Cast<float, half, false>(ubXFloatND, ubXHalfND, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsH2F);

      if(tilingData->smoothScale) {
        SetVectorMask<half, MaskMode::COUNTER>(realBaseKAligned32);
        Cast<float, half, false>(ubSmoothScaleFloat, ubSmoothScale, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsH2F);
        PipeBarrier<PIPE_V>();
        uint32_t mulOffset = 0;
        for(int32_t k = 0; k < NM_FRACTAL_INT8; k++){
          Mul<float, false>(ubXFloatND[mulOffset], ubXFloatND[mulOffset], ubSmoothScaleFloat, MASK_PLACEHOLDER, 1, binaryParams);
          mulOffset += processXKBaseN;
        }
      }

      PipeBarrier<PIPE_V>();
      if(tilingData->perToken){
        SetVectorMask<half, MaskMode::COUNTER>(realBaseKAligned32);
        uint32_t mulOffset = 0;
        if(fracMIdx_ != (int)fracMIdx) {
          WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS[0]);
        }
        for(int32_t k = 0; k < NM_FRACTAL_INT8; k++){
          if(fracMIdx_ != (int)fracMIdx) {
            if(tilingData->dynamicQuant) {
              pertokenScale[k] = FLOAT_1 / ubXScaleFloat.GetValue(k * FLOAT_PERBLOCK);
            } else {
              pertokenScale[k] = FLOAT_1 / ubXScaleFloat.GetValue(k);
            }
          }
          Muls<float, false>(ubXFloatND[mulOffset], ubXFloatND[mulOffset], pertokenScale[k], MASK_PLACEHOLDER, 1, unaryParams);
          mulOffset += processXKBaseN;
        }
        fracMIdx_ = fracMIdx;
      } else {
        SetVectorMask<half, MaskMode::COUNTER>(NM_FRACTAL_INT8 * processXKBaseN);
        Muls<float, false>(ubXFloatND, ubXFloatND, x_scale_quant, MASK_PLACEHOLDER, 1, unaryParams);
      }

      if(realBaseK != realBaseKAligned32) {
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        uint32_t mulOffset = 0;
        for(int32_t k = 0; k < NM_FRACTAL_INT8; k++){
          Duplicate<float>(ubXFloatND[mulOffset + realBaseKAligned32 - K_FRACTAL_INT8], FLOAT_0, mulAlignMask, 1,
                        DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
          mulOffset += processXKBaseN;
        }
        SetMaskCount();
      }

      SetVectorMask<half, MaskMode::COUNTER>(NM_FRACTAL_INT8 * processXKBaseN);
      PipeBarrier<PIPE_V>();
      Cast<int32_t, float, false>(ubXFloatND.ReinterpretCast<int32_t>(), ubXFloatND, RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1, unaryParamsF2I);
      PipeBarrier<PIPE_V>();
      SetDeqScale((half)1.000000e+00f);
      Cast<half, int32_t, false>(ubXHalfND, ubXFloatND.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsH2I8);

      PipeBarrier<PIPE_V>();
      WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
      transDataParamsX.repeatTimes = realBaseKAligned32 / HALF_PERBLOCK;
      transDataParamsX.dstRepStride = NUMBER_16;
      transDataParamsX.srcRepStride = 1;
      TransDataTo5HD<half>(ZNLocalList, NDLocalList, transDataParamsX);
      SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);

      PipeBarrier<PIPE_V>();
      SetVectorMask<half, MaskMode::COUNTER>(NM_FRACTAL_INT8 * realBaseKAligned32);
      Cast<int8_t, half, false>(ubXInt8ZN, ubXHalfZN, RoundMode::CAST_ROUND, MASK_PLACEHOLDER, 1, unaryParamsH2I8);
      SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);

      WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
      repeatParamsInt8.blockLen = realBaseKAligned32 * NM_FRACTAL_INT8 / INT8_PERBLOCK;
      DataCopy<int8_t>(xZNGM[kOffsetZN + mOffsetAlignedK], ubXInt8ZN, repeatParamsInt8);
      SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    }
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[1]);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    SetMaskNorm();
    ResetMask();
  }

  __aicore__ inline void ProcessMM() {
    if(realSingleCoreM == 0 || realSingleCoreN==0) {
      return;
    }
    DataCopyParams repeatParamsX, repeatParamsW, repeatParamsY, repeatParamsWScale, repeatParamsPertokenScale, repeatParamsYND, repeatParamsNZ2ND;
    LoadData2dParams loadData2DX, loadData2DW;
    DataCopyEnhancedParams enhancedParams;
    MmadParams mmadParams;
    UnaryRepeatParams unaryParams, unaryParamsI2F, unaryParamsF2H;
    BinaryRepeatParams binaryParams, binaryParamsPertoken;
    TransDataTo5HDParams transDataParams;
    unaryParamsF2H.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    loadData2DX.srcStride = 1;
    loadData2DX.ifTranspose = true;
    loadData2DW.srcStride = 1;
    loadData2DW.ifTranspose = false;
    binaryParams.dstBlkStride = NUMBER_2;
    binaryParams.src0BlkStride = NUMBER_2;
    binaryParams.src1BlkStride = 0;
    binaryParams.dstRepStride = NUMBER_16;
    binaryParams.src0RepStride = NUMBER_16;
    binaryParams.src1RepStride = 0;
    SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
    SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    uint32_t baseMNNumI = baseNNum;
    uint32_t baseMNNumJ = baseMNum;
    uint32_t offsetMNI = globalOffsetN;
    uint32_t offsetMNJ = globalOffsetM;
    uint32_t baseMNI = baseN;
    uint32_t baseMNJ = baseM;
    uint32_t baseMNTailI = baseNTail;
    uint32_t baseMNTailJ = baseMTail;
    bool MNK = MCoreNum > NCoreNum;
    if(MNK){
      baseMNNumI = baseMNum;
      baseMNNumJ = baseNNum;
      offsetMNI = globalOffsetM;
      offsetMNJ = globalOffsetN;
      baseMNI = baseM;
      baseMNJ = baseN;
      baseMNTailI = baseMTail;
      baseMNTailJ = baseNTail;
    }

    uint32_t offsetMNI_ = offsetMNI;
    for(uint32_t i=0;i<baseMNNumI;i++){
      uint32_t realBaseMNI = baseMNI - ((i < baseMNTailI) ? 0 : 1);
      if(realBaseMNI == 0)break;

      uint32_t offsetMNJ_ = offsetMNJ;
      for(uint32_t j=0;j<baseMNNumJ;j++){
        uint32_t realBaseMNJ = baseMNJ - ((j < baseMNTailJ) ? 0 : 1);

        if(realBaseMNJ == 0)break;

        uint32_t realBaseM = realBaseMNJ;
        uint32_t realBaseN = realBaseMNI;
        uint32_t offsetM = offsetMNJ_;
        uint32_t offsetN = offsetMNI_;
        if(MNK) {
          realBaseM = realBaseMNI;
          realBaseN = realBaseMNJ;
          offsetM = offsetMNI_;
          offsetN = offsetMNJ_;
        }

        mmadParams.m = realBaseM * NM_FRACTAL_INT8;
        mmadParams.n = realBaseN * NM_FRACTAL_INT8;
        repeatParamsX.blockCount = realBaseM;
        repeatParamsW.blockLen = realBaseN * NM_FRACTAL_INT8 * K_FRACTAL_INT8 / INT8_PERBLOCK;
        repeatParamsW.srcStride = (tilingData->fracN - realBaseN) * NM_FRACTAL_INT8 * K_FRACTAL_INT8 / INT8_PERBLOCK;

        mmadParams.cmatrixInitVal = true;
        uint32_t offsetK = 0;
        for(uint32_t k=0;k<baseKNum;k++){
          uint32_t realBaseK = baseK - ((k < baseKTail) ? 0 : 1);
          if(realBaseK == 0)break;

          mmadParams.k = realBaseK * K_FRACTAL_INT8;
          WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
          repeatParamsX.blockLen = realBaseK * NM_FRACTAL_INT8 * K_FRACTAL_INT8 / INT8_PERBLOCK;
          repeatParamsX.srcStride = (tilingData->fracK - realBaseK) * NM_FRACTAL_INT8 * K_FRACTAL_INT8 / INT8_PERBLOCK;
          DataCopy<int8_t>(l1X, xZNGM[(offsetM * tilingData->fracK + offsetK) * NM_FRACTAL_INT8 * K_FRACTAL_INT8], repeatParamsX);
          SetFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[0]);

          WaitFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[0]);
          loadData2DX.repeatTimes = realBaseM * realBaseK;
          LoadData<int8_t>(l0aX, l1X, loadData2DX);
          SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);

          WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
          repeatParamsW.blockCount = realBaseK;
          DataCopy<int8_t>(l1W, quantizedWeightGm[(offsetK * tilingData->fracN + offsetN) * NM_FRACTAL_INT8 * K_FRACTAL_INT8], repeatParamsW);
          SetFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[0]);

          WaitFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1[0]);
          loadData2DW.repeatTimes = realBaseK * realBaseN;
          LoadData<int8_t>(l0bW, l1W, loadData2DW);
          SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);

          SetFlag<HardEvent::MTE1_M>(eventIdMTE1ToM[0]);
          WaitFlag<HardEvent::MTE1_M>(eventIdMTE1ToM[0]);
          Mmad<int32_t, int8_t, int8_t>(l0cY, l0aX, l0bW, mmadParams);
          SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[0]);
          WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[0]);

          if(k==0) mmadParams.cmatrixInitVal = false;
          offsetK += realBaseK;
        }
        if(tilingData->perToken){
          if(tilingData->dynamicQuant) {
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
            repeatParamsPertokenScale.blockLen = realBaseM * L0C_FRACTAL;
            DataCopy<float>(ubPertokenScaleRaw_, xScaleGm[offsetM * L0C_FRACTAL * FLOAT_PERBLOCK], repeatParamsPertokenScale);
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
            transDataParams.repeatTimes = realBaseM * L0C_FRACTAL / NUMBER_16;
            if(transDataParams.repeatTimes==1){
              transDataParams.dstRepStride = 0;
              transDataParams.srcRepStride = 0;
            } else {
              transDataParams.dstRepStride = NUMBER_2;
              transDataParams.srcRepStride = NUMBER_16;
            }
            TransDataTo5HD<float>(pertokenDynamicLocalList, pertokenDynamicRawLocalList, transDataParams);

            PipeBarrier<PIPE_V>();
            transDataParams.repeatTimes = realBaseM * L0C_FRACTAL / FLOAT_PERBLOCK;
            transDataParams.dstRepStride = NUMBER_16;
            transDataParams.srcRepStride = 1;
            TransDataTo5HD<float>(pertokenLocalList, pertokenRawLocalList, transDataParams);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
          } else {
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
            repeatParamsPertokenScale.blockLen = realBaseM * L0C_FRACTAL / FLOAT_PERBLOCK;
            DataCopy<float>(ubPertokenScaleRaw, xScaleGm[offsetM * L0C_FRACTAL], repeatParamsPertokenScale);
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
            transDataParams.repeatTimes = realBaseM * L0C_FRACTAL / FLOAT_PERBLOCK;
            transDataParams.dstRepStride = NUMBER_16;
            transDataParams.srcRepStride = 1;
            TransDataTo5HD<float>(pertokenLocalList, pertokenRawLocalList, transDataParams);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
          }
        }

        SetFlag<HardEvent::M_V>(eventIdMToV[0]);
        WaitFlag<HardEvent::M_V>(eventIdMToV[0]);
        repeatParamsY.blockLen = realBaseN * realBaseM;
        DataCopy<int32_t>(ubYInt32, l0cY, repeatParamsY, enhancedParams);
        SetFlag<HardEvent::V_M>(eventIdVToM[0]);
        WaitFlag<HardEvent::V_M>(eventIdVToM[0]);

        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(realBaseN * realBaseM * L0C_FRACTAL * L0C_FRACTAL);
        Cast<float, int32_t, false>(ubYFloat, ubYInt32, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsI2F);

        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        repeatParamsWScale.blockLen = realBaseN * L0C_FRACTAL / FLOAT_PERBLOCK;
        DataCopy<float>(ubWScale, wScaleGm[offsetN * L0C_FRACTAL], repeatParamsWScale);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        uint32_t repeatTimes = realBaseM * L0C_FRACTAL / FLOAT_PERBLOCK;
        uint32_t mulOffsetA = 0;
        uint32_t mulOffsetB = 0;
        for(int32_t k=0;k<realBaseN;k++){
          Mul<float, false>(ubYFloat[mulOffsetB], ubYFloat[mulOffsetB], ubWScale[mulOffsetA], MASK_PLACEHOLDER, repeatTimes, binaryParams);
          Mul<float, false>(ubYFloat[mulOffsetB+FLOAT_PERBLOCK], ubYFloat[mulOffsetB+FLOAT_PERBLOCK], ubWScale[mulOffsetA+FLOAT_PERBLOCK], MASK_PLACEHOLDER, repeatTimes, binaryParams);
          mulOffsetA += L0C_FRACTAL;
          mulOffsetB += realBaseM * L0C_FRACTAL * L0C_FRACTAL;
        }
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        SetMaskCount();

        PipeBarrier<PIPE_V>();
        if(tilingData->perToken){
          SetVectorMask<float, MaskMode::COUNTER>(realBaseM * L0C_FRACTAL * L0C_FRACTAL);
          uint32_t mulOffset = 0;
          for(int32_t k=0;k<realBaseN;k++){
            Mul<float, false>(ubYFloat[mulOffset], ubYFloat[mulOffset], ubPertokenScale, MASK_PLACEHOLDER, 1, binaryParamsPertoken);
            mulOffset += realBaseM * L0C_FRACTAL * L0C_FRACTAL;
          }
        } else {
          SetVectorMask<float, MaskMode::COUNTER>(realBaseN * realBaseM * L0C_FRACTAL * L0C_FRACTAL);
          Muls<float, false>(ubYFloat, ubYFloat, x_scale_dequant, MASK_PLACEHOLDER, 1, unaryParams);
        }

        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(realBaseN * realBaseM * L0C_FRACTAL * L0C_FRACTAL);
        Cast<half, float, false>(ubYHalfNZ, ubYFloat, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParamsF2H);

        PipeBarrier<PIPE_V>();
        uint32_t copyOffsetA = 0;
        uint32_t copyOffsetB = 0;
        repeatParamsNZ2ND.blockCount = realBaseM * L0C_FRACTAL;
        repeatParamsNZ2ND.blockLen = 1;
        repeatParamsNZ2ND.dstStride = (realBaseN - 1) * L0C_FRACTAL / HALF_PERBLOCK;
        for(int32_t k=0;k<realBaseN;k++){
          DataCopy<half>(ubYHalfND[copyOffsetA], ubYHalfNZ[copyOffsetB], repeatParamsNZ2ND);
          copyOffsetA += L0C_FRACTAL;
          copyOffsetB += realBaseM * L0C_FRACTAL * L0C_FRACTAL;
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3[0]);
        uint32_t tmp = 0;
        if((offsetM + realBaseM) == fracM) {
          tmp = L0C_FRACTAL - tailM;
        }
        repeatParamsYND.blockCount = realBaseM * L0C_FRACTAL - tmp;
        repeatParamsYND.blockLen = realBaseN * L0C_FRACTAL / HALF_PERBLOCK;
        repeatParamsYND.dstStride = (tilingData->fracN - realBaseN) * L0C_FRACTAL / HALF_PERBLOCK;
        DataCopy<half>(yGm[offsetM * L0C_FRACTAL * tilingData->fracN * L0C_FRACTAL + offsetN * L0C_FRACTAL], ubYHalfND, repeatParamsYND);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
        offsetMNJ_ += realBaseMNJ;
      }
      offsetMNI_ += realBaseMNI;
    }
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
  }

  LocalTensor<half> ubXHalfND;
  LocalTensor<half> ubXHalfZN;
  LocalTensor<float> ubXFloatND;
  LocalTensor<int8_t> ubXInt8ZN;
  LocalTensor<half> ubSmoothScale;
  LocalTensor<float> ubXScaleFloat;
  LocalTensor<float> ubSmoothScaleFloat;

  LocalTensor<int8_t> l1X;
  LocalTensor<int8_t> l1W;
  LocalTensor<int8_t> l0aX;
  LocalTensor<int8_t> l0bW;
  LocalTensor<int32_t> l0cY;

  LocalTensor<int32_t> ubYInt32;
  LocalTensor<float> ubYFloat;
  LocalTensor<half> ubYHalfNZ;
  LocalTensor<half> ubYHalfND;
  LocalTensor<float> ubWScale;
  LocalTensor<float> ubPertokenScale;
  LocalTensor<float> ubPertokenScaleRaw;
  LocalTensor<float> ubPertokenScaleRaw_;

  uint64_t ZNLocalList[NUMBER_16];
  uint64_t NDLocalList[NUMBER_16];
  uint64_t pertokenDynamicLocalList[NUMBER_16];
  uint64_t pertokenDynamicRawLocalList[NUMBER_16];
  uint64_t pertokenLocalList[NUMBER_16];
  uint64_t pertokenRawLocalList[NUMBER_16];

  uint32_t processXKloop;
  uint32_t processXKloopPerfracM;
  uint32_t processXKBaseN;
  uint32_t processXKTailN;

  int32_t MCoreNum;
  int32_t NCoreNum;
  int32_t realSingleCoreM;
  int32_t realSingleCoreN;
  int32_t globalOffsetM;
  int32_t globalOffsetN;
  uint32_t baseMNum;
  uint32_t baseNNum;
  uint32_t baseKNum;
  uint32_t baseM;
  uint32_t baseN;
  uint32_t baseK;
  uint32_t baseMTail;
  uint32_t baseNTail;
  uint32_t baseKTail;
};
}  // namespace AscendC
#endif