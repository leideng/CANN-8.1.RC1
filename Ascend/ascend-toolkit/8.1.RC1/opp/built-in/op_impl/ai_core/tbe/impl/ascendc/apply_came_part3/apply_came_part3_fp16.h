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
 * \file apply_came_part3_fp16.h
 * \brief
 */
#ifndef _ASCENDC_APPLY_CAME_PART3_FP16_H_
#define _ASCENDC_APPLY_CAME_PART3_FP16_H_

#include "apply_came_part3_common.h"

template <typename T>
class ApplyCamePart3FP16 {
 public:
  __aicore__ inline ApplyCamePart3FP16() = default;
  __aicore__ inline void Process();
  __aicore__ inline void Init(CamePart3InOut camePart3InOut, GM_ADDR workspace,
                              const ApplyCamePart3TilingData* __restrict cameTiling);

 protected:
  __aicore__ inline void ParseTilingData(const ApplyCamePart3TilingData* __restrict tilingData);
  __aicore__ inline void InitVars();
  __aicore__ inline void InitBuffers(CamePart3InOut camePart3InOut, GM_ADDR workspace);
  __aicore__ inline void InitInBuffers(CamePart3InOut camePart3InOut);
  __aicore__ inline void InitOutBuffers(CamePart3InOut camePart3InOut, GM_ADDR workspace);
  __aicore__ inline void CalcGMOffset();
  __aicore__ inline void ProcessOneLoop(int64_t mIdx, int64_t nIdx);
  __aicore__ inline void ProcessNormal();

  __aicore__ inline void CalcOneOffset(int64_t mIdx, int64_t nIdx);

  __aicore__ inline void MoveOut(AscendC::LocalTensor<float>& ubLocal, int64_t size);
  __aicore__ inline void CopyOut(AscendC::GlobalTensor<float>& outGlobal, int64_t offset, int64_t size);
  __aicore__ inline void CopyUB2Out(AscendC::GlobalTensor<float>& outGlobal, AscendC::LocalTensor<float>& ubLocal,
                                    int64_t offset, int64_t size);
  __aicore__ inline void CopyUB2Workspace(AscendC::GlobalTensor<float>& outGlobal,
                                          AscendC::LocalTensor<float>& ubLocal,
                                          int64_t offset, int64_t size);

  __aicore__ inline void CalcScalar();
  __aicore__ inline void CopyScalar(AscendC::GlobalTensor<float>& scaleGm, float& scaleValue);
  __aicore__ inline void SetNM();

  __aicore__ inline void CalcSumURC(AscendC::LocalTensor<float>& ubLocal4, AscendC::LocalTensor<float>& ubLocal3);
  __aicore__ inline void CalcSumUR(AscendC::LocalTensor<float>& ubLocal2, AscendC::LocalTensor<float>& ubLocal3,
                                   AscendC::LocalTensor<float>& ubLocal4);
  __aicore__ inline void CalcSumURReduce(AscendC::LocalTensor<float>& ubLocal3, AscendC::LocalTensor<float>& ubLocal4,
                                         uint8_t repStride);
  __aicore__ inline void CalcSumURAddBlock(AscendC::LocalTensor<float>& ubLocal2, int64_t repeatTimes);
  __aicore__ inline void CalcSumUC(AscendC::LocalTensor<float>& ubLocal2, AscendC::LocalTensor<float>& ubLocal4);
  __aicore__ inline void CalcSumUCTailBlock(AscendC::LocalTensor<float>& ubLocal4,
                                            AscendC::LocalTensor<float>& ubLocal3, int64_t rowNum, int64_t calcSize);
  __aicore__ inline void CalcAddEps(AscendC::LocalTensor<float>& ubLocal2);
  __aicore__ inline void CalcOutM(AscendC::LocalTensor<float>& ubLocal2, AscendC::LocalTensor<float>& ubLocal3,
                                  AscendC::LocalTensor<float>& ubLocal4);
  __aicore__ inline void CopyInU(AscendC::LocalTensor<float>& ubLocal, AscendC::GlobalTensor<float>& tensorGm);
  __aicore__ inline void CopyInM(AscendC::LocalTensor<float>& ubLocal, AscendC::GlobalTensor<T>& tensorGm);
  __aicore__ inline void CastIn(AscendC::LocalTensor<float>& ubLocal, AscendC::LocalTensor<T>& ubLocalIn);
  __aicore__ inline int64_t Ceil(int64_t a, int64_t b);
  __aicore__ inline int64_t DivCeil(int64_t a, int64_t b);
  __aicore__ inline void CopyOutM(AscendC::GlobalTensor<T>& outGlobal, AscendC::LocalTensor<float>& ubLocal,
                                  int64_t offset);
  __aicore__ inline void ClearAcculateMatrix();

 public:
  AscendC::TPipe pipe;

 protected:
  // input GlobalTensors
  AscendC::GlobalTensor<float> uGm;
  AscendC::GlobalTensor<T> mInputGm;
  AscendC::GlobalTensor<float> epsGm;
  AscendC::GlobalTensor<float> beta1Gm;
  AscendC::GlobalTensor<float> clipThresholdGm;
  AscendC::GlobalTensor<float> sumSquareUGm;
  AscendC::GlobalTensor<int64_t> globalShapeGm;

  // output
  AscendC::GlobalTensor<T> mOutputGm;
  AscendC::GlobalTensor<float> sumURGm;
  AscendC::GlobalTensor<float> sumUCGm;
  AscendC::GlobalTensor<float> sumURCGm;

  AscendC::GlobalTensor<float> workspaceSumGradRC_;
  AscendC::GlobalTensor<float> workspaceSumGradC_;
  AscendC::GlobalTensor<int32_t> gmDetWorkspace; // workspace gm for deterministic algorithm

  // LocalTensor
  AscendC::LocalTensor<float> ubLocal2, ubLocal3, ubLocal4;
  AscendC::LocalTensor<int32_t> ubDetWorkspace; // workspace ub for deterministic algorithm

  // Queue
  AscendC::TQue<AscendC::QuePosition::VECIN, DEFAULT_QUEUE_BUFFE_SIZE> inQueue;
  AscendC::TQue<AscendC::QuePosition::VECOUT, DEFAULT_QUEUE_BUFFE_SIZE> outQueue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf, detBuf;

  int64_t mOffset;
  int64_t oriMOffset;
  int64_t nOffset;
  int64_t oriNOffset;
  int64_t blockOffset;
  int64_t outMOffset;

  int64_t workspaceRCSize;
  int64_t workspaceRCOffset;
  int64_t workspaceCOffset;
  int64_t cBlockNum;
  int64_t workspaceRCOffsetBase;
  int64_t workspaceCOffsetBase;
  int64_t tilingBaseM;

  float eps;
  float beta1;
  float beta2;
  float clipThreshold;
  float sumSquareU;
  float globalM;
  float globalN;
  float maxValue = 1;

  int64_t mLoop;
  int64_t nLoop;
  int64_t bufSize;
  int64_t oriBaseM;
  int64_t oriBaseN;
  int64_t outCOffset;
  int64_t rowBlockStride;
  int64_t tailBlockStride;

  int64_t usedCoreNum;
  int64_t curN;
  int64_t curM;
  int64_t rNumCalc;
  int64_t cNumCalc;
  int64_t baseN;
  int64_t baseM;
  int64_t rCoreNum;
  int64_t cCoreNum;
  int64_t isGlobalShape;
  int64_t useFirstMoment;

  int64_t maxMLoop{0};
  int64_t maxNLoop{0};
};

#endif
