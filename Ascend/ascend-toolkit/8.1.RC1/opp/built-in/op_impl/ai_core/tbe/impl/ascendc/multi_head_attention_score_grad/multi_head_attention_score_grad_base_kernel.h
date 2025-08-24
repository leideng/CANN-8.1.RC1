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
 * \file multi_head_attention_score_grad_base_kernel.h
 * \brief
 */
#ifndef _MULTI_HEAD_ATTENTION_SCORE_GRAD_BASE_H_
#define _MULTI_HEAD_ATTENTION_SCORE_GRAD_BASE_H_

#include "lib/matmul_intf.h"

using namespace matmul;

template <typename T>
class MultiHeadAttentionScoreGradBase {
 public:
  __aicore__ inline MultiHeadAttentionScoreGradBase(){};

  using aType1 = MatmulType<TPosition::TSCM, CubeFormat::NZ, T, false>;
  using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
  using cType1 = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, T>;
  using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

  using aType2 = MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
  using bType2 = MatmulType<TPosition::TSCM, CubeFormat::NZ, T, false>;
  using cType2 = MatmulType<TPosition::GM, CubeFormat::ND, T>;
  using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

  using aType3 = MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
  using bType3 = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
  using cType3 = MatmulType<TPosition::GM, CubeFormat::ND, T>;
  using biasType3 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

  using aType4 = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
  using bType4 = MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
  using cType4 = MatmulType<TPosition::GM, CubeFormat::ND, T>;
  using biasType4 = MatmulType<TPosition::GM, CubeFormat::ND, float>;

  Matmul<aType1, bType1, cType1, biasType1> mm1;
  Matmul<aType2, bType2, cType2, biasType2> mm2;
  Matmul<aType3, bType3, cType3, biasType3> mm3;
  Matmul<aType4, bType4, cType4, biasType4> mm4;

 protected:
  /* define the que */
  TQue<QuePosition::VECIN, 1> vecInQue1;
  TQue<QuePosition::VECOUT, 1> vecPseOutQue2;
  TBuf<> vecClc1;

  TSCM<TPosition::VECCALC> scm1;

  const MultiHeadAttentionScoreGradTilingData* __restrict ordTilingData;

  TPipe* pipe;

  GlobalTensor<T> keyGm, valueGm, dxGm, queryGm, attenMaskGm, forwardResGm, dqGm, dkGm, dvGm, dpseGm, workspaceGm;
  GlobalTensor<uint8_t> dropMaskGm;
  GlobalTensor<int32_t> syncGlobal;

  uint64_t block_count = 1;
  uint64_t src_stride = 8;
  uint64_t dst_stride = 8;
  uint64_t block_number = 8;
  uint64_t dst_blk_stride = 1;
  uint64_t src0_blk_stride = 1;
  uint64_t src1_blk_stride = 1;
  uint64_t dst_rep_stride = 8;
  uint64_t src0_rep_stride = 8;
  uint64_t src1_rep_stride = 8;

  uint64_t attenMaskOffset = 0;

  uint32_t softmaxInputShape[2];
  uint32_t frontResInnerShape[2];
  uint32_t dpseResultOutShape[2];

  uint32_t softmaxInputOriShape[2];
  uint32_t frontResInnerOriShape[2];
  uint32_t dpseResultOutOriShape[2];

  T mulsValue = 1.0;
  T scaleValue = 0.5;

  uint64_t mBlockIdx;

  int32_t splitedBatchRange;
  int32_t flashKLoopRange;
  int32_t s1OutRange;
  int32_t sInner;
  int32_t sInnerAlign;
  int32_t sOut;
  int32_t sOutCubeAlign;
  int32_t s1OutIdx;

  int32_t innerMatResNum;
  int32_t outMatInputNum;

  __aicore__ inline void InitGm();

  __aicore__ inline void DetermineLoopParams(const SplitCoreParams& coreParams, uint64_t s1OutIdx);

  __aicore__ inline void UpdateTailLoopParams();

  __aicore__ inline void UpdateMatShape();

  __aicore__ inline void FrontCompute(const uint64_t& batchS1LoopOffset, const uint64_t& s0OutLoopOffset,
                                      const uint64_t& softmaxS0OutLoopOffset);

  __aicore__ inline void SetFrontClcShape(LocalTensor<T>& forwardResInner, LocalTensor<T>& sftFrontResInner,
                                          LocalTensor<T>& frontResInner);

  __aicore__ inline void CopyDyInScm(LocalTensor<T>& bmm1Scm, const uint64_t& dxOffset);

  __aicore__ inline void Bmm2Compute(const LocalTensor<T>& scmTensor, const uint64_t& valueOffset);

  __aicore__ inline void ClcDv(const uint64_t& batchS1LoopOffset, const uint64_t& softmaxS0OutLoopOffset,
                               LocalTensor<T>& scmTensor);

  __aicore__ inline void CopyInSoftMaxGrad(LocalTensor<T>& forwardResInner, const GlobalTensor<T>& forwardResGmIn);

  __aicore__ inline void ClcSoftMaxGrad(LocalTensor<T>& sftFrontResInner, LocalTensor<T>& dxInner,
                                        LocalTensor<T>& forwardResInner);

  __aicore__ inline void ClcMulScalar(LocalTensor<T>& frontResInner, T scalar_mul, LocalTensor<T>& sftFrontResInner);

  __aicore__ inline void FrontResCopyOut(const uint64_t& softmaxS0OutLoopOffset);

  __aicore__ inline void MyMatMulClc(const GlobalTensor<T>& tensorA, const GlobalTensor<T>& tensorB,
                                     const GlobalTensor<T>& tensorC, uint64_t isAtomicAddFlag = 0, bool transA = false,
                                     bool transB = false, bool isDq = false);

  __aicore__ inline void ClcDqk(const uint64_t& batchIdx, const uint64_t& batchS0LoopOffset,
                                const uint64_t& batchS1LoopOffset, const uint64_t& batchSoftmaxLoopOffset);
};

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::InitGm() {
  uint64_t allSize =
      this->ordTilingData->baseParams.B * this->ordTilingData->baseParams.Skv * this->ordTilingData->baseParams.H;

  if (mBlockIdx == 0) {
    InitOutput<T>(this->dvGm, allSize, 0);
  }

  LocalTensor<int32_t> workLocal = vecInQue1.AllocTensor<int32_t>();
  SyncAll<false>(syncGlobal, workLocal);
  vecInQue1.FreeTensor(workLocal);
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::DetermineLoopParams(const SplitCoreParams& coreParams,
                                                                               uint64_t s1OutIdx) {
  bool isS1OutTail = (s1OutIdx == (uint64_t)this->s1OutRange - 1);

  if (!isS1OutTail) {
    // sOut parameters
    this->sOut = coreParams.sOut;
    this->sOutCubeAlign = coreParams.sOutCubeAlign;
    this->sInner = coreParams.sFla;
    this->sInnerAlign = coreParams.sFlaAlign;
    this->innerMatResNum = coreParams.innerMatResNumAlign;
    this->outMatInputNum = coreParams.outMatInputNumCubeAlign;
  } else {
    this->sOut = coreParams.sOutTail;
    this->sOutCubeAlign = coreParams.sOutTailCubeAlign;
    this->sInner = coreParams.sFla;
    this->sInnerAlign = coreParams.sFlaAlign;
    this->innerMatResNum = coreParams.innerMatResNumAlign;
    this->outMatInputNum = coreParams.outMatInputNumCubeAlign;
  }

  this->softmaxInputShape[0] = this->sOut;
  this->softmaxInputShape[1] = this->sInnerAlign;
  this->frontResInnerShape[0] = this->sOut;
  this->frontResInnerShape[1] = this->sInnerAlign;
  this->dpseResultOutShape[0] = this->sOut;
  this->dpseResultOutShape[1] = this->sInnerAlign;

  this->softmaxInputOriShape[0] = this->sOut;
  this->softmaxInputOriShape[1] = this->ordTilingData->baseParams.Skv;
  this->frontResInnerOriShape[0] = this->sOut;
  this->frontResInnerOriShape[1] = this->ordTilingData->baseParams.Skv;
  this->dpseResultOutOriShape[0] = this->sOut;
  this->dpseResultOutOriShape[1] = this->ordTilingData->baseParams.Skv;
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::ClcMulScalar(LocalTensor<T>& frontResInner, T scalar_mul,
                                                                        LocalTensor<T>& sftFrontResInner) {
  // todo
  // [m,n] - [m,16] -> [m,n] 按n轴的block数repeat，每个指令repeat算[m,16] - [m,16], subRange循环处理 超过mask情况
  Muls(frontResInner[0], sftFrontResInner[0], scalar_mul, innerMatResNum);
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::ClcSoftMaxGrad(LocalTensor<T>& sftFrontResInner,
                                                                          LocalTensor<T>& dxInner,
                                                                          LocalTensor<T>& forwardResInner) {
  vecInQue1.DeQue<T>();
  // [Performance]:: SoftmaxGrad优化
  SoftmaxGrad<T, true>(sftFrontResInner, dxInner, forwardResInner, this->ordTilingData->softmaxGradTilingData, false);
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::CopyInSoftMaxGrad(LocalTensor<T>& forwardResInner,
                                                                             const GlobalTensor<T>& forwardResGmIn) {
  struct DataCopyParams dataCopyParams;
  struct DataCopyPadParams padParams;

  dataCopyParams.blockCount = this->sOut;                                                 // repeat time
  dataCopyParams.blockLen = (uint16_t)(this->ordTilingData->baseParams.Skv * sizeof(T));  // 单位 Byte  可以不对齐
  dataCopyParams.srcStride = 0;                                                           // 单位block
  dataCopyParams.dstStride = 0;                                                           // 单位 Byte 可以不对其

  padParams.isPad = true;
  padParams.leftPadding = 0;
  padParams.rightPadding = 0;
  padParams.paddingValue = 0;

  DataCopyPad(forwardResInner, forwardResGmIn, dataCopyParams, padParams);

  vecInQue1.EnQue(forwardResInner);
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::ClcDv(const uint64_t& batchS1LoopOffset,
                                                                 const uint64_t& softmaxS0OutLoopOffset,
                                                                 LocalTensor<T>& scmTensor) {
  bool transA = true;
  bool transB = false;
  uint64_t isAtomicAddFlag = 1;
  if (s1OutIdx == 0) {
    isAtomicAddFlag = 0;
  }

  this->mm2.SetTail(sInner, -1, sOut);
  this->mm2.SetTensorA(this->forwardResGm[softmaxS0OutLoopOffset], transA);  // todo
  this->mm2.SetTensorB(scmTensor, transB);

  this->mm2.template IterateAll<true>(dvGm[batchS1LoopOffset], isAtomicAddFlag);
  this->mm2.End();

  this->scm1.FreeTensor(scmTensor);
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::Bmm2Compute(const LocalTensor<T>& scmTensor,
                                                                       const uint64_t& valueOffset) {
  // do first bmm2,
  mm1.SetTensorB(valueGm[valueOffset], true);

  this->mm1.SetTail(sOut, sInner, this->ordTilingData->baseParams.D);  // M N K

  mm1.SetTensorA(scmTensor[0]);
  mm1.template Iterate<true>();  // [sInner, H/N] * [H/N, sInner]
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::CopyDyInScm(LocalTensor<T>& bmm1Scm,
                                                                       const uint64_t& dxOffset) {
  bmm1Scm.SetSize(outMatInputNum * 2);  // check_size
  Nd2NzParams nd2nzParams;

  nd2nzParams.ndNum = 1;
  nd2nzParams.nValue = sOut;                               //
  nd2nzParams.dValue = this->ordTilingData->baseParams.D;  //
  nd2nzParams.srcNdMatrixStride = 0;
  nd2nzParams.srcDValue = this->ordTilingData->baseParams.H;
  nd2nzParams.dstNzC0Stride = sOutCubeAlign;
  nd2nzParams.dstNzNStride = 1;
  DataCopy(bmm1Scm[0], dxGm[dxOffset], nd2nzParams);

  this->scm1.EnQue(bmm1Scm);
  this->scm1.template DeQue<T>();
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::SetFrontClcShape(LocalTensor<T>& softmaxInput,
                                                                            LocalTensor<T>& frontResInner,
                                                                            LocalTensor<T>& dpseResult) {
  softmaxInput.SetShapeInfo(ShapeInfo(2, softmaxInputShape, 2, softmaxInputOriShape, DataFormat::ND));
  frontResInner.SetShapeInfo(ShapeInfo(2, frontResInnerShape, 2, frontResInnerOriShape, DataFormat::ND));
  dpseResult.SetShapeInfo(ShapeInfo(2, dpseResultOutShape, 2, dpseResultOutOriShape, DataFormat::ND));
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::FrontCompute(const uint64_t& batchS1LoopOffset,
                                                                        const uint64_t& s0OutLoopOffset,
                                                                        const uint64_t& softmaxS0OutLoopOffset) {
  LocalTensor<T> softmaxInput = vecInQue1.AllocTensor<T>();  // [s_inner, H/N]
  LocalTensor<T> dpMatmmulResInner = vecClc1.Get<T>(innerMatResNum);

  LocalTensor<T> dpseResult = vecPseOutQue2.AllocTensor<T>();
  softmaxInput.SetSize(innerMatResNum);
  dpseResult.SetSize(innerMatResNum);

  SetFrontClcShape(softmaxInput, dpMatmmulResInner, dpseResult);

  LocalTensor<T> bmm1Scm = scm1.AllocTensor<T>();
  CopyDyInScm(bmm1Scm, s0OutLoopOffset);

  Bmm2Compute(bmm1Scm, batchS1LoopOffset);

  mm1.template GetTensorC<true>(dpMatmmulResInner, false, false);
  mm1.End();

  // temp
  ClcDv(batchS1LoopOffset, softmaxS0OutLoopOffset, bmm1Scm);  // todo

  CopyInSoftMaxGrad(softmaxInput, forwardResGm[softmaxS0OutLoopOffset]);  // todo

  ClcSoftMaxGrad(dpMatmmulResInner, dpMatmmulResInner, softmaxInput);

  ClcMulScalar(dpseResult, scaleValue, dpMatmmulResInner);

  vecInQue1.FreeTensor(softmaxInput);
  vecPseOutQue2.EnQue(dpseResult);
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::FrontResCopyOut(const uint64_t& softmaxS0OutLoopOffset) {
  LocalTensor<T> forwardResInner = this->vecPseOutQue2.template DeQue<T>();
  struct DataCopyParams dataCopyParams;
  // todo
  dataCopyParams.blockCount = this->sOut;                                                 // repeat time
  dataCopyParams.blockLen = (uint16_t)(this->ordTilingData->baseParams.Skv * sizeof(T));  // 单位 Byte  可以不对齐
  dataCopyParams.srcStride = 0;                                                           // 单位block
  dataCopyParams.dstStride = 0;                                                           // 单位 Byte 可以不对其

  DataCopyPad(this->dpseGm[softmaxS0OutLoopOffset], forwardResInner, dataCopyParams);
  this->vecPseOutQue2.FreeTensor(forwardResInner);
}

template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::MyMatMulClc(const GlobalTensor<T>& tensorA,
                                                                       const GlobalTensor<T>& tensorB,
                                                                       const GlobalTensor<T>& tensorC,
                                                                       uint64_t isAtomicAddFlag, bool transA,
                                                                       bool transB, bool isDq) {
  if (isDq) {
    mm4.SetTensorA(tensorA, transA);
    mm4.SetTensorB(tensorB, transB);
    mm4.template IterateAll<false>(tensorC, isAtomicAddFlag);
  } else {
    mm3.SetTensorA(tensorA, transA);
    mm3.SetTensorB(tensorB, transB);
    mm3.template IterateAll<false>(tensorC, isAtomicAddFlag);
  }
}
// batchIdx, batchS0LoopOffset, batchS1LoopOffset, batchSoftmaxLoopOffset
template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBase<T>::ClcDqk(const uint64_t& batchIdx,
                                                                  const uint64_t& batchS0LoopOffset,
                                                                  const uint64_t& batchS1LoopOffset,
                                                                  const uint64_t& batchSoftmaxLoopOffset) {
  // compute dq
  this->MyMatMulClc(this->dpseGm[batchSoftmaxLoopOffset], this->keyGm[batchS1LoopOffset], this->dqGm[batchS0LoopOffset],
                    false, false, false, true);

  // compute dk
  this->MyMatMulClc(this->dpseGm[batchSoftmaxLoopOffset], this->queryGm[batchS0LoopOffset],
                    this->dkGm[batchS1LoopOffset], false, true, false, false);
}

#endif