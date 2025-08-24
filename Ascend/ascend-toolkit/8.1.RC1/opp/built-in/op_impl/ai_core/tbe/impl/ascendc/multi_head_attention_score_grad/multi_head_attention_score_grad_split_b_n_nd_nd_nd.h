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
 * \file multi_head_attention_score_grad_split_b_n_nd_nd_nd.h
 * \brief
 */
#ifndef _MULTI_HEAD_ATTENTION_SCORE_GRAD_SPLIT_B_N_ND_ND_ND_TRANSPOSE_H_
#define _MULTI_HEAD_ATTENTION_SCORE_GRAD_SPLIT_B_N_ND_ND_ND_TRANSPOSE_H_

#include "multi_head_attention_score_grad_base_kernel.h"
#include "kernel_operator.h"

template <typename T>
class MultiHeadAttentionScoreGradBNNdNdNdTranspose : public MultiHeadAttentionScoreGradBase<T> {
 public:
  __aicore__ inline MultiHeadAttentionScoreGradBNNdNdNdTranspose(){};

  __aicore__ inline void Init(__gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dx, __gm__ uint8_t* query,
                              __gm__ uint8_t* drop_mask, __gm__ uint8_t* atten_mask, __gm__ uint8_t* forward_res,
                              __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dpse,
                              __gm__ uint8_t* workspace,
                              const MultiHeadAttentionScoreGradTilingData* __restrict ordTilingData, TPipe* pipe_in);
  __aicore__ inline void Process();

 protected:
  __aicore__ inline bool isBatchTailCore();
};

// innerMatResSize innerMatInputSize  this->sOut
template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBNNdNdNdTranspose<T>::Init(
    __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dx, __gm__ uint8_t* query, __gm__ uint8_t* drop_mask,
    __gm__ uint8_t* atten_mask, __gm__ uint8_t* forward_res, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv,
    __gm__ uint8_t* dpse, __gm__ uint8_t* workspace,
    const MultiHeadAttentionScoreGradTilingData* __restrict ordTilingData, TPipe* pipe_in) {
  this->keyGm.SetGlobalBuffer((__gm__ T*)key);
  this->valueGm.SetGlobalBuffer((__gm__ T*)value);
  this->dxGm.SetGlobalBuffer((__gm__ T*)dx);
  this->queryGm.SetGlobalBuffer((__gm__ T*)query);
  this->dropMaskGm.SetGlobalBuffer((__gm__ uint8_t*)drop_mask);
  this->attenMaskGm.SetGlobalBuffer((__gm__ T*)atten_mask);
  this->forwardResGm.SetGlobalBuffer((__gm__ T*)forward_res);
  this->dqGm.SetGlobalBuffer((__gm__ T*)dq);
  this->dkGm.SetGlobalBuffer((__gm__ T*)dk);
  this->dvGm.SetGlobalBuffer((__gm__ T*)dv);
  this->dpseGm.SetGlobalBuffer((__gm__ T*)dpse);
  this->workspaceGm.SetGlobalBuffer((__gm__ T*)workspace);
  this->syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace, 100 * 8);
  InitOutput<int32_t>(this->syncGlobal[GetBlockIdx() * 8], 8, 0);  // atomic clean

  this->mBlockIdx = GetBlockIdx();
  this->ordTilingData = ordTilingData;
  this->pipe = pipe_in;

  this->sOut = this->ordTilingData->splitCoreParams.sOut;
  this->scaleValue = this->ordTilingData->baseParams.scaleValue;

  this->softmaxInputShape[0] = this->sOut;
  this->softmaxInputShape[1] = this->ordTilingData->baseParams.Skv;

  this->frontResInnerShape[0] = this->sOut;
  this->frontResInnerShape[1] = this->ordTilingData->baseParams.Skv;

  this->dpseResultOutShape[0] = this->sOut;
  this->dpseResultOutShape[1] = this->ordTilingData->baseParams.Skv;

  this->pipe->InitBuffer(this->vecInQue1, 1, this->ordTilingData->splitCoreParams.innerMatResSizeAlign);
  this->pipe->InitBuffer(this->vecClc1, this->ordTilingData->splitCoreParams.innerMatResSizeAlign);
  this->pipe->InitBuffer(this->vecPseOutQue2, 1, this->ordTilingData->splitCoreParams.innerMatResSizeAlign);
  this->pipe->InitBuffer(this->scm1, 1, this->ordTilingData->splitCoreParams.outMatInputSizeCubeAlign * 2);
}

// batchLoopOffset [B, N, S, D]
// softmaxBatchLoopOffset [B, N, S, S]
// s1OutLoopOffset [B, N, Souter, Sinner, D]
// softmaxS1OutLoopOffset[B, N, Souter, Sinner, S]
template <typename T>
__aicore__ inline void MultiHeadAttentionScoreGradBNNdNdNdTranspose<T>::Process() {
  this->InitGm();                                                                    // atomic clean 0
  this->splitedBatchRange = this->ordTilingData->splitCoreParams.splitedBatchRange;  // batch repeat times
  this->s1OutRange = this->ordTilingData->splitCoreParams.s1OutRange;                // s1_out repeat times
  this->sOut = this->ordTilingData->splitCoreParams.sOut;
  this->sInner = this->ordTilingData->splitCoreParams.sFla;

  uint64_t bIdx = 0;
  uint64_t nIdx = 0;
  uint64_t bnOffset = 0;
  uint64_t skvh = this->ordTilingData->baseParams.Skv * this->ordTilingData->baseParams.H;
  uint64_t sqh = this->ordTilingData->baseParams.Sq * this->ordTilingData->baseParams.H;
  uint64_t sqskv = this->ordTilingData->baseParams.Sq * this->ordTilingData->baseParams.Skv;

  if (isBatchTailCore()) {  // Tail Core Engagement
    this->splitedBatchRange = this->ordTilingData->tailSplitCoreParams.splitedBatchRange;
    this->s1OutRange = this->ordTilingData->tailSplitCoreParams.s1OutRange;
    this->sOut = this->ordTilingData->tailSplitCoreParams.sOut;
    bnOffset = this->ordTilingData->baseParams.formerCoreNum * this->ordTilingData->splitCoreParams.splitedBatchRange +
               (this->mBlockIdx - this->ordTilingData->baseParams.formerCoreNum) * this->splitedBatchRange;
  } else {
    bnOffset = this->mBlockIdx * this->splitedBatchRange;
  }

  uint64_t south = this->sOut * this->ordTilingData->baseParams.H;
  uint64_t soutskv = this->sOut * this->ordTilingData->baseParams.Skv;

  for (int32_t batchIdx = 0; batchIdx < this->splitedBatchRange; batchIdx++) {
    // idx  有三个  batch_idx, n_idx, s1_out_idx

    bIdx = (bnOffset + batchIdx) / this->ordTilingData->baseParams.N;
    nIdx = (bnOffset + batchIdx) % this->ordTilingData->baseParams.N;

    uint64_t batchS1LoopOffset = bIdx * skvh + nIdx * this->ordTilingData->baseParams.D;

    uint64_t batchS0LoopOffset = bIdx * sqh + nIdx * this->ordTilingData->baseParams.D;

    uint64_t batchSoftmaxLoopOffset = bIdx * this->ordTilingData->baseParams.N * sqskv + nIdx * sqskv;

    for (int32_t s0OutIdx = 0; s0OutIdx < this->s1OutRange; s0OutIdx++) {  // s/s1OutRges = s_out
      this->s1OutIdx = s0OutIdx;
      uint64_t s0OutLoopOffset = batchS0LoopOffset + s0OutIdx * south;  // [B, N, S, H] tensor gm_start

      uint64_t softmaxS0OutLoopOffset = batchSoftmaxLoopOffset + s0OutIdx * soutskv;  // [B, N, S, S] tensor gm_start

      this->DetermineLoopParams(this->ordTilingData->splitCoreParams, s0OutIdx);

      this->FrontCompute(batchS1LoopOffset, s0OutLoopOffset, softmaxS0OutLoopOffset);
      this->FrontResCopyOut(softmaxS0OutLoopOffset);
    }
    this->ClcDqk(batchIdx, batchS0LoopOffset, batchS1LoopOffset, batchSoftmaxLoopOffset);
  }
}

template <typename T>
__aicore__ inline bool MultiHeadAttentionScoreGradBNNdNdNdTranspose<T>::isBatchTailCore() {
  return this->mBlockIdx >= this->ordTilingData->baseParams.formerCoreNum;
}

#endif