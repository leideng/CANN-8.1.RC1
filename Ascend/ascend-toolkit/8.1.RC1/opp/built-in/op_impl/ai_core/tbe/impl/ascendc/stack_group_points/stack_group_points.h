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
 * \file stack_group_points.h
 * \brief
 */
#ifndef _STACK_GROUP_POINTS_H_
#define _STACK_GROUP_POINTS_H_
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BYTE_SIZE = 16;
template <typename T>
class StackGroupPoints {
 public:
  __aicore__ inline StackGroupPoints() {}
  __aicore__ inline void Init(GM_ADDR features, GM_ADDR features_batch_cnt, GM_ADDR indices, GM_ADDR indices_batch_cnt,
  GM_ADDR y, GM_ADDR workspace, int64_t m, int64_t b, int64_t c, int64_t n, int64_t nsample, int64_t res, int64_t reminder,
  int64_t featuresSize, int64_t indicesSize, int64_t fbcSize, int64_t ibcSize, int64_t outLength,int64_t actCore, int64_t standard) {
    this->m = m;
    this->b = b;
    this->c = c;
    this->n = n;
    this->nsample = nsample;
    this->res = res;
    this->reminder = reminder;
    this->featuresSize = featuresSize;
    this->indicesSize = indicesSize;
    this->fbcSize = fbcSize;
    this->ibcSize = ibcSize;
    this->outLength = outLength;
    this->actCore = actCore;
    this->standard = standard;
    // Set Tiling Value

    featuresGm.SetGlobalBuffer((__gm__ T*)features, this->featuresSize);
    features_batch_cntGm.SetGlobalBuffer((__gm__ DTYPE_FEATURES_BATCH_CNT*)features_batch_cnt, this->fbcSize);
    indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, this->indicesSize);
    indices_batch_cntGm.SetGlobalBuffer((__gm__ DTYPE_INDICES_BATCH_CNT*)indices_batch_cnt, this->ibcSize);
    outputGm.SetGlobalBuffer((__gm__ T*)y, this->outLength);
    // GM Allocate

    pipe.InitBuffer(out,BYTE_SIZE * sizeof(T));
    outputLocal = out.Get<T>();
  }

  __aicore__ inline void Process() {
    int32_t tmp = this-> res;
    if (GetBlockIdx()<this->reminder){
      tmp = tmp + 1;
    }
    for (int32_t i = 0; i < tmp; i++){
      ComputeAndCopyOut(i);
    }
  }

 private:
  __aicore__ inline void ComputeAndCopyOut(int32_t progress) {
      Duplicate<T>(outputLocal,0,BYTE_SIZE);
      for (int32_t index = progress*actCore*BYTE_SIZE + GetBlockIdx()*BYTE_SIZE; 
      index < progress*actCore*BYTE_SIZE + GetBlockIdx()*BYTE_SIZE + BYTE_SIZE; ++index){
        if (index>this->standard){
          continue;
        }
        int sample_idx = index % this->nsample;
        int c_idx = index / this->nsample % this->c;
        int pt_idx = index / this->nsample / this->c;
        
        int pt_cnt = *indices_batch_cntGm.GetPhyAddr(0);
        int bs_idx = 0;
        for (int k = 1; k < this->b; k++){
          if (pt_idx >= pt_cnt){
            bs_idx = k;
            int pt_tmp = *indices_batch_cntGm.GetPhyAddr(k);
            pt_cnt += pt_tmp;
          }
        }
        int features_batch_start_idx = 0;
        int features_batch_end_idx = *features_batch_cntGm.GetPhyAddr(0);
        for (int k=0; k<bs_idx;k++){
          features_batch_start_idx += *features_batch_cntGm.GetPhyAddr(k);
          int fbc_tmp = k + 1;
          int fbc_cnt = *features_batch_cntGm.GetPhyAddr(fbc_tmp);
          features_batch_end_idx = features_batch_start_idx + fbc_cnt;
        }
        int tmp_cin = pt_idx * this->nsample + sample_idx;
        int cin = 0;
        if (tmp_cin < this->m * this->nsample) {
          cin = *indicesGm.GetPhyAddr(tmp_cin);
        }
        int in_idx = cin * this->c + c_idx;
        if (in_idx < features_batch_end_idx * this->c && in_idx < this->n * this->c - features_batch_start_idx *this->c){
          int fs_idx = in_idx + features_batch_start_idx * this->c;
          T result = *featuresGm.GetPhyAddr(fs_idx);
          outputLocal.SetValue(index-progress*actCore*BYTE_SIZE-GetBlockIdx()*BYTE_SIZE,result);
        }
      }
      pipe_barrier(PIPE_ALL);
      if(g_coreType == AIV) {
        DataCopy(outputGm[progress*this->actCore*BYTE_SIZE + GetBlockIdx()*BYTE_SIZE],outputLocal[0],BYTE_SIZE);
        pipe_barrier(PIPE_ALL);
      }
    }
  
 private:
  TPipe pipe;
  TBuf<QuePosition::VECCALC> out;
  GlobalTensor<T> featuresGm;
  GlobalTensor<DTYPE_FEATURES_BATCH_CNT> features_batch_cntGm;
  GlobalTensor<DTYPE_INDICES> indicesGm;
  GlobalTensor<DTYPE_INDICES_BATCH_CNT> indices_batch_cntGm;
  GlobalTensor<T> outputGm;
  LocalTensor<T> outputLocal;
  
  int64_t m;
  int64_t b;
  int64_t c;
  int64_t n;
  int64_t nsample;
  int64_t res;
  int64_t reminder;
  int64_t featuresSize;
  int64_t indicesSize;
  int64_t fbcSize;
  int64_t ibcSize;
  int64_t outLength;
  int64_t actCore;
  int64_t standard;
};

#endif  // _STACK_GROUP_POINTS_H_