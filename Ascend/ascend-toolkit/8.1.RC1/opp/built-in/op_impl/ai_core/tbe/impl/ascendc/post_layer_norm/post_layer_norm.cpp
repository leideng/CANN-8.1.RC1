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
 * \file post_layer_norm.cpp
 * \brief
 */
#include "post_layer_norm_base.h"

template<bool FastComputeMode = true>
class PostLayerNorm : public PostLayerNormBase<half> {
public:
    __aicore__ inline PostLayerNorm() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
                                __gm__ uint8_t *z, PostLayerNormTilingData &tilingData)
    {
        InitBase(x, y, gamma, beta, tilingData);
        zGm.SetGlobalBuffer((__gm__ half *)z + gmOffset_);
        pipe.InitBuffer(zQue, BUFFER_NUM, rowNumPerStep_ * RoundUp(sliceSize_) * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        if constexpr (FastComputeMode) {
            FastCompute();
        } else {
            SliceCompute();
        }
    }

private:
    __aicore__ inline void SliceCompute()
    {
        for (uint32_t rowIdx = 0; rowIdx < rowNumCurrCore_; rowIdx++) {
            float mean = ComputeRowMean(rowIdx);
            float std = ComputeRowStd(rowIdx, mean);
            for (uint64_t sliceIdx = 0; sliceIdx < sliceNum_; sliceIdx++) {
                uint32_t size = 0;
                uint64_t offset = 0;
                uint64_t sliceOffset = 0;
                GetSizeAndOffset(rowIdx, sliceIdx, size, offset, sliceOffset);
                ComputeSliceLayernorm(offset, sliceOffset, size, mean, std);
                Cast16AndCopyOut(yLocalFp32, zGm, zQue, offset, size);
            }
        }
    }

    __aicore__ inline void FastCompute()
    {
        uint32_t StepNum = CeilDiv(rowNumCurrCore_, rowNumPerStep_);
        for (uint64_t i = 0; i < StepNum; ++i) {
            uint32_t rowNum = (i < StepNum - 1) ? rowNumPerStep_ : rowNumLastStep_;
            CopyInMultiRow(i, rowNum * colNum_);
            Compute(rowNum);
            CopyOutMultiRow(i, rowNum * colNum_);
        }
    }

private:
    __aicore__ inline void Compute(int32_t rowNum)
    {
        AscendC::LocalTensor<half> xLocal = xQue.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = yQue.DeQue<half>();
        AscendC::LocalTensor<half> beta = betaQue.DeQue<half>();
        AscendC::LocalTensor<half> gamma = gammaQue.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = zQue.AllocTensor<half>();
        for (int32_t rid = 0; rid < rowNum; ++rid) {
            const AscendC::LocalTensor<half> &zLocalInner = zLocal[rid * colNum_];
            ComputeRes(rid, xLocal, yLocal);
            ComputeRowLayernorm(gamma, beta);
            CastFrom32To16(zLocalInner, yLocalFp32, colNum_);
        }
        xQue.FreeTensor(xLocal);
        yQue.FreeTensor(yLocal);
        betaQue.FreeTensor(beta);
        gammaQue.FreeTensor(gamma);
        zQue.EnQue(zLocal);
    }

    __aicore__ inline void CopyOutMultiRow(uint64_t procId, int32_t size)
    {
        uint64_t offset = procId * rowNumPerStep_ * colNum_;
        CopyOut(zGm, zQue, offset, size);
    }

private:
    AscendC::GlobalTensor<half> zGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> zQue;
};

extern "C" __global__ __aicore__ void post_layer_norm(GM_ADDR x, GM_ADDR res_in, GM_ADDR gamma, GM_ADDR beta,
                                                      GM_ADDR z,GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)) {
        PostLayerNorm<true> op;
        op.Init(x, res_in, gamma, beta, z, tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(1)) {
        PostLayerNorm<false> op;
        op.Init(x, res_in, gamma, beta, z, tiling_data);
        op.Process();
    }
}