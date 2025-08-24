/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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
 * \file conv3dv2_hw_mode.h
 * \brief
 */

#ifndef CONV3DV2_HW_MODE_H
#define CONV3DV2_HW_MODE_H

#include "conv3dv2.h"

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class CONV_CFG>
class KernelConv3DV2HwMode : public KernelConv3DV2<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONV_CFG> {
public:
    __aicore__ inline KernelConv3DV2HwMode() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const Conv3DTilingData *allTilingData)
    {
        this->InitTilingData(allTilingData);
        this->InitC1N1();
        bool isRealDim = InitSingleCoreData();
        if (!isRealDim) [[unlikely]] {
            this->normalInit = false;
        }
        InitBuffer(x, filter, bias, y);
    }

    __aicore__ inline void Process()
    {
        if (!this->normalInit || this->blockIdx > this->blockDim) [[unlikely]] {
            return;
        }
        Conv3DV2KernelImpl();
    }

protected:

    __aicore__ inline bool InitSingleCoreDataWithHout()
    {
        DataToFill mStruct(this->singleCoreM, this->mIdxStart, this->isMDimTail);
        bool isRealDim = this->CountIdxTail(
            1, this->conv3dRunInfo->mDim, this->conv3dRunInfo->hout, this->conv3dRunInfo->hout, mStruct);
        if (!isRealDim) [[unlikely]] {
            return false;
        }

        return true;
    }

    __aicore__ inline bool InitSingleCoreData()
    {
        if (!this->InitSingleCoreDataWithBatch() || !this->InitSingleCoreDataWithDout() ||
            !this->InitSingleCoreDataWithCout() || !InitSingleCoreDataWithHout()) {
            return false;
        }

        return true;
    }

    __aicore__ inline void InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y)
    {
        this->fmapOneBatchSize =
            this->conv3dRunInfo->din * this->c1In * this->conv3dRunInfo->hin * this->conv3dRunInfo->win * this->c0In;
        this->outputOneBatchSize =
            this->conv3dRunInfo->dout * this->c1Out * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout * this->c0Out;

        int64_t diIdxStart = this->doIdxStart * this->conv3dRunInfo->strideD - this->conv3dRunInfo->padHead;
        int64_t hiIdxStart = this->mIdxStart * this->conv3dRunInfo->strideH - this->conv3dRunInfo->padUp;

        uint64_t fmStartAddr =
            this->batchIdxStart * this->fmapOneBatchSize +
            this->Max(diIdxStart, 0) * this->c1In * this->conv3dRunInfo->hin * this->conv3dRunInfo->win * this->c0In +
            this->Max(hiIdxStart, 0) * this->conv3dRunInfo->win * this->c0In;
        uint64_t weightStartAddr = this->nIdxStart * this->c0K;
        uint64_t outputStartAddr =
            this->batchIdxStart * this->outputOneBatchSize +
            this->doIdxStart * this->c1Out * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout * this->c0Out +
            this->nIdxStart * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout +
            this->mIdxStart * this->conv3dRunInfo->wout * this->c0Out;
        ASC_OP_LOGD("[Conv3DV2HwMode] fmStartAddr %d weightStartAddr %d outputStartAddr %d.\n",
            fmStartAddr,
            weightStartAddr,
            outputStartAddr);

        this->fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(x + fmStartAddr * sizeof(A_T)));
        this->filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(filter + weightStartAddr * sizeof(B_T)));
        this->outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(y + outputStartAddr * sizeof(C_T)));
        if (this->conv3dRunInfo->hasBias) {
            uint64_t biasStartAddr = this->nIdxStart;
            ASC_OP_LOGD("[Conv3DV2HwMode] biasStartAddr %d.\n", biasStartAddr);
            this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_T *>(bias + biasStartAddr * sizeof(BIAS_T)));
        }
    }

    __aicore__ inline void Conv3DV2KernelImpl()
    {
        this->conv.Init(this->conv3dApiTiling);
        if (this->isDoDimTail || this->isNDimTail || this->isMDimTail) [[unlikely]] {
            this->conv.SetSingleOutputShape(
                1, this->singleCoreN, this->singleCoreDout, this->singleCoreM, this->conv3dRunInfo->wout, 0);
        }

        int64_t diIdxStart = this->doIdxStart * this->conv3dRunInfo->strideD;
        int64_t hiIdxStart = this->mIdxStart * this->conv3dRunInfo->strideH;
        ASC_OP_LOGD("[Conv3DV2HwMode] doIdxStart %d diIdxStart %d hiIdxStart %d.\n",
            this->doIdxStart, diIdxStart, hiIdxStart);

        this->conv.SetFmapStartPosition(this->Max(diIdxStart, 0), this->Max(hiIdxStart, 0), 0, 0);
        this->conv.SetWeight(this->filterGm);
        if (this->conv3dRunInfo->hasBias) {
            this->conv.SetBias(this->biasGm);
        }
        for (uint64_t batchIter = 0; batchIter < this->singleCoreBatch; ++batchIter) {
            this->conv.SetFmap(this->fmapGm[batchIter * this->fmapOneBatchSize]);
            this->conv.IterateAll(this->outputGm[batchIter * this->outputOneBatchSize]);
            this->conv.End();
        }
    }

protected:
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;
};

#endif
