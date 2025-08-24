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
 * \file conv3dv2_pointwise.h
 * \brief
 */

#ifndef CONV3DV2_POINTWISE_H
#define CONV3DV2_POINTWISE_H

#include "conv3dv2.h"

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class CONV_CFG>
class KernelConv3DV2PointWise : public KernelConv3DV2<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONV_CFG> {
public:
    __aicore__ inline KernelConv3DV2PointWise()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y, const Conv3DTilingData *allTilingData)
    {
        this->InitTilingData(allTilingData);
        this->InitC1N1();
        bool isRealDim = this->InitSingleCoreData();
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
        this->Conv3DV2KernelImpl();
    }

protected:
    __aicore__ inline void InitBuffer(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y)
    {
        this->fmapOneBatchSize = this->conv3dRunInfo->cin * this->conv3dRunInfo->din *
                                this->conv3dRunInfo->hin * this->conv3dRunInfo->win;
        this->outputOneBatchSize = this->conv3dRunInfo->cout * this->conv3dRunInfo->dout *
                                    this->conv3dRunInfo->hout * this->conv3dRunInfo->wout;

        uint64_t fmStartAddr = this->batchIdxStart * this->fmapOneBatchSize +
                                this->doIdxStart * this->conv3dRunInfo->hin * this->conv3dRunInfo->win +
                                this->mIdxStart;
        uint64_t weightStartAddr = this->nIdxStart * this->conv3dRunInfo->cin;
        uint64_t outputStartAddr = this->batchIdxStart * this->outputOneBatchSize + this->nIdxStart *
                                this->conv3dRunInfo->dout * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout +
                                this->doIdxStart * this->conv3dRunInfo->hout * this->conv3dRunInfo->wout +
                                this->mIdxStart;
        ASC_OP_LOGD("[Conv3DV2PointWise] fmStartAddr %d weightStartAddr %d outputStartAddr %d.\n",
            fmStartAddr,
            weightStartAddr,
            outputStartAddr);

        this->fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(x + fmStartAddr * sizeof(A_T)));
        this->filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(filter + weightStartAddr * sizeof(B_T)));
        this->outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(y + outputStartAddr * sizeof(C_T)));
        if (this->conv3dRunInfo->hasBias) {
            ASC_OP_LOGD("[Conv3DV2PointWise] biasStartAddr %d.\n", this->nIdxStart);
            this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_T *>(bias + this->nIdxStart * sizeof(BIAS_T)));
        }
    }

protected:
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;
};

#endif // __CONV3DV2_POINTWISE_H__