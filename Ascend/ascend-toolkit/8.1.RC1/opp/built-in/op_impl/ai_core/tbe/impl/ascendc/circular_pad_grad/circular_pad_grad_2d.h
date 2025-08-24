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
 * \file circular_pad_grad_2d.h
 * \brief
 */
#ifndef CIRCULAR_PAD_GRAD_2D_H
#define CIRCULAR_PAD_GRAD_2D_H
#include "circular_pad_grad.h"
using namespace AscendC;

template <typename T1, typename T2, bool ISCAST = false>
class CircularPadGrad2D : public CircularPadGrad<T1, T2, ISCAST>
{
public:
    __aicore__ inline CircularPadGrad2D(TPipe* pipe) : CircularPadGrad<T1, T2, ISCAST>(pipe) {};

    __aicore__ inline void Init2D(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, 
                                const CircularPadCommonTilingData& tiling_data)
    {
        this->Init(tiling_data);

        int64_t blockId = GetBlockIdx();
        int64_t startIdx = this->perCoreTaskNum_ * blockId;
        if (blockId < this->tailTaskNum_) {
            this->perCoreTaskNum_ += 1;
            startIdx += blockId;
        } else {
            startIdx += this->tailTaskNum_;
        }

        this->xGM_.SetGlobalBuffer((__gm__ T1*)x + this->inputLen_ * startIdx,
            this->inputLen_ * this->perCoreTaskNum_);
        this->yGM_.SetGlobalBuffer((__gm__ T1*)y + this->outputLen_ * startIdx,
            this->outputLen_ * this->perCoreTaskNum_);
        this->workspaceGM_.SetGlobalBuffer((__gm__ T2*)workspace + this->workspaceLen_ * startIdx,
            this->workspaceLen_ * this->perCoreTaskNum_);
        if (this->left_ < 0 || this->right_ < 0 || this->top_ < 0 || this->bottom_ < 0) {
            this->SetGMtoZero(this->outputLen_ * this->perCoreTaskNum_);
        }
    }

    __aicore__ inline void ProcessSmallShape()
    {
        this->AddTopAndBottomSmallShape();
        this->MTE3ToMTE2Sync();
        this->AddLeftAndRightSmallShape();
        this->MTE3ToMTE2Sync();
        CopyToOutSmallShape();
    }

    __aicore__ inline void ProcessBigShape()
    {
        this->AddTopAndBottomBigShape();
        this->MTE3ToMTE2Sync();
        this->AddLeftAndRightBigShape();
        this->MTE3ToMTE2Sync();
        CopyToOutBigShapeBig();
    }

private:
    /************************************小shape***************************************************/
    __aicore__ inline void CopyToOutSmallShape()
    {
        sDataCopyExtParams params;
        for (int32_t i = 0; i < this->perCoreTaskNum_; i++) {
            this->CalculateOutParms(params);
            this->CopyToOutSmallShapeOnePage(i, i, params);
        }
    }

    /************************************大shape**********************************************/
    __aicore__ inline void CopyToOutBigShapeBig()
    {
        sDataCopyExtParams params;

        for (int64_t i = 0; i < this->perCoreTaskNum_; i++) {
            this->CalculateOutParms(params);
            this->CopyToOutBigShapeOnePage(i, i, params);
        }
    }
};
#endif