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
 * \file conv3d_dx_v2_basic_block.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_BASIC_BLOCK_H
#define CONV3D_BACKPROP_INPUT_BASIC_BLOCK_H

#include "conv3d_bp_input.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_type.h"
#include "conv3d_backprop_input_v2.h"

constexpr uint32_t ROW_FIRST = 1;
constexpr uint32_t COL_FIRST = 2;

namespace AscendC {
template <typename filterType, int filterFormat, typename dedyType, int dedyFormat, typename yType, int yFormat,
Convolution3DBackprop::B2Condition b2Condition, bool enableKernelSplit = false>
class Conv3dDxBasicBlockSplitMN : public Conv3dDx<filterType, filterFormat, dedyType, dedyFormat, yType, yFormat,
    b2Condition, enableKernelSplit> {
public:
    __aicore__ inline Conv3dDxBasicBlockSplitMN() {};
    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const Conv3DBackpropInputV2TilingData *tilingData)
    {
        if constexpr (yFormat == FORMAT_NDC1HWC0) {
            if ASCEND_IS_AIV {
                return;
            }
        }
        InitTilingData(tilingData);

        this->filterGm_.SetGlobalBuffer((__gm__ filterType *)filter);
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType *)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType *)y);
        this->dedx_.Init(&(tilingData->conv3DDxTiling));
#if __CCE_AICORE__ == 220
        if constexpr (yFormat == FORMAT_NCDHW) {
            this->InitMixCoreBuffer(workSpace);
        }
#endif
    }

    __aicore__ inline void Process() {
        if constexpr (yFormat == FORMAT_NDC1HWC0) {
            if ASCEND_IS_AIV {
                return;
            }
        }

        if (block_idx >= this->usedCoreNum_) {
            return;
        }
        CalBasicBlock();
        this->dedx_.End();
    }

protected:
    __aicore__ inline void InitTilingData(const Conv3DBackpropInputV2TilingData* tilingData) {
        Conv3dDx<filterType, filterFormat, dedyType, dedyFormat, yType, yFormat,
            b2Condition, enableKernelSplit>::InitTilingData(tilingData);
        this->usedCoreNum_ = tilingData->params.coreNum;
        this->singleShapeM_ = this->singleCoreHi_ * this->wi_;
        this->singleShapeK_ = this->k_;
        this->singleShapeN_ = this->singleCoreCin1_ * this->c0_;
    }

    __aicore__ inline void CalBasicBlock() {
        uint64_t mCnt = DivCeil(this->m_, this->singleShapeM_);
        uint64_t mCoreTail = this->m_ - (mCnt - 1) * this->singleShapeM_;

        int64_t hiLen = static_cast<int32_t>(this->backpropPadDown_ - (this->dilationH_ * (this->hk_ - 1))) * this->wi_;
        mCoreTail = (mCoreTail > 0 && mCoreTail > hiLen) ? mCoreTail - hiLen : mCoreTail;

        uint64_t nCnt = DivCeil(this->n_, this->singleShapeN_);
        uint64_t nCoreTail = this->cin_ - (nCnt - 1) * this->singleShapeN_;

        // 记录基本块的位置
        uint64_t batchDepth = static_cast<uint64_t>(this->batch_) * this->di_;
        uint64_t totalCnt = batchDepth * mCnt * nCnt;
        uint64_t calRound = totalCnt / this->usedCoreNum_;
        uint64_t tailCnt = totalCnt - calRound * this->usedCoreNum_;
        uint64_t basicBlockIdx = 0;

        // 拖尾的部分依次分配到前面的核计算，这些核会多算一轮
        if (block_idx < tailCnt) {
            basicBlockIdx = block_idx * calRound + block_idx;
            ++calRound;
        } else {
            basicBlockIdx = block_idx * calRound + tailCnt;
        }

        uint64_t batchDepthNcnt = batchDepth * nCnt;
        uint64_t currentMCoreUse = this->singleShapeM_;
        uint64_t currentNCoreUse = this->singleShapeN_;
        this->dedx_.SetSingleShape(currentMCoreUse, this->singleShapeK_, currentNCoreUse, 1);
        for (uint64_t j = 0; j < calRound; ++j) {
            this->mCoreIndx_ = basicBlockIdx / batchDepthNcnt;
            uint64_t batchDepthIndex = (basicBlockIdx - this->mCoreIndx_ * batchDepthNcnt) / nCnt;
            this->nCoreIndx_ = basicBlockIdx - this->mCoreIndx_ * batchDepthNcnt - batchDepthIndex * nCnt;

            int64_t batchIdx = batchDepthIndex / this->di_;
            int64_t depthIdx = batchDepthIndex - batchIdx * this->di_;
            ++basicBlockIdx;

           // 不可用totalCnt - 1作为尾块, totalCnt包含batch*din
            uint64_t mCoreUse = (this->mCoreIndx_ == (mCnt - 1)) ? mCoreTail : this->singleShapeM_;
            uint64_t nCoreUse = (this->nCoreIndx_ == (nCnt - 1)) ? nCoreTail : this->singleShapeN_;
            if (mCoreUse != currentMCoreUse || nCoreUse != currentNCoreUse) {
                currentMCoreUse = mCoreUse;
                currentNCoreUse = nCoreUse;
                this->dedx_.SetSingleShape(currentMCoreUse, this->singleShapeK_, currentNCoreUse, 1);
            }

            this->curHoStartIdx_ = static_cast<int32_t>(this->mCoreIndx_ * this->singleCoreHi_) - this->backpropPadUp_;
            this->alignedHoStartIdx_ = this->curHoStartIdx_ < 0 ? 0 : ((this->curHoStartIdx_ + this->strideH_ - 1) / this->strideH_);
            if (this->alignedHoStartIdx_ >= this->ho_) {
                continue;
            }

            this->dedx_.SetStartIdx(depthIdx, this->curHoStartIdx_);
            this->CalcBlockOffset(batchIdx);
            this->dedx_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
            this->dedx_.SetWeight(this->filterGm_[this->offsetB_]);
            this->DedxIterateAll();
        }
    }
};
}

#endif // CONV3D_BACKPROP_INPUT_BASIC_BLOCK_H