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
 * \file amp_update_scale.h
 * \brief
 */
#ifndef AMP_UPDATE_SCALE_H
#define AMP_UPDATE_SCALE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
constexpr uint32_t BlockBytes = 32;

class AmpUpdateScale {
    public:
        __aicore__ inline AmpUpdateScale() {
        }

        __aicore__ inline void Init(GM_ADDR current_scale, GM_ADDR growth_tracker, GM_ADDR found_inf, GM_ADDR updated_scale,
                                    GM_ADDR updated_growth_tracker, float growthFactor, float backoffFactor, int32_t growthInterval, TPipe *tmpPipe) {
            pipe_ = tmpPipe;

            growthFactor_ = growthFactor;
            backoffFactor_ = backoffFactor;
            growthInterval_ = growthInterval;

            currentScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_CURRENT_SCALE*>(current_scale), 1);
            growthTrackerGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_GROWTH_TRACKER*>(growth_tracker), 1);
            foundInfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FOUND_INF*>(found_inf), 1);

            updatedScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_CURRENT_SCALE*>(updated_scale), 1);
            updatedGrowthTrackerGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_GROWTH_TRACKER*>(updated_growth_tracker), 1);

            pipe_->InitBuffer(currentScaleBuf_, BlockBytes);
            pipe_->InitBuffer(growthTrackerBuf_, BlockBytes);
            pipe_->InitBuffer(foundInfBuf_, BlockBytes);

            pipe_->InitBuffer(updatedScaleBuf_, BlockBytes);
            pipe_->InitBuffer(updatedGrowthTrackerBuf_, BlockBytes);
        }

        __aicore__ inline void Process() {
            Compute();
        }

    private:
        __aicore__ inline void Compute() {
            LocalTensor<DTYPE_CURRENT_SCALE> currentScaleLocalT = currentScaleBuf_.Get<DTYPE_CURRENT_SCALE>();
            LocalTensor<DTYPE_GROWTH_TRACKER> growthTrackerLocalT = growthTrackerBuf_.Get<DTYPE_GROWTH_TRACKER>();
            LocalTensor<DTYPE_FOUND_INF> foundInfLocalT = foundInfBuf_.Get<DTYPE_FOUND_INF>();

            LocalTensor<DTYPE_CURRENT_SCALE> updatedScaleLocalT = updatedScaleBuf_.Get<DTYPE_CURRENT_SCALE>();
            LocalTensor<DTYPE_GROWTH_TRACKER> updatedGrowthTrackerLocalT = updatedGrowthTrackerBuf_.Get<DTYPE_GROWTH_TRACKER>();

            DataCopyExtParams copyParamsCurrentScale{1, 1 * sizeof(DTYPE_CURRENT_SCALE), 0, 0, 0};
            DataCopyExtParams copyParamsGrowthTracker{1, 1 * sizeof(DTYPE_GROWTH_TRACKER), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_CURRENT_SCALE> padParamsCurrentScale{false, 0, 0, 0};
            DataCopyPadExtParams<DTYPE_GROWTH_TRACKER> padParamsGrowthTracker{false, 0, 0, 0};

            DataCopyPad(currentScaleLocalT, currentScaleGm_, copyParamsCurrentScale, padParamsCurrentScale);
            DataCopyPad(growthTrackerLocalT, growthTrackerGm_, copyParamsGrowthTracker, padParamsGrowthTracker);
            DataCopyPad(foundInfLocalT, foundInfGm_, copyParamsCurrentScale, padParamsCurrentScale);

            currentScale_ = currentScaleLocalT.GetValue(0);
            growthTracker_ = growthTrackerLocalT.GetValue(0);
            foundInf_ = foundInfLocalT.GetValue(0);
            if (foundInf_ >= 1) {
                currentScale_ *= backoffFactor_;
                growthTracker_ = 0;
            } else {
                successful_ = growthTracker_ + 1;
                if (successful_ == growthInterval_) {
                    currentScale_ *= growthFactor_;
                    growthTracker_ = 0;
                } else {
                    growthTracker_ = successful_;
                }
            }
            updatedScaleLocalT.SetValue(0, currentScale_);
            updatedGrowthTrackerLocalT.SetValue(0, growthTracker_);

            DataCopyPad(updatedScaleGm_, updatedScaleLocalT, copyParamsCurrentScale);
            DataCopyPad(updatedGrowthTrackerGm_, updatedGrowthTrackerLocalT, copyParamsGrowthTracker);
        }

        TPipe *pipe_;

        float growthFactor_;
        float backoffFactor_;
        float currentScale_;
        float foundInf_;
        int32_t growthInterval_;
        int32_t growthTracker_;
        int32_t successful_;

        GlobalTensor<DTYPE_CURRENT_SCALE> currentScaleGm_, foundInfGm_, updatedScaleGm_;
        GlobalTensor<DTYPE_GROWTH_TRACKER> growthTrackerGm_, updatedGrowthTrackerGm_;

        TBuf<TPosition::VECCALC> growthTrackerBuf_, foundInfBuf_, currentScaleBuf_;
        TBuf<TPosition::VECCALC> updatedGrowthTrackerBuf_, updatedScaleBuf_;
};
#endif // AMP_UPDATE_SCALE_H