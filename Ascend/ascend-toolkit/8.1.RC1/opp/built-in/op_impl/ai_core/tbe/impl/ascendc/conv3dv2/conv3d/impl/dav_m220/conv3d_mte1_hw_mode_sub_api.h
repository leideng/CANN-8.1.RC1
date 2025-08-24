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
 * \file conv3d_mte1_hw_mode_sub_api.h
 * \brief
 */

#ifndef CONV3D_MTE1_HW_MODE_SUB_API_H
#define CONV3D_MTE1_HW_MODE_SUB_API_H

#include "conv3d_mte2_hw_mode_sub_api.h"

using namespace AscendC;
using namespace conv;
using namespace conv3d;

namespace Conv3dFunc {

template <class Intf>
class LoadAL0WithHwModeTools {
public:
    __aicore__ inline LoadAL0WithHwModeTools() {}

    __aicore__ inline void SetParams(Intf *self, LoadAL1WithHwModeTools<Intf> *aL1Tools)
    {
        self_ = self;
        aL1Tools_ = aL1Tools;
    }

    __aicore__ inline void SetM(uint64_t m)
    {
        currentML0_ = m;
    }

    __aicore__ inline void LoadAL0()
    {
        currentKL0_ = self_->ctx.kIter == self_->ctx.maxKL0Iter ? self_->ctx.kL0Tail : self_->ctx.singleCoreKL0;
        // get if set 2d flag from L1load tools
        if (aL1Tools_->GetSet2dFlag()) {
            SetAl02d();
            return;
        }

        // img2col fmap param info, which set in each ddr->L1
        LoadData3DParamsV2<typename Intf::FmapT> loadData3Dv2Params = aL1Tools_->GetLoadData3DParams();
        SetLoadData3DParamsV2(loadData3Dv2Params);

        LoadData<typename Intf::FmapT, CONV3D_LOAD3DV2_DEFAULT_CONFIG>(
            self_->ctx.al0, self_->ctx.al1, loadData3Dv2Params);
    };

    __aicore__ inline void SetAl02d()
    {
        // only support bf16 now, set pad value to be 0
        al0Set2dSpacesize_ = currentML0_ * currentKL0_ * self_->ctx.sizeOfFmap / BLOCK_SIZE;
        InitConstValueParams<typename Intf::FmapT> initConstValueParams(1, (uint16_t)al0Set2dSpacesize_, 0, 0);
        InitConstValue<typename Intf::FmapT>(self_->ctx.al0, initConstValueParams);
    }

private:
    __aicore__ inline void SetLoadData3DParamsV2(LoadData3DParamsV2<typename Intf::FmapT> &loadData3Dv2Params)
    {
        // params about k dicision
        loadData3Dv2Params.kExtension = currentKL0_;
        loadData3Dv2Params.kStartPt = self_->ctx.kAL0Iter * self_->ctx.singleCoreKL0;
        // params about m dicision
        loadData3Dv2Params.mExtension = currentML0_;
        loadData3Dv2Params.mStartPt = self_->ctx.mAL0Iter * self_->ctx.conv3dTiling->mL0;
        ASC_OP_LOGD(
            "[LoadAL0] loadData3Dv2Params.channelSize %d, loadData3Dv2Params.kExtension %d, "
            "loadData3Dv2Params.kStartPt %d, loadData3Dv2Params.mExtension %d, loadData3Dv2Params.mStartPt %d.\n",
            loadData3Dv2Params.channelSize,
            loadData3Dv2Params.kExtension,
            loadData3Dv2Params.kStartPt,
            loadData3Dv2Params.mExtension,
            loadData3Dv2Params.mStartPt);
    }

private:
    Intf *self_ = nullptr;
    LoadAL1WithHwModeTools<Intf> *aL1Tools_;
    uint64_t currentML0_ = 0;
    uint64_t currentKL0_ = 0;
    uint64_t al0Set2dSpacesize_ = 0;
};

};  // namespace Conv3dFunc

#endif  // __CONV3D_MTE1_HW_MODE_SUB_API_H__
