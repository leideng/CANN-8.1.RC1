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
 * \file conv3d_common_set_func.h
 * \brief
 */

#ifndef CONV3D_COMMON_SET_FUNC_H
#define CONV3D_COMMON_SET_FUNC_H

#include "../conv_common/conv_framework_util.h"
#include "conv3d_common_sub_api.h"
#include "conv3d_config.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

namespace Conv3dFunc {

template <class Intf, uint32_t ImplType>
struct SetOrgFmapShape {
    static __aicore__ inline void call(Intf *self, uint64_t orgCi, uint64_t orgDi, uint64_t orgHi, uint64_t orgWi)
    {
        self->ctx.oriCi = orgCi;
        self->ctx.orgDi = orgDi;
        self->ctx.orgHi = orgHi;
        self->ctx.orgWi = orgWi;
    }
};

template <class Intf, uint32_t ImplType>
struct SetOrgWeightShape {
    static __aicore__ inline void call(
        Intf *self, uint64_t orgCo, uint64_t orgCi, uint64_t orgKd, uint64_t orgKh, uint64_t orgKw)
    {
        self->ctx.orgCo = orgCo;
        self->ctx.orgCi = orgCi;
        self->ctx.kernelD = orgKd;
        self->ctx.kernelH = orgKh;
        self->ctx.kernelW = orgKw;
    }
};

template <class Intf, uint32_t ImplType>
struct SetOrgOutputShape {
    static __aicore__ inline void call(Intf *self, uint64_t orgCo, uint64_t orgDo, uint64_t orgHo, uint64_t orgWo)
    {
        self->ctx.orgCo = orgCo;
        self->ctx.orgDo = orgDo;
        self->ctx.orgHo = orgHo;
        self->ctx.orgWo = orgWo;
    }
};

template <class Intf, uint32_t ImplType>
struct SetSingleFmapShape {
    static __aicore__ inline void call(
        Intf *self, uint64_t singleCi, uint64_t singleDi, uint64_t singleHi, uint64_t singleWi)
    {
        self->ctx.singleCoreCin = singleCi;
        InitKDirectionBaseValue<Intf>(self);
    }
};

template <class Intf, uint32_t ImplType>
struct SetSingleOutputShape {
    static __aicore__ inline void call(
        Intf *self, uint64_t singleCoreBatch, uint64_t singleCo, uint64_t singleDo, uint64_t singleHo,
        uint64_t singleWo, uint64_t singleGroupOpt)
    {
        self->ctx.singleCoreCo = singleCo;
        self->ctx.singleCoreDo = singleDo;
        self->ctx.singleCoreHo = singleHo;
        self->ctx.singleCoreGroupOpt = singleGroupOpt;
        InitHoutDirectionValue<Intf>(self);
        InitCoutDirectionBaseValue<Intf>(self);
        InitDoutDirectionBaseValue<Intf>(self);
        InitGroupOptDirectionValue<Intf>(self);
    }

    static __aicore__ inline void call(
        Intf *self, uint64_t singleCoreBatch, uint64_t singleCo, uint64_t singleDo, uint64_t singleCoreM,
        uint64_t singleGroupOpt)
    {
        self->ctx.singleCoreCo = singleCo;
        self->ctx.singleCoreDo = singleDo;
        self->ctx.singleCoreM = singleCoreM;
        self->ctx.singleCoreGroupOpt = singleGroupOpt;
        InitMDirectionBaseValue<Intf>(self);
        InitCoutDirectionBaseValue<Intf>(self);
        InitDoutDirectionBaseValue<Intf>(self);
        InitGroupOptDirectionValue<Intf>(self);
    }
};

template <class Intf, uint32_t ImplType>
struct SetFmapStartPosition {
    static __aicore__ inline void call(
        Intf *self, int64_t diStartPos, int64_t hiStartPos, int64_t wiStartPos, int64_t ciStartPos)
    {
        self->ctx.diStartPos = diStartPos;
        self->ctx.hiStartPos = hiStartPos;
    }

    static __aicore__ inline void call(Intf *self, int64_t diStartPos, int64_t mStartPos, int64_t ciStartPos)
    {
        self->ctx.diStartPos = diStartPos;
        self->ctx.mStartPos = mStartPos;
    }
};

template <class Intf, uint32_t ImplType>
struct SetGroupOptInfo {
    static __aicore__ inline void call(
        Intf *self, uint64_t singleCoreCinTail, uint64_t singleCoreCoutTail, bool isGroupOptDimTail = false)
    {
        self->ctx.singleCoreCinTail = singleCoreCinTail;
        self->ctx.singleCoreCoutTail = singleCoreCoutTail;
        self->ctx.isGroupOptDimTail = isGroupOptDimTail;
    }
};

}  // namespace Conv3dFunc

#endif
