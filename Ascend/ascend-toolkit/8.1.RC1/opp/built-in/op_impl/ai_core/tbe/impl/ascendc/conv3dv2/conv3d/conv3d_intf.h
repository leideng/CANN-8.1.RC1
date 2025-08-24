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
 * \file conv3d_intf.h
 * \brief
 */

#ifndef CONV3D_INTF_H
#define CONV3D_INTF_H

#include "kernel_utils.h"
#include "conv3d_util.h"
#include "conv3d_common_func.h"
#include "../conv_common/conv_common_func.h"
#include "../conv_common/conv_framework_util.h"

namespace conv3d {

template <class Config, template <typename, class, bool> class Impl>
struct Conv3dIntf {
    using Ext = Impl<Conv3dIntf, Config, false>;
    using FmapT = typename Config::SrcAT;
    using WeightT = typename Config::SrcBT;
    using OutputT = typename Config::DstT;
    using BiasT = typename Config::BiasT;
    using L0cT = typename Config::L0cT;
    using ContextType = typename Ext::ContextData;
    using ImplDataType = typename Ext::ImplDataType;
    using ConvParam = typename Config::ConvParam;

public:
    ContextType ctx;
    ImplDataType impl;
    constexpr static bool outputOrder = Ext::outputOrder;
    constexpr static int8_t l0pingpong = Ext::l0pingpong;
    constexpr static int8_t bl1bypass = Ext::bl1bypass;
    constexpr static int8_t groupConvType = Ext::groupConvType;
    constexpr static auto formatType = Config::formatA;

public:
    __aicore__ inline Conv3dIntf()
    {}

    __aicore__ inline void Init(const void *__restrict cubeTiling)
    {
        using local = typename Ext::Init;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, cubeTiling)) {
            local::call(this, cubeTiling);
        }
    }

    __aicore__ inline void SetFmap(const GlobalTensor<FmapT> &fmap)
    {
        using local = typename Ext::SetFmap;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, fmap)) {
            local::call(this, fmap);
        }
    }

    __aicore__ inline void SetWeight(const GlobalTensor<WeightT> &weight)
    {
        using local = typename Ext::SetWeight;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, weight)) {
            local::call(this, weight);
        }
    }

    __aicore__ inline void SetBias(const GlobalTensor<BiasT> &bias)
    {
        using local = typename Ext::SetBias;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, bias)) {
            local::call(this, bias);
        }
    }

    __aicore__ inline void SetOrgFmapShape(uint64_t orgCi, uint64_t orgDi, uint64_t orgHi, uint64_t orgWi)
    {
        using local = typename Ext::SetOrgFmapShape;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, orgCi, orgDi, orgHi, orgWi)) {
            local::call(this, orgCi, orgDi, orgHi, orgWi);
        }
    }

    __aicore__ inline void SetOrgWeightShape(
        uint64_t orgCo, uint64_t orgCi, uint64_t orgKd, uint64_t orgKh, uint64_t orgKw)
    {
        using local = typename Ext::SetOrgWeightShape;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, orgCo, orgCi, orgKd, orgKh, orgKw)) {
            local::call(this, orgCo, orgCi, orgKd, orgKh, orgKw);
        }
    }

    __aicore__ inline void SetOrgOutputShape(uint64_t orgCo, uint64_t orgDo, uint64_t orgHo, uint64_t orgWo)
    {
        using local = typename Ext::SetOrgOutputShape;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, orgCo, orgDo, orgHo, orgWo)) {
            local::call(this, orgCo, orgDo, orgHo, orgWo);
        }
    }

    __aicore__ inline void SetSingleFmapShape(
        uint64_t singleCi, uint64_t singleDi, uint64_t singleHi, uint64_t singleWi)
    {
        using local = typename Ext::SetSingleFmapShape;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, singleCi, singleDi, singleHi, singleWi)) {
            local::call(this, singleCi, singleDi, singleHi, singleWi);
        }
    }

    __aicore__ inline void SetSingleOutputShape(
        uint64_t singleCoreBatch, uint64_t singleCo, uint64_t singleDo, uint64_t singleHo, uint64_t singleWo,
        uint64_t singleGroupOpt)
    {
        using local = typename Ext::SetSingleOutputShape;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, singleCoreBatch, singleCo, singleDo, singleHo, singleWo,
                                     singleGroupOpt)) {
            local::call(this, singleCoreBatch, singleCo, singleDo, singleHo, singleWo, singleGroupOpt);
        }
    }

    __aicore__ inline void SetSingleOutputShape(
        uint64_t singleCoreBatch, uint64_t singleCo, uint64_t singleDo, uint64_t singleCoreM, uint64_t singleGroupOpt)
    {
        using local = typename Ext::SetSingleOutputShape;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, singleCoreBatch, singleCo, singleDo, singleCoreM,
                                     singleGroupOpt)) {
            local::call(this, singleCoreBatch, singleCo, singleDo, singleCoreM, singleGroupOpt);
        }
    }

    __aicore__ inline void SetFmapStartPosition(
        int64_t diStartPos, int64_t hiStartPos, int64_t wiStartPos, int64_t ciStartPos)
    {
        using local = typename Ext::SetFmapStartPosition;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, diStartPos, hiStartPos, wiStartPos, ciStartPos)) {
            local::call(this, diStartPos, hiStartPos, wiStartPos, ciStartPos);
        }
    }

    __aicore__ inline void SetFmapStartPosition(int64_t diStartPos, int64_t mStartPos, int64_t ciStartPos)
    {
        using local = typename Ext::SetFmapStartPosition;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this, diStartPos, mStartPos, ciStartPos)) {
            local::call(this, diStartPos, mStartPos, ciStartPos);
        }
    }

    __aicore__ inline void SetGroupOptInfo(
        uint64_t singleCoreCinTail, uint64_t singleCoreCoutTail, bool isGroupOptDimTail = false)
    {
        using local = typename Ext::SetGroupOptInfo;
        if constexpr (CONV_CHECK_FUN(
                          local, Conv3dFunc, this, singleCoreCinTail, singleCoreCoutTail, isGroupOptDimTail)) {
            local::call(this, singleCoreCinTail, singleCoreCoutTail, isGroupOptDimTail);
        }
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<OutputT> &output, bool enPartialSum = false)
    {
        using local = typename Ext::IterateAll;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, ConvFunc, sync, this, output, enPartialSum)) {
            local::template call<sync>(this, output, enPartialSum);
        }
    }

    __aicore__ inline void End()
    {
        using local = typename Ext::End;
        if constexpr (CONV_CHECK_FUN(local, ConvFunc, this)) {
            local::call(this);
        }
    }

private:
    template <bool sync = true>
    __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        using local = typename Ext::Iterate;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, ConvFunc, sync, this, enPartialSum)) {
            return local::template call<sync>(this, enPartialSum);
        }
        return false;
    }

    template <bool sync = true>
    __aicore__ inline void GetTensorC(const GlobalTensor<OutputT> &output, bool enSequentialWrite = false)
    {
        using local = typename Ext::GetTensorC;
        if constexpr (CONV_CHECK_FUN_TEMPLATE(local, ConvFunc, sync, this, output, enSequentialWrite)) {
            local::template call<sync>(this, output, enSequentialWrite);
        }
    }
};

}  // namespace conv3d

#endif
