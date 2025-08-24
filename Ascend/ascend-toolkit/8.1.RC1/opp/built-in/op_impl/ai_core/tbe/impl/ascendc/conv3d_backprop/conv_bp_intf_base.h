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
 * \file conv_bp_intf_base.h
 * \brief
 */

#ifndef CONV_BP_INTF_H
#define CONV_BP_INTF_H

#include "conv_bp_func.h"
#include "conv_bp_util.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ConvolutionBackprop {
// 用户可见的api原型集合
template <class Config_, template <typename, class> class Impl>
struct ConvBpIntf {
    using Config = Config_;
    using Ext = Impl<ConvBpIntf, Config>;
    using SrcT = typename Config::SrcT;
    using DstT = typename Config::DstT;
    using L0cT = typename Config::L0cT;
    using ContextData = typename Ext::ContextData;

public:
    ContextData ctx;

public:
    __aicore__ inline ConvBpIntf()
    {}

    __aicore__ inline void Init(const TConv3DDwTiling *__restrict tiling)
    {
        using Local = typename Ext::Init;
        // CheckFun检查impl是否实现了Init的call函数
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, tiling)) {
            Local::call(this, tiling);
        }
    }

    __aicore__ inline void SetFmap(const GlobalTensor<SrcT> &fmap)
    {
        using Local = typename Ext::SetFmap;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, fmap)) {
            Local::call(this, fmap);
        }
    }

    __aicore__ inline void SetOutBackprop(const GlobalTensor<SrcT> &outBackprop)
    {
        using Local = typename Ext::SetOutBackprop;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, outBackprop)) {
            Local::call(this, outBackprop);
        }
    }

    __aicore__ inline void SetSingleShape(uint64_t singleCoreM, uint64_t singleCoreN, uint64_t singleCoreK,
                                          uint32_t hoStartIdx)
    {
        using Local = typename Ext::SetSingleShape;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, singleCoreM, singleCoreN, singleCoreK,
                                                                      hoStartIdx)) {
            Local::call(this, singleCoreM, singleCoreN, singleCoreK, hoStartIdx);
        }
    }

    template <bool sync = true>
    __aicore__ inline bool Iterate(bool enPartialSum = false)
    {
        using Local = typename Ext::template Iterate<sync>;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, enPartialSum)) {
            return Local::call(this, enPartialSum);
        }
    }

    template <bool sync = true>
    __aicore__ inline void IterateAll(const GlobalTensor<DstT> &output, uint8_t enAtomic = 0)
    {
        using Local = typename Ext::template IterateAll<sync>;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, output, enAtomic)) {
            Local::call(this, output, enAtomic);
        }
    }

    template <bool sync = true>
    __aicore__ inline void GetTensorC(
        const GlobalTensor<DstT> &output, uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        using Local = typename Ext::template GetTensorC<sync>;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this, output, enAtomic, enSequentialWrite)) {
            Local::call(this, output, enAtomic, enSequentialWrite);
        }
    }

    __aicore__ inline void End()
    {
        using Local = typename Ext::End;
        if constexpr (CHECK_FUN(Local, ConvolutionBackpropFunc, this)) {
            Local::call(this);
        }
    }
};

}  // namespace ConvolutionBackprop

#endif
