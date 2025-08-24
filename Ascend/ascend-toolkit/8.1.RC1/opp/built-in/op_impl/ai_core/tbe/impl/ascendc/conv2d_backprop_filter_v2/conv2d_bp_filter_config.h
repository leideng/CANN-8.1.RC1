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
 * \file conv2d_bp_filter_config.h
 * \brief
 */

#ifndef CONV2D_BP_FILTER_CONFIG_H
#define CONV2D_BP_FILTER_CONFIG_H

#include "../convolution_backprop/conv_bp_config_base.h"

namespace ConvolutionBackprop {

template <class A, class B, class C, class D>
struct Conv2DBpFilterCfg : public ConvBpContext<A, B, C, D>{
public:
    __aicore__ inline Conv2DBpFilterCfg() {}

    using ContextData = struct _ : public ConvBpContext<A, B, C, D>::ContextData {
        __aicore__ inline _() {}
    };
};

}  // namespace ConvolutionBackprop
#endif
