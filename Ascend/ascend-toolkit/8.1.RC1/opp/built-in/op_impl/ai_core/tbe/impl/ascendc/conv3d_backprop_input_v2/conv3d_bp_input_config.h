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
 * \file conv3d_bp_input_config.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_CONFIG_H
#define CONV3D_BP_INPUT_CONFIG_H

#include "../convolution_3d_backprop/conv3d_bp_config_base.h"

namespace Convolution3DBackprop {

template <class A, class B, class C, class D, const Conv3dConfig& CONV3D_CONFIG = CONV3D_CFG_DEFAULT>
struct Conv3DBpInputCfg : public ConvBpContext<A, B, C, D> {
public:
    __aicore__ inline Conv3DBpInputCfg() {}

    using ContextData = struct _ : public ConvBpContext<A, B, C, D>::ContextData {
        __aicore__ inline _() {}
    };
    constexpr static Conv3dConfig conv3dConfig_ = CONV3D_CONFIG;
};

}  // namespace Convolution3DBackprop
#endif
