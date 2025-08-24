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
 * \file conv3d_bp_input_impl.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_IMPL_H
#define CONV3D_BP_INPUT_IMPL_H

#include "../convolution_3d_backprop/conv3d_bp_func.h"
#include "../convolution_3d_backprop/conv3d_bp_impl_base.h"
#include "../convolution_3d_backprop/conv3d_bp_util.h"
#include "conv3d_bp_input_config.h"
#include "kernel_utils.h"

namespace Convolution3DBackprop {
template <typename Intf_, class Config_>
struct Conv3DBpInputImpl : public ConvBpImpl<Intf_, Config_> {
public:
    __aicore__ inline Conv3DBpInputImpl() {}
    struct ContextData : public ConvBpImpl<Intf_, Config_>::ContextData {
        __aicore__ inline ContextData() {}
    };
};

}  // namespace Convolution3DBackprop

#endif