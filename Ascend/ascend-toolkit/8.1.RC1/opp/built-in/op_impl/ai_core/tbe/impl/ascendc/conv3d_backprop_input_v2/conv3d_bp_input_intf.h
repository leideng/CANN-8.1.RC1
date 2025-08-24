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
 * \file conv3d_bp_input_intf.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_INTF_H
#define CONV3D_BP_INPUT_INTF_H

#include "../convolution_3d_backprop/conv3d_bp_func.h"
#include "../convolution_3d_backprop/conv3d_bp_util.h"

namespace Convolution3DBackprop {
template <class Config_, template <typename, class> class Impl>
struct Conv3DBpInputIntf : public ConvBpIntf<Config_, Impl> {
public:
    __aicore__ inline Conv3DBpInputIntf() {}
};

}  // namespace Convolution3DBackprop

#endif
