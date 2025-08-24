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
 * \file conv3d_bp_input.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_H
#define CONV3D_BP_INPUT_H

#include "../convolution_3d_backprop/conv3d_bp_register.h"
#include "conv3d_bp_input_config.h"
#include "conv3d_bp_input_impl.h"
#include "conv3d_bp_input_intf.h"

namespace Convolution3DBackprop {

REGISTER_DX_IMPL(Conv3DBackpropInput, Conv3DBpInputCfg, Conv3DBpInputImpl, Conv3DBpInputIntf);

}  // namespace Convolution3DBackprop
#endif
