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
 * \file conv2d_bp_filter.h
 * \brief
 */

#ifndef CONV2D_BP_FILTER_H
#define CONV2D_BP_FILTER_H

#include "../convolution_backprop/conv_bp_register.h"
#include "conv2d_bp_filter_config.h"
#include "conv2d_bp_filter_impl.h"
#include "conv2d_bp_filter_intf.h"

namespace ConvolutionBackprop {

REGISTER_DW_IMPL(Conv2DBackpropFilter, Conv2DBpFilterCfg, Conv2DBpFilterImpl, Conv2DBpFilterIntf);

}  // namespace ConvolutionBackprop
#endif
