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
 * \file conv_bp_register.h
 * \brief
 */

#ifndef CONV_BP_REGISTER_H
#define CONV_BP_REGISTER_H

#include "conv_bp_config_base.h"
#include "conv_bp_impl_base.h"
#include "conv_bp_intf_base.h"

namespace ConvolutionBackprop {
// 注册，通过别名定义用户接口
#define REGISTER_DW_IMPL(name, context, impl, intf)             \
    template <class X_T, class W_TYPE, class DEDY_T, class Y_T> \
    using name = intf<context<X_T, W_TYPE, DEDY_T, Y_T>, impl>
}  // namespace ConvolutionBackprop
#endif
