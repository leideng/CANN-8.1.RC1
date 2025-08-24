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
 * \file conv3d_config.h
 * \brief
 */

#ifndef CONV3D_CONFIG_H
#define CONV3D_CONFIG_H

#include "../conv_common/conv_framework_util.h"
#include "../conv_common/conv_config.h"

using namespace conv;

namespace conv3d {

enum class ConvL0PingPong {
    ALL_CLOSE = 0,
    L0A_OPEN,
    L0B_OPEN,
    ALL_OPEN
};

enum class ConvBL1ByPass {
    BYPASS_OFF = 0,
    BYPASS_ON = 1
};

enum class GroupConvType {
    NoGroup_Conv = 0, // 非group卷积
	GroupConv_Weight_Gfz // group卷积，weight数据为私有group_fractalz格式
};

enum class OutputOrder {
    M_Mode = 0,
    HW_Mode
};

struct Conv3dParam : public ConvParam {
    __aicore__ inline Conv3dParam(){};
};

template <class ConvDataType>
struct Conv3dCfg : public ConvConfig<ConvDataType> {
public:
    __aicore__ inline Conv3dCfg()
    {}

    using ContextData = struct _ : public ConvConfig<ConvDataType>::ContextData {
        __aicore__ inline _()
        {}
    };
};
}  // namespace conv3d

#endif