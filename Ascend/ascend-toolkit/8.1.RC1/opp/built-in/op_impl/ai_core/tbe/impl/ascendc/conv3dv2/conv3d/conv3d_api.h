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
 * \file conv3d_api.h
 * \brief
 */

#ifndef CONV3D_API_H
#define CONV3D_API_H

#include "conv3d_intf.h"
#include "conv3d_config.h"
#include "conv3d_api_impl.h"

namespace conv3d {

template <class Config, template <typename, class, bool> class Impl = Conv3dApiImpl,
    template <class, template <typename, class, bool> class> class Intf = Conv3dIntf>
struct Conv3dIntfExt : public Intf<Config, Impl> {
    __aicore__ inline Conv3dIntfExt()
    {}
};

#define REGISTER_CONV3D_API(name, Config, Impl, Intf)                                                                \
    template <class INPUT_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE, class CONV_CFG = Conv3dParam> \
    using name =                                                                                                     \
        Conv3dIntfExt<Config<ConvDataType<INPUT_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, CONV_CFG>>, Impl, Intf>

REGISTER_CONV3D_API(Conv3d, Conv3dCfg, Conv3dApiImpl, Conv3dIntf);
}  // namespace conv3d
#endif
