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
 * \file conv3d_common_sub_api.h
 * \brief
 */

#ifndef CONV3D_COMMON_SUB_API_H
#define CONV3D_COMMON_SUB_API_H

#if (__CCE_AICORE__ > 300)
    #include "impl/dav_c310/conv3d_sub_api.h"
#elif (__CCE_AICORE__ > 200)
    #include "impl/dav_m220/conv3d_sub_api.h"
    #include "impl/dav_m220/conv3d_pointwise_sub_api.h"
    #include "impl/dav_m220/conv3d_groupopt_sub_api.h"
    #include "impl/dav_m220/conv3d_hw_mode_sub_api.h"
#endif
#endif // __CONV3D_COMMON_SUB_API_H__