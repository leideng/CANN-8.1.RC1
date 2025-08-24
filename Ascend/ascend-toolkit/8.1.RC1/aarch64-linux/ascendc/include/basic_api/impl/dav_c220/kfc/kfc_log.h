/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file kfc_log.h
 * \brief
 */
#ifndef __KFC_LOG_H__
#define __KFC_LOG_H__

#ifdef ASCENDC_CPU_DEBUG
#include "string"
#include <sys/types.h>
#include <unistd.h>

#define MIX_LOG(format, ...)                                                                                     \
    do {                                                                                                         \
        std::string core_type = "";                                                                              \
        std::string block_id = "Block_";                                                                         \
        if (g_coreType == AscendC::AIC_TYPE) {                                                                   \
            core_type = "AIC_";                                                                                  \
        } else if (g_coreType == AscendC::AIV_TYPE) {                                                            \
            core_type = "AIV_";                                                                                  \
        } else {                                                                                                 \
            core_type = "MIX_";                                                                                  \
        }                                                                                                        \
        core_type += std::to_string(sub_block_idx);                                                              \
        block_id += std::to_string(block_idx);                                                                   \
        printf("[%s][%s][%s:%d][%s][%ld] " format "\n", block_id.c_str(), core_type.c_str(), __FILE__, __LINE__, \
            __FUNCTION__, (long)getpid(), ##__VA_ARGS__);                                                        \
    } while (0)

#else

#define MIX_LOG(format)

#endif
#endif