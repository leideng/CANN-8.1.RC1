/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef AIR_CXX_INC_FRAMEWORK_COMMON_GE_VISIBILITY_H_
#define AIR_CXX_INC_FRAMEWORK_COMMON_GE_VISIBILITY_H_

#if defined(_MSC_VER)
#define VISIBILITY_EXPORT _declspec(dllexport)
#define VISIBILITY_HIDDEN _declspec(dllimport)
#else
#define VISIBILITY_EXPORT __attribute__((visibility("default")))
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

#endif  // AIR_CXX_INC_FRAMEWORK_COMMON_GE_VISIBILITY_H_
