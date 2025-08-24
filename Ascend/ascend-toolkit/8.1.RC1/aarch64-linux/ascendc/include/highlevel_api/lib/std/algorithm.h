/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. 2024
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file algorithm.h
 * \brief
 */
#ifndef LIB_STD_ALGORITHM_H
#define LIB_STD_ALGORITHM_H
#include "../../impl/std/algorithm/max.h"
#include "../../impl/std/algorithm/min.h"

namespace AscendC {
namespace Std {
template <typename T, typename U> __aicore__ inline T min(const T src0, const U src1);
template <typename T, typename U> __aicore__ inline T max(const T src0, const U src1);
}
}
#endif  // LIB_STD_ALGORITHM_H