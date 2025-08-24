/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * ascendc_ops is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 * http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

/*!
 * \file moe_finalize_routing_v2_common.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_V2_COMMON
#define MOE_FINALIZE_ROUTING_V2_COMMON

#include "kernel_operator.h"

namespace MoeFinalizeRoutingV2 {
using namespace AscendC;

constexpr int64_t ONE_BLK_SIZE = 32;
constexpr int64_t ONCE_ALGN_NUM_INT32 = 8;
constexpr int64_t INT32_BYTES = 4;
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t PARALLEL_NUM = 2;
constexpr int64_t INVALID_ROW_INDEX = -1;
constexpr int64_t MODE_VALUE_0 = 0;
constexpr int64_t MODE_VALUE_1 = 1;
constexpr int64_t MODE_VALUE_2 = 2;
constexpr int64_t MODE_VALUE_3 = 3;

__aicore__ inline int64_t PadProcessInt32(int64_t param)
{
    return  ONCE_ALGN_NUM_INT32 - param % ONCE_ALGN_NUM_INT32;
}

__aicore__ inline int64_t Int32AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_INT32 - 1) / ONCE_ALGN_NUM_INT32 * ONCE_ALGN_NUM_INT32;
}
}
#endif  // MOE_FINALIZE_ROUTING_V2_COMMON
