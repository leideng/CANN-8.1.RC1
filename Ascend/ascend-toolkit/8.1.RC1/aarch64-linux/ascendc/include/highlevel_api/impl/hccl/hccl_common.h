/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_common.h
 * \brief
 */
#ifndef IMPL_HCCL_HCCL_COMMON_H
#define IMPL_HCCL_HCCL_COMMON_H

namespace AscendC {

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#define ASCENDC_HCCL_API_ASSERT(cond, ret, fmt, ...)        \
    do {                                                    \
        if(!(cond)) {                                       \
            KERNEL_LOG(KERNEL_ERROR, fmt, ##__VA_ARGS__);   \
            ret;                                            \
        }                                                   \
    } while(0)
#elif defined(ASCENDC_DEBUG)
#define ASCENDC_HCCL_API_ASSERT(cond, ret, fmt, ...)        \
    do {                                                    \
        ASCENDC_DEBUG_ASSERT(cond, fmt, ##__VA_ARGS__);     \
        if(!(cond)) {                                       \
            ret;                                            \
        }                                                   \
    } while(0)
#else
#define ASCENDC_HCCL_API_ASSERT(cond, ret, fmt, ...)
#endif

__aicore__ inline void FlushDataCache(GlobalTensor<int64_t> &globalHcclMsgArea, __gm__ void *gmAddr)
{
    AscendC::Barrier();
    globalHcclMsgArea.SetGlobalBuffer((__gm__ int64_t *)gmAddr);
    __asm__("NOP");
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(globalHcclMsgArea);
    dsb(DSB_ALL);
}

__aicore__ inline void FlushDataCache(__gm__ void *gmAddr)
{
    GlobalTensor<int64_t> globalHcclMsgArea;
    FlushDataCache(globalHcclMsgArea, gmAddr);
}

}  // namespace AscendC

#endif  // IMPL_HCCL_HCCL_COMMON_H