/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file init_global_memory.h
 * \brief
 */
#ifndef LIB_UTILS_INIT_GLOBAL_MEMORY_H
#define LIB_UTILS_INIT_GLOBAL_MEMORY_H

#if __CCE_AICORE__ == 200
#include "../../impl/utils/init_global_memory/init_global_memory_v200_impl.h"
#elif __CCE_AICORE__ == 220
#include "../../impl/utils/init_global_memory/init_global_memory_v220_impl.h"
#endif

namespace AscendC {
/* !
 * \brief This function realizes the clear global memory function. 
 *
 * \note support data type: uint16_t, int16_t, half, float, uint32_t, int32_t
 *
 * \param [out] GlobalTensor
 * \param [in] size, size of space to be initialized
 * \param [in] value, value to be initialized in global memory
 */
#if __CCE_AICORE__ == 200
template <typename T>
__aicore__ inline __in_pipe__(V)
    __out_pipe__(MTE3, S) void InitGlobalMemory(GlobalTensor<T> &gmWorkspaceAddr, const uint64_t size, const T value)
{
    InitGlobalMemoryImpl<T>(gmWorkspaceAddr, size, value);
}

#elif __CCE_AICORE__ == 220
template <typename T>
__aicore__ inline __in_pipe__(V)
    __out_pipe__(MTE3) void InitGlobalMemory(GlobalTensor<T> &gmWorkspaceAddr, const uint64_t size, const T value)
{
    InitGlobalMemoryImpl<T>(gmWorkspaceAddr, size, value);
}
#endif
} // namespace AscendC
#endif // LIB_UTILS_INIT_GLOBAL_MEMORY_H
