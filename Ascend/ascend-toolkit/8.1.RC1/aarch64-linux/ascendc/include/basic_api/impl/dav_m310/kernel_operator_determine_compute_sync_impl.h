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
 * \file kernel_operator_determine_compute_sync_impl.h
 * \brief
 */
#include "kernel_operator_common_intf.h"
#ifndef ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_IMPL_H

namespace AscendC {
__aicore__ inline void InitDetermineComputeWorkspaceCalc(GlobalTensor<int32_t> &gmWorkspace,
    LocalTensor<int32_t> &ubWorkspace)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported InitDetermineComputeWorkspace on current device"); });
}

__aicore__ inline void WaitPreBlockCalc(const GlobalTensor<int32_t> &gmWorkspace, LocalTensor<int32_t> &ubWorkspace)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported WaitPreBlock on current device"); });
}

__aicore__ inline void NotifyNextBlockCalc(GlobalTensor<int32_t> &gmWorkspace, LocalTensor<int32_t> &ubWorkspace)
{
    ASCENDC_ASSERT(false,
        { KERNEL_LOG(KERNEL_ERROR, "unsupported NotifyNextBlock on current device"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_IMPL_H