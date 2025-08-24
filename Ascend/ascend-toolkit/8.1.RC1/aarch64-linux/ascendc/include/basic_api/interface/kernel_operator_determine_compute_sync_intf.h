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
 * \file kernel_operator_determine_compute_sync_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_INTF_H
#define ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_INTF_H

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_determine_compute_sync_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_determine_compute_sync_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_determine_compute_sync_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_determine_compute_sync_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_determine_compute_sync_impl.h"
#endif

namespace AscendC {
/*
 * @ingroup WaitPreBlock
 * @brief wait previous compute finish
 * @param [in] global memory workspace
 */
__aicore__ inline void InitDetermineComputeWorkspace(GlobalTensor<int32_t>& gmWorkspace,
    LocalTensor<int32_t>& ubWorkspace);
/*
 * @ingroup WaitPreBlock
 * @brief wait previous compute finish
 * @param [in] global memory workspace
 */
__aicore__ inline void WaitPreBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace);

/*
 * @ingroup NotifyNextBlock
 * @brief wait previous compute finish
 * @param [in] global memory workspace
 */
__aicore__ inline void NotifyNextBlock(GlobalTensor<int32_t>& gmWorkspace, LocalTensor<int32_t>& ubWorkspace);
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DETERMINE_COMPUTE_SYNC_INTF_H