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
 * \file inner_kernel_operator_sys_var_intf.cppm
 * \brief
 */

#ifndef ASCENDC_MODULE_INNER_OPERATOR_SYS_VAR_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_SYS_VAR_INTERFACE_H

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_sys_var_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_sys_var_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_sys_var_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_sys_var_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_sys_var_impl.h"
#endif

namespace AscendC {
__aicore__ inline void GetArchVersion(uint32_t& coreVersion)
{
    GetArchVersionImpl(coreVersion);
}

__aicore__ inline int64_t GetSubBlockNum()
{
    return GetSubBlockNumImpl();
}

__aicore__ inline int64_t GetProgramCounter()
{
    return GetProgramCounterImpl();
}

__aicore__ inline void Trap()
{
    TrapImpl();
}

__aicore__ inline int64_t GetSystemCycle()
{
    return GetSystemCycleImpl();
}
}  // namespace AscendC
#endif  // ASCENDC_MODULE_INNER_OPERATOR_SYS_VAR_INTERFACE_H