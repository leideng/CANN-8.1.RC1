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
 * \file kernel_operator_sys_var_impl.h
 * \brief
 */

#ifndef ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H
#include "kernel_common.h"
#include "kernel_utils.h"

namespace AscendC {
__aicore__ inline int64_t GetSubBlockNumImpl()
{
    return 1;
}

__aicore__ inline void GetArchVersionImpl(uint32_t& coreVersion)
{
    const int32_t coreVersionOffset = 32;
    coreVersion = (uint32_t)((uint64_t)(get_arch_ver() >> coreVersionOffset) & 0xFFF);
}

__aicore__ inline int64_t GetProgramCounterImpl()
{
    const int32_t pcOffset = 16;
    int64_t pc = (get_pc() >> pcOffset) & 0xFFFFFFFFFFFF;
    return pc;
}

__aicore__ inline int64_t GetSystemCycleImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "GetSystemCycle");
    return 0;
}

__aicore__ inline void TrapImpl()
{
    trap();
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H