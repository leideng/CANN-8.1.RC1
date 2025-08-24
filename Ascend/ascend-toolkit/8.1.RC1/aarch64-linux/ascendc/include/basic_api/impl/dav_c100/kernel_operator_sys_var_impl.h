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

namespace AscendC {
__aicore__ inline void GetArchVersionImpl(uint32_t& coreVersion)
{
    static_assert((__CCE_AICORE__ == 100), "unsupported GetArchVersion on current device");
}

__aicore__ inline int64_t GetSubBlockNumImpl()
{
    return 1;
}

__aicore__ inline int64_t GetProgramCounterImpl()
{
    static_assert((__CCE_AICORE__ == 100), "unsupported GetProgramCounter on current device");
    return 0;
}

__aicore__ inline int64_t GetSystemCycleImpl()
{
    static_assert((__CCE_AICORE__ == 100), "unsupported GetSystemCycle on current device");
    return 0;
}

__aicore__ inline void SetPcieRDCtrlImpl(bool isSetPcie, uint8_t maxBurstLen)
{
    static_assert((__CCE_AICORE__ == 100), "unsupported SetPcieRDCtrl on current device");
}

__aicore__ inline void SetPcieWRCtrlImpl(bool isSetPcie, uint8_t maxBurstLen)
{
    static_assert((__CCE_AICORE__ == 100), "unsupported SetPcieWRCtrl on current device");
}

__aicore__ inline void TrapImpl()
{
    static_assert((__CCE_AICORE__ == 100), "unsupported Trap on current device");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SYS_VAR_IMPL_H