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
 * \file kernel_prof_trace.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_PROF_TRACE_IMPL_H
#define ASCENDC_KERNEL_PROF_TRACE_IMPL_H
#include "kernel_utils.h"

namespace AscendC {
#ifdef ASCENDC_TRACE_ON
constexpr uint32_t PROF_START_EVENT = 0x80000000;
constexpr uint32_t PROF_STOP_EVENT = 0xc0000000;

__aicore__ __inline__ void ProfMarkEvent(void)
{
    if (g_coreType == AIV) {
        __asm__ volatile("NOP_BAR.V");
    } else if (g_coreType == AIC) {
        __asm__ volatile("NOP_BAR.M");
        __asm__ volatile("NOP_BAR.MTE1");
    } else {
        __asm__ volatile("NOP_BAR.V");
        __asm__ volatile("NOP_BAR.M");
        __asm__ volatile("NOP_BAR.MTE1");
    }
    __asm__ volatile("NOP_BAR.MTE2");
    __asm__ volatile("NOP_BAR.MTE3");
}
#endif

__aicore__ inline void ProfStartImpl()
{
#ifndef ASCENDC_CPU_DEBUG
#if __CCE_AICORE__ == 220
    bisheng::cce::metrics_prof_start();
#else
    ASCENDC_DEBUG_ASSERT(false, "MetricsProfStart is not supported on current device\n");
#endif
#endif
}

__aicore__ inline void ProfStopImpl()
{
#ifndef ASCENDC_CPU_DEBUG
#if __CCE_AICORE__ == 220
    bisheng::cce::metrics_prof_stop();
#else
    ASCENDC_DEBUG_ASSERT(false, "MetricsProfStart is not supported on current device\n");
#endif
#endif
}

} // namespace AscendC
#endif // ASCENDC_KERNEL_PROF_TRACE_IMPL_H
