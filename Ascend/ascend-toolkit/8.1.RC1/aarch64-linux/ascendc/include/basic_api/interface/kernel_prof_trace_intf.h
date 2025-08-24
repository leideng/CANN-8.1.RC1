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
 * \file kernel_prof_trace_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_KERNEL_PROF_TRACE_INTERFACE_H
#define ASCENDC_MODULE_KERNEL_PROF_TRACE_INTERFACE_H
#include "impl/kernel_prof_trace.h"

namespace AscendC {
#ifdef ASCENDC_TRACE_ON
enum class TraceId : uint32_t {
    KFC_CLIENT_POST_MSG = 0x7001,
    KFC_CLIENT_REV_MSG_GM = 0x7002,
    KFC_CLIENT_REV_MSG_UB = 0x7003,
    KFC_SERVER_RUN = 0x7101,
    KFC_SERVER_REV_MSG = 0x7102,
    KFC_SERVER_PROCESS_MSG = 0x7103,
    MatMul_PROCESS_MSG = 0x8001,
    MatMul_CALC,
    Conv = 0x8101,
    DropOut = 0x8201,
    SoftMax = 0x8301,
    SoftmaxGrad,
    SoftmaxFlash,
    SoftmaxFlashV2,
    LogSoftMax,
    SoftmaxFlashV3,
    LayerNorm = 0x8401,
    LayerNormGrad,
    LayerNormGradBeta,
    Pad = 0x8501,
    UnPad,
    BroadCast = 0x8601,
};

#define TRACE_START(apid)                                          \
    do {                                                           \
        set_lpcnt(PROF_START_EVENT | static_cast<uint32_t>(apid)); \
        ProfMarkEvent();                                           \
    } while (0)

#define TRACE_STOP(apid)                                          \
    do {                                                          \
        set_lpcnt(PROF_STOP_EVENT | static_cast<uint32_t>(apid)); \
        ProfMarkEvent();                                          \
    } while (0)
#else

#define TRACE_START(apid)
#define TRACE_STOP(apid)
#endif
__aicore__ inline void MetricsProfStart();

__aicore__ inline void MetricsProfStop();
} // namespace AscendC
#endif // ASCENDC_MODULE_KERNEL_PROF_TRACE_INTERFACE_H