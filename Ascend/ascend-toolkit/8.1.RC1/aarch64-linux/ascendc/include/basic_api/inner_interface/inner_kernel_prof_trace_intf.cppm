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
 * \file inner_kernel_prof_trace_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_PROF_TRACE_INTERFACE_H
#define ASCENDC_MODULE_INNER_PROF_TRACE_INTERFACE_H
#include "impl/kernel_prof_trace.h"

namespace AscendC {
__aicore__ inline void MetricsProfStart()
{
    ProfStartImpl();
}

__aicore__ inline void MetricsProfStop()
{
    ProfStopImpl();
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_PROF_TRACE_INTERFACE_H