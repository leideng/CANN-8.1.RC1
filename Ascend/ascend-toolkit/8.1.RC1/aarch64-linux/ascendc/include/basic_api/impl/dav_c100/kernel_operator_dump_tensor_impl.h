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
 * \file kernel_operator_dump_tensor_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_DUMP_TENSOR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DUMP_TENSOR_IMPL_H

#include "kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_operator_common_impl.h"


namespace AscendC {
/* **************************************************************************************************
 * DumpTensorImpl                                             *
 * ************************************************************************************************* */
__aicore__ inline void InitDumpImpl(bool mixFlag, uint32_t gmLen)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "InitDump");
}

template <typename T>
__aicore__ void DumpTensorLocal2GMImpl(const LocalTensor<T>& tensor, uint32_t desc, uint32_t size)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Dump tensor");
}

__aicore__ inline void DumpShapeImpl(const ShapeInfo &shapeInfo)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Dump shape");
}

template <typename T>
__aicore__ void DumpTensorGM2GMImpl(const GlobalTensor<T>& tensor, uint32_t desc, uint32_t size)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Dump tensor");
}

template <class... Args>
__aicore__ inline void PrintfImpl(DumpType printType, __gm__ const char* fmt, Args&&... args)
{
#ifdef ASCENDC_DUMP
    ASCENDC_REPORT_NOT_SUPPORT(false, "Dump scalar");
#endif
}

__aicore__ inline void InitDump(bool mixFlag, uint32_t gmLen)
{
    (void)gmLen;
    (void)mixFlag;
    return;
}
__aicore__ inline void InitDump(bool mixFlag, GM_ADDR dumpStartAddr, uint32_t gmLen)
{
    (void)dumpStartAddr;
    (void)gmLen;
    (void)mixFlag;
    return;
}

__aicore__ inline void DumpTimeStampImpl(uint32_t descId)
{
    return;
}
__aicore__ inline void AscendCTimeStamp(uint32_t descId, uint64_t pcPtr = 0)
{
    return;
}
}
#endif
