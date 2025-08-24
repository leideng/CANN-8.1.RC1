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
 * \file inner_kernel_operator_dump_tensor_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_DUMP_TENSOR_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_DUMP_TENSOR_INTERFACE_H
#include "kernel_tensor.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_dump_tensor_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_dump_tensor_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_dump_tensor_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_dump_tensor_impl.h"
#endif

#ifdef ASCENDC_CPU_DEBUG
#include <cstdio>
#include <utility>
#endif

namespace AscendC {
template <typename T>
__aicore__ inline void DumpTensor(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)
{
    ASCENDC_ASSERT((false), {KERNEL_LOG(KERNEL_ERROR, "DumpTensor is not supported in cpu mode.");});
#ifdef ASCENDC_DUMP
    DumpTensorLocal2GMImpl(tensor, desc, dumpSize);
#endif
    return;
}
template <typename T>
__aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize)
{
    ASCENDC_ASSERT((false), {KERNEL_LOG(KERNEL_ERROR, "DumpTensor is not supported in cpu mode.");});
#ifdef ASCENDC_DUMP
    DumpTensorGM2GMImpl(tensor, desc, dumpSize);
#endif
    return;
}
template <typename T>
__aicore__ inline void DumpTensor(const GlobalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize, const ShapeInfo& shapeInfo)
{
    ASCENDC_ASSERT((false), {KERNEL_LOG(KERNEL_ERROR, "DumpTensor is not supported in cpu mode.");});
#ifdef ASCENDC_DUMP
    DumpShapeImpl(shapeInfo);
    DumpTensorGM2GMImpl(tensor, desc, dumpSize);
#endif
    return;
}
template <typename T>
__aicore__ inline void DumpTensor(const LocalTensor<T>& tensor, uint32_t desc, uint32_t dumpSize, const ShapeInfo& shapeInfo)
{
    ASCENDC_ASSERT((false), {KERNEL_LOG(KERNEL_ERROR, "DumpTensor is not supported in cpu mode.");});
#ifdef ASCENDC_DUMP
    DumpShapeImpl(shapeInfo);
    DumpTensorLocal2GMImpl(tensor, desc, dumpSize);
#endif
    return;
}

template <typename T>
__aicore__ inline void DumpAccChkPoint(const LocalTensor<T> &tensor, uint32_t index, uint32_t countOff, uint32_t dumpSize)
{
    ASCENDC_ASSERT((false), {KERNEL_LOG(KERNEL_ERROR, "DumpAccChkPoint is not supported in cpu mode.");});
#if defined(ASCENDC_DUMP) || defined(ASCENDC_ACC_DUMP)
    if (countOff > tensor.GetSize()) {
        ASCENDC_ASSERT((false),
            { KERNEL_LOG(KERNEL_ERROR, "tensor offset [%d] exceeds limit [%d]",
                        countOff, tensor.GetSize()); });
        return;
    }
    LocalTensor<T> tmpTensor = tensor[countOff];
    DumpTensorLocal2GMImpl(tmpTensor, index, dumpSize);
#endif
    return;
}
template <typename T>
__aicore__ inline void DumpAccChkPoint(const GlobalTensor<T> &tensor, uint32_t index, uint32_t countOff, uint32_t dumpSize)
{
    ASCENDC_ASSERT((false), {KERNEL_LOG(KERNEL_ERROR, "DumpAccChkPoint is not supported in cpu mode.");});
#if defined(ASCENDC_DUMP) || defined(ASCENDC_ACC_DUMP)
    if (countOff > tensor.GetSize()) {
        ASCENDC_ASSERT((false),
            { KERNEL_LOG(KERNEL_ERROR, "tensor offset [%d] exceeds limit [%d]",
                        countOff, tensor.GetSize()); });
        return;
    }
    GlobalTensor<T> tmpTensor = tensor[countOff];
    DumpTensorGM2GMImpl(tmpTensor, index, dumpSize);
#endif
    return;
}

#ifdef ASCENDC_CPU_DEBUG
using ::printf;

template<typename... Args>
inline auto PRINTF(Args&&... args) -> decltype(printf(std::forward<Args>(args)...))
{
#ifdef ASCENDC_DUMP
    return printf(std::forward<Args>(args)...);
#else
    return 0;
#endif
}
#else
template <class... Args>
__aicore__ inline void PRINTF(__gm__ const char* fmt, Args&&... args)
{
#ifdef ASCENDC_DUMP
    PrintfImpl(DumpType::DUMP_SCALAR, fmt, args...);
#endif
}

template <class... Args>
__aicore__ inline void printf(__gm__ const char* fmt, Args&&... args)
{
#ifdef ASCENDC_DUMP
    PrintfImpl(DumpType::DUMP_SCALAR, fmt, args...);
#endif
}
#endif

template <class... Args>
__aicore__ inline void AssertImpl(__gm__ const char* fmt, Args&&... args)
{
#ifdef ASCENDC_DUMP
    PrintfImpl(DumpType::DUMP_ASSERT, fmt, args...);
#else
    return;
#endif
}

// for auto open ASCENDC_DUMP macros
#ifdef __CHECK_FEATURE_AT_PRECOMPILE
#define DumpTensor(...)            \
    do {                           \
        ENABLE_PRINTF();           \
        ENABLE_PRINTF_DUMP_SIZE(); \
    } while (0)

#define DumpAccChkPoint(...)       \
    do {                           \
        ENABLE_PRINTF();           \
        ENABLE_PRINTF_DUMP_SIZE(); \
    } while (0)

#define printf(...)                \
    do {                           \
        ENABLE_PRINTF();           \
        ENABLE_PRINTF_DUMP_SIZE(); \
    } while (0)

#define PRINTF(...)                \
    do {                           \
        ENABLE_PRINTF();           \
        ENABLE_PRINTF_DUMP_SIZE(); \
    } while (0)
#endif


__aicore__ inline void PrintTimeStamp(uint32_t descId)
{
#ifdef ASCENDC_TIME_STAMP_ON  // 打点开关宏
    DumpTimeStampImpl(descId);
#endif
}


}  // namespace AscendC
#endif  // END OF ASCENDC_MODULE_INNER_OPERATOR_DUMP_TENSOR_INTERFACE_H

