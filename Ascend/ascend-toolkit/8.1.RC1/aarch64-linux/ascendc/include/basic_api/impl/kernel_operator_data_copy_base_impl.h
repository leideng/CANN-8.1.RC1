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
 * \file kernel_operator_data_copy_base_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_DATA_COPY_BASE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DATA_COPY_BASE_IMPL_H
#include "kernel_tensor.h"
#include "kernel_process_lock.h"
#include "kernel_struct_data_copy.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_data_copy_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_data_copy_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_data_copy_impl.h"
#include "dav_c220/kernel_operator_set_atomic_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_data_copy_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_data_copy_impl.h"
#endif

namespace AscendC {
#if __CCE_AICORE__ == 220
enum class ReduceType : uint8_t {
    NO_REDUCE,
    REDUCE_ADD,
    REDUCE_MIN,
    REDUCE_MAX,
};

template <typename T, enum ReduceType reduceType = ReduceType::NO_REDUCE>
__aicore__ inline void DataCopyWithReduce(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const uint32_t calCount)
{
    struct DataCopyParams repeatParams;
    repeatParams.blockLen = calCount / AscendCUtils::GetC0Count(sizeof(T));
    DataCopyWithReduce<T, reduceType>(dstGlobal, srcLocal, repeatParams);
}

template <typename T, enum ReduceType reduceType = ReduceType::NO_REDUCE>
__aicore__ inline void DataCopyWithReduce(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const DataCopyParams& repeatParams)
{
    AscendC::SetAtomicNoneImpl();
    if constexpr (reduceType == ReduceType::REDUCE_ADD) {
        AscendC::SetAtomicAddImpl<T>();
    } else if constexpr (reduceType == ReduceType::REDUCE_MIN) {
        AscendC::SetAtomicMinImpl<T>();
    } else if constexpr (reduceType == ReduceType::REDUCE_MAX) {
        AscendC::SetAtomicMaxImpl<T>();
    }
    DataCopy(dstGlobal, srcLocal, repeatParams);
    AscendC::SetAtomicNoneImpl();
}

template <typename T, enum ReduceType reduceType = ReduceType::NO_REDUCE>
__aicore__ inline void DataCopyPadWithReduce(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const DataCopyExtParams& dataCopyExtParams)
{
    AscendC::SetAtomicNoneImpl();
    if constexpr (reduceType == ReduceType::REDUCE_ADD) {
        AscendC::SetAtomicAddImpl<T>();
    } else if constexpr (reduceType == ReduceType::REDUCE_MIN) {
        AscendC::SetAtomicMinImpl<T>();
    } else if constexpr (reduceType == ReduceType::REDUCE_MAX) {
        AscendC::SetAtomicMaxImpl<T>();
    }
    DataCopyPad(dstGlobal, srcLocal, dataCopyExtParams);
    AscendC::SetAtomicNoneImpl();
}
#endif

__aicore__ inline void DataCopyGetOffsetList(
    const SliceInfo sliceInfo[], uint32_t shapeInfo[], const uint32_t dimValue, uint32_t *count, uint32_t *offsetList)
{
    uint32_t sliceSize = 1;
    uint32_t copyCount = 1;
    uint32_t currentCount = 1;
    uint32_t preCopyCount = 0;
    uint32_t iter = 0;
    uint32_t totalSliceCount = 0;

    for (uint32_t i = 0; i < dimValue; i++) {
        if (i == 0) {
            *(offsetList + totalSliceCount) = 0;
            totalSliceCount++;
            continue;
        }
        iter = 0;
        sliceSize = sliceSize * shapeInfo[i - 1];
        currentCount =
            (sliceInfo[i].endIndex - sliceInfo[i].startIndex + 1 + sliceInfo[i].stride) / (1 + sliceInfo[i].stride);
        preCopyCount = copyCount;
        copyCount = copyCount * currentCount;
        for (uint32_t j = preCopyCount; j < copyCount; j += preCopyCount) {
            iter++;
            for (uint32_t k = 0; k < preCopyCount; k++) {
                *(offsetList + totalSliceCount) =
                    (*(offsetList + k)) + (iter * (1 + sliceInfo[i].stride)) * sliceSize;
                totalSliceCount++;
            }
        }
    }
    *count = totalSliceCount;
}

__aicore__ inline uint32_t DataCopyGetPhyStartIndex(
    const SliceInfo sliceInfo[], uint32_t shapeInfo[], const uint32_t dimValue)
{
    uint32_t phyStartIndex = 0;
    uint32_t sliceSize = 1;
    for (uint32_t i = 0; i < dimValue; i++) {
        if (i == 0) {
            phyStartIndex = phyStartIndex + sliceInfo[i].startIndex;
        } else {
            sliceSize = sliceSize * shapeInfo[i - 1];
            phyStartIndex = phyStartIndex + sliceSize * sliceInfo[i].startIndex;
        }
    }
    return phyStartIndex;
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_DATA_COPY_BASE_IMPL_H