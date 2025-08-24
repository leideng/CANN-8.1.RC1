/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file group_norm_silu_base.h
 * \brief
 */

#ifndef GROUP_NORM_SILU_BASE_H_
#define GROUP_NORM_SILU_BASE_H_

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace GroupNormSilu {
using namespace AscendC;

constexpr int32_t BLOCK_SIZE = 32;

template <typename T>
class GroupNormSiluBase {
public:
    __aicore__ inline GroupNormSiluBase(){};

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b) {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    };

    __aicore__ inline RoundMode GetRoundMode() {
#if __CCE_AICORE__ == 220
        return RoundMode::CAST_ROUND;
#else
        return RoundMode::CAST_NONE;
#endif
    };

    template <typename T1, bool isAlign = true>
    __aicore__ inline void CopyInData(const LocalTensor<T1>& dstUB, const GlobalTensor<T1>& srcGM,
                                      const int64_t dataCount) {
        if constexpr (isAlign) {
            DataCopy(dstUB, srcGM, dataCount);
        } else {
            if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
                DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
                copyParams.blockLen = dataCount * sizeof(T1);
                DataCopyPadExtParams<T1> padParams = {false, 0, 0, 0};
                DataCopyPad(dstUB, srcGM, copyParams, padParams);
            } else {
                int64_t elementsPerBlock = BLOCK_SIZE / sizeof(T1);
                int64_t dataCountAlign = CeilDiv(dataCount, elementsPerBlock) * elementsPerBlock;
                DataCopy(dstUB, srcGM, dataCountAlign);
            }
        }
    }

    template <typename T1, bool isAlign = true>
    __aicore__ inline void CopyOutData(const GlobalTensor<T1>& dstGM, const LocalTensor<T1>& srcUB,
                                       const int64_t dataCount) {
        if constexpr (isAlign) {
            DataCopy(dstGM, srcUB, dataCount);
        } else {
            if constexpr (PlatformSocInfo::IsDataCopyPadSupport()) {
                DataCopyExtParams copyParams = {1, 0, 0, 0, 0};
                copyParams.blockLen = dataCount * sizeof(T1);
                DataCopyPad(dstGM, srcUB, copyParams);
            } else {
                int64_t elementsPerBlock = BLOCK_SIZE / sizeof(T1);
                int64_t dataCountAlign = CeilDiv(dataCount, elementsPerBlock) * elementsPerBlock;
                DataCopy(dstGM, srcUB, dataCountAlign);
            }
        }
    }
};

}  // namespace GroupNormSilu

#endif  // GROUP_NORM_SILU_BASE_H_
