/**
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
* \file cube_in_buffer_single_global_buffer.h
* \brief
*/
#ifndef IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_SINGLE_GLOBAL_BUFFER_H
#define IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_SINGLE_GLOBAL_BUFFER_H

#include "cube_in_buffer_intf.h"
#include "global_cache.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    CubeInBuffer is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    CubeInBuffer is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class CubeInBuffer<IMPL, INPUT_TYPE, MM_CFG, enable_if_t<
GetCubeInBufferType<INPUT_TYPE, MM_CFG>() == CubeInBufferType::SINGLE_GLOBAL_BUFFER>> {
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline CubeInBuffer() {}
    __aicore__ inline ~CubeInBuffer() {}
    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheNum)
    {
        baseBlockSize_ = baseBlockSize;
        int32_t matrixByteSize = baseBlockSize_ * AscendC::GetBitSize<TransT>() / ONE_BYTE_BIT_SIZE;
        GetGlobalCachePtr()->InitBuffer(matrixByteSize * cacheNum);
    }

    __aicore__ inline void Destroy() {}

    __aicore__ inline LocalTensor<TransT> AllocTensor(int32_t bufferPos = -1)
    {
        ASCENDC_ASSERT(bufferPos != -1,
            { KERNEL_LOG(KERNEL_ERROR, "bufferPos in AllocTensor for global que version should not be -1."); });
        GlobalTensor<TransT> inputTensor;
        inputTensor.SetGlobalBuffer(inputAddr_);
        if (GetGlobalCachePtr()->template Hit<TransT>(inputTensor)) {
            return GetGlobalCachePtr()->template GetCacheHead<TransT>()[bufferPos * baseBlockSize_];
        } else {
            GetGlobalCachePtr()->template SetOrgTensor<TransT>(inputTensor);
            return GetGlobalCachePtr()->template AllocTensor<TransT>();
        }
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>)
    {
        (void) bufferPos;
        (void) tensor;
    }

    __aicore__ inline void Reset() {}

    __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1)
    {
        (void) bufferPos;
        GlobalTensor<TransT> inputTensor;
        inputTensor.SetGlobalBuffer(inputAddr_);
        return GetGlobalCachePtr()->template Hit<TransT>(inputTensor) &&
            (iterIndex + 1 <= GetGlobalCachePtr()->GetCacheSize()) && IsDataAmountEqual();
    }

    __aicore__ inline LocalTensor<TransT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1)
    {
        (void) bufferPos;
        return GetGlobalCachePtr()->template GetCacheHead<TransT>()[iterIndex * baseBlockSize_];
    }

    __aicore__ inline void SetOrgTensor(const GlobalTensor<TransT>& globalMatrix)
    {
        inputAddr_ = globalMatrix.address_;
        if (!GetGlobalCachePtr()->template Hit<TransT>(globalMatrix)) {
            GetGlobalCachePtr()->template ClearCache<TransT>();
        }
    }

    __aicore__ inline void EnQue(LocalTensor<TransT>& tensor)
    {
        GetGlobalCachePtr()->template EnQue<TransT>(tensor);
        if (IsTailBlock()) {
            GetGlobalCachePtr()->ReduceCacheSize();
        }
    }

    __aicore__ inline void DeQue()
    {
        GetGlobalCachePtr()->template DeQue<TransT>();
    }

private:
    __aicore__ inline bool IsTailBlock()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            return (MATMUL_MODULE(MLoop)->GetBaseShape() != MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM()) ||
                   (MATMUL_MODULE(KLoop)->GetBaseShape() != MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK());
        } else {
            return (MATMUL_MODULE(NLoop)->GetBaseShape() != MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN()) ||
                   (MATMUL_MODULE(KLoop)->GetBaseShape() != MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK());
        }
    }

    __aicore__ inline bool IsDataAmountEqual()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            return (MATMUL_MODULE(MLoop)->GetBaseShape() == MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM()) &&
                   (MATMUL_MODULE(KLoop)->GetBaseShape() == MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK());
        } else {
            return (MATMUL_MODULE(KLoop)->GetBaseShape() == MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK()) &&
                   (MATMUL_MODULE(NLoop)->GetBaseShape() == MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN());
        }
    }

    int32_t baseBlockSize_;
    __gm__ TransT* inputAddr_;
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _CUBE_IN_BUFFER_SINGLE_GLOBAL_BUFFER_H_
