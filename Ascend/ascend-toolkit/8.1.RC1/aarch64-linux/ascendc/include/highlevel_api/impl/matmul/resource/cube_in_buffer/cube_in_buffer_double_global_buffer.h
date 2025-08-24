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
* \file cube_in_buffer_double_global_buffer.h
* \brief
*/
#ifndef IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_DOUBLE_GLOBAL_BUFFER_H
#define IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_DOUBLE_GLOBAL_BUFFER_H

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
GetCubeInBufferType<INPUT_TYPE, MM_CFG>() == CubeInBufferType::DOUBLE_GLOBAL_BUFFER>> {
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(KLoop);
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline CubeInBuffer() = default;
    __aicore__ inline ~CubeInBuffer() = default;
    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheNum)
    {
        baseBlockSize_ = baseBlockSize;
        groupCache0_.Init();
        groupCache1_.Init();
        int32_t matrixByteSize = baseBlockSize_ * AscendC::GetBitSize<TransT>() / ONE_BYTE_BIT_SIZE;
        groupCache0_.InitBuffer(matrixByteSize * cacheNum);
        groupCache1_.InitBuffer(matrixByteSize * cacheNum);
    }

    __aicore__ inline void Destroy() {}

    __aicore__ inline LocalTensor<TransT> AllocTensor(int32_t bufferPos = -1)
    {
        ASCENDC_ASSERT(bufferPos != -1,
            { KERNEL_LOG(KERNEL_ERROR, "bufferPos in AllocTensor for global que version should not be -1."); });
        if (isCache0SameAddr_) {
            return groupCache0_.template GetCacheHead<TransT>()[bufferPos * baseBlockSize_];
        } else if (isCache1SameAddr_) {
            return groupCache1_.template GetCacheHead<TransT>()[bufferPos * baseBlockSize_];
        } else {
            GlobalCache* curGroupCache = isCache0_ ? &groupCache0_ : &groupCache1_;
            GlobalTensor<TransT> inputTensor;
            inputTensor.SetGlobalBuffer(inputAddr_);
            curGroupCache->template SetOrgTensor<TransT>(inputTensor);
            isCache0_ = !isCache0_;
            return curGroupCache->template AllocTensor<TransT>();
        }
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>)
    {
        (void) bufferPos;
        (void) tensor;
    }

    __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1)
    {
        (void) bufferPos;
        GlobalTensor<TransT> inputTensor;
        inputTensor.SetGlobalBuffer(inputAddr_);
        isCache0SameAddr_ = groupCache0_.template Hit<TransT>(inputTensor);
        isCache1SameAddr_ = groupCache1_.template Hit<TransT>(inputTensor);
        return (isCache0SameAddr_ && (iterIndex + 1 <= groupCache0_.GetCacheSize())) ||
            (isCache1SameAddr_ && (iterIndex + 1 <= groupCache1_.GetCacheSize()));
    }

    __aicore__ inline void Reset() {}

    __aicore__ inline LocalTensor<TransT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1)
    {
        (void) bufferPos;
        if (isCache0SameAddr_) {
            return groupCache0_.template GetCacheHead<TransT>()[bufferPos * baseBlockSize_];
        } else if (isCache1SameAddr_) {
            return groupCache1_.template GetCacheHead<TransT>()[bufferPos * baseBlockSize_];
        } else {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Please call GetBuffer only when Hit is true."); });
            return NULL_TENSOR<TransT>;
        }
    }

    __aicore__ inline void SetOrgTensor(GlobalTensor<TransT> globalMatrix)
    {
        inputAddr_ = globalMatrix.address_;
        if (!groupCache0_.template Hit<TransT>(globalMatrix) && !groupCache1_.template Hit<TransT>(globalMatrix)) {
            GlobalCache* curGroupCache = isCache0_ ? &groupCache0_ : &groupCache1_;
            curGroupCache->template ClearCache<TransT>();
        }
    }

    __aicore__ inline void EnQue(LocalTensor<TransT>& tensor)
    {
        if (isCache0SameAddr_) {
            groupCache0_.template EnQue<TransT>(tensor);
            if (IsTailBlock()) {
                groupCache0_.ReduceCacheSize();
            }
        } else if (isCache1SameAddr_) {
            groupCache1_.template EnQue<TransT>(tensor);
            if (IsTailBlock()) {
                groupCache1_.ReduceCacheSize();
            }
        }
    }

    __aicore__ inline void DeQue()
    {
        if (isCache0SameAddr_) {
            groupCache0_.template DeQue<TransT>();
        } else if (isCache1SameAddr_) {
            groupCache1_.template DeQue<TransT>();
        }
    }

private:
    __aicore__ inline bool IsTailBlock()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            return (MATMUL_MODULE(NLoop)->GetBaseShape() != MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN()) ||
                   (MATMUL_MODULE(KLoop)->GetBaseShape() != MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK());
        } else {
            return (MATMUL_MODULE(MLoop)->GetBaseShape() != MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM()) ||
                   (MATMUL_MODULE(KLoop)->GetBaseShape() != MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK());
        }
    }

    GlobalCache groupCache0_;
    GlobalCache groupCache1_;
    __gm__ TransT* inputAddr_;
    int32_t baseBlockSize_;
    bool isCache0_ { true };
    bool isCache0SameAddr_ { false };
    bool isCache1SameAddr_ { false };
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _CUBE_IN_BUFFER_DOUBLE_GLOBAL_BUFFER_H_
