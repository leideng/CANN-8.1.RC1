/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
* \file cube_in_buffer_double_buffer_sparse.h
* \brief
*/

#ifndef IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_DOUBLE_BUFFER_SPARSE_H
#define IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_DOUBLE_BUFFER_SPARSE_H

#include "cube_in_buffer_intf.h"

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
GetCubeInBufferType<INPUT_TYPE, MM_CFG>() == CubeInBufferType::DOUBLE_BUFFER_SPARSE>> {
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(Context);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline CubeInBuffer() {}
    __aicore__ inline ~CubeInBuffer() {}
    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheNum)
    {
        int32_t matrixByteSize =  baseBlockSize * AscendC::GetBitSize<TransT>() / ONE_BYTE_BIT_SIZE;
        int32_t stepSize = GetTotalCacheNum();
        ASCENDC_ASSERT(stepSize > 0,
            { KERNEL_LOG(KERNEL_ERROR, "stepSize should not be less than 0."); });
        cacheFactor_ = (cacheNum / stepSize - 1) & 1;
        int32_t queDepth = cacheFactor_ == 0 ? SINGLE_QUE : DOUBLE_QUE;
        GetTPipePtr()->InitBuffer(qid_, queDepth, ((matrixByteSize * stepSize) >> 1));
        GetTPipePtr()->InitBuffer(idxQid_, queDepth, ((matrixByteSize * stepSize) >> 3)); // Index buffer size:B/2/4
    }

    __aicore__ inline void Destroy()
    {
        isCachingPing_ = false;
        isCachingPong_ = false;
        qid_.FreeAllEvent();
        idxQid_.FreeAllEvent();
    }

    __aicore__ inline LocalTensor<TransT> AllocTensor(int32_t bufferPos = -1)
    {
        ASCENDC_ASSERT(bufferPos != -1,
            { KERNEL_LOG(KERNEL_ERROR, "bufferPos in AllocTensor for only db version should not be -1."); });
        LocalTensor<TransT> tensor = qid_.template AllocTensor<TransT>();
        LocalTensor<uint8_t> idxTensor = idxQid_.template AllocTensor<uint8_t>();
        int32_t cachePos = bufferPos & cacheFactor_;
        SetCache(cachePos, tensor, idxTensor);
        SetBufferCaching(cachePos, true);
        MATMUL_MODULE(MatmulTensorInfo)->SetLocalSparseIndex(idxTensor);
        return tensor;
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>)
    {
        ASCENDC_ASSERT(bufferPos != -1, { KERNEL_LOG(KERNEL_ERROR,
            "bufferPos in FreeTensor for only db version should not be -1."); });
        int32_t cachePos = bufferPos & cacheFactor_;
        if (IsBufferCaching(cachePos)) {
            qid_.FreeBuffer(GetCache(cachePos));
            idxQid_.FreeBuffer(GetIdxCache(cachePos));
            SetBufferCaching(cachePos, false);
        }
    }

    __aicore__ inline void Reset()
    {
        if (IsBufferCaching(0)) {
            qid_.FreeBuffer(GetCache(0));
            idxQid_.FreeBuffer(GetIdxCache(0));
            SetBufferCaching(0, false);
        }
        if (IsBufferCaching(1)) {
            qid_.FreeBuffer(GetCache(1));
            idxQid_.FreeBuffer(GetIdxCache(1));
            SetBufferCaching(1, false);
        }
    }

    __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1)
    {
        ASCENDC_ASSERT(bufferPos != -1, { KERNEL_LOG(KERNEL_ERROR,
            "bufferPos in Hit for only db version should not be -1."); });
        return iterIndex != 0 || IsBufferCaching(bufferPos & cacheFactor_);
    }

    __aicore__ inline LocalTensor<TransT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1)
    {
        (void) iterIndex;
        ASCENDC_ASSERT(bufferPos != -1, { KERNEL_LOG(KERNEL_ERROR,
            "bufferPos in Hit for only db version should not be -1."); });
        LocalTensor<TransT> tensor;
        tensor.SetAddr(qid_.GetBufferAddr(GetCache(bufferPos & cacheFactor_)));

        LocalTensor<uint8_t> idxTensor;
        idxTensor.SetAddr(idxQid_.GetBufferAddr(GetIdxCache(bufferPos & cacheFactor_)));
        MATMUL_MODULE(MatmulTensorInfo)->SetLocalSparseIndex(idxTensor);
        return tensor;
    }

    __aicore__ inline void EnQue(LocalTensor<TransT>& tensor)
    {
        qid_.EnQue(tensor);
        auto idxTensor = MATMUL_MODULE(MatmulTensorInfo)->GetLocalSparseIndex();
        idxQid_.EnQue(idxTensor);
    }

    __aicore__ inline void DeQue()
    {
        (void) qid_.DeQue();
        (void) idxQid_.DeQue();
    }

private:
    __aicore__ inline int32_t GetTotalCacheNum()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb() *
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN();
        } else {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa() *
                MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM();
        }
    }

    __aicore__ inline int32_t GetMajorCacheNum()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb();
        } else {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa();
        }
    }

    __aicore__ inline auto& GetCache(bool isPong)
    {
        return isPong ? cachePong_ : cachePing_;
    }

    __aicore__ inline auto& GetIdxCache(bool isPong)
    {
        return isPong ? idxCachePong_ : idxCachePing_;
    }

    __aicore__ inline void SetBufferCaching(bool isPong, bool isCaching)
    {
        if (isPong) {
            isCachingPong_ = isCaching;
        } else {
            isCachingPing_ = isCaching;
        }
    }

    __aicore__ inline bool IsBufferCaching(bool isPong)
    {
        return isPong ? isCachingPong_ : isCachingPing_;
    }

    __aicore__ inline void SetCache(bool isPong, const LocalTensor<TransT>& cacheTensor, const LocalTensor<uint8_t>& idxCacheTensor)
    {
        if (isPong) {
            cachePong_ = cacheTensor.GetBufferHandle();
            idxCachePong_ = idxCacheTensor.GetBufferHandle();
        } else {
            cachePing_ = cacheTensor.GetBufferHandle();
            idxCachePing_ = idxCacheTensor.GetBufferHandle();
        }
    }

private:
    TBufHandle cachePing_;
    typename CubeInQueType<INPUT_TYPE>::QUE qid_;
    TBufHandle cachePong_;
    typename CubeInQueType<INPUT_TYPE>::QUE idxQid_;
    TBufHandle idxCachePing_;
    int32_t cacheFactor_;
    TBufHandle idxCachePong_;
    bool isCachingPing_ { false };
    bool isCachingPong_ { false };
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _CUBE_IN_BUFFER_DOUBLE_BUFFER_SPARSE_H_
