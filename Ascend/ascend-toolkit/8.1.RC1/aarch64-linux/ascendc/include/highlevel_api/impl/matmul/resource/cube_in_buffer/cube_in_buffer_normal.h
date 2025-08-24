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
* \file cube_in_buffer_normal.h
* \brief
*/

#ifndef IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_NORMAL_H
#define IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_NORMAL_H

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
GetCubeInBufferType<INPUT_TYPE, MM_CFG>() == CubeInBufferType::NORMAL>> {
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(KLoop);
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline CubeInBuffer() {}
    __aicore__ inline ~CubeInBuffer() {}
    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheNum)
    {
        baseBlockSize_ = baseBlockSize;
        int32_t matrixByteSize =  baseBlockSize_ * AscendC::GetBitSize<TransT>() / ONE_BYTE_BIT_SIZE;
        int32_t reduceAxisCnt = MATMUL_MODULE(KLoop)->GetTotalIter();
        auto tpipePtr = GetTPipePtr();
        if (cacheNum > DB_FACTOR) {
            if (cacheNum < reduceAxisCnt * GetMajorCacheNum()) {
                // k not full load
                cacheSize_ = cacheNum - DB_FACTOR;
                tpipePtr->InitBuffer(qidCache_, SINGLE_QUE, cacheSize_ * matrixByteSize);
                tpipePtr->InitBuffer(qid_, DB_FACTOR, matrixByteSize);
            } else {
                // k full load
                cacheSize_ = cacheNum;
                tpipePtr->InitBuffer(qidCache_, SINGLE_QUE, cacheSize_ * matrixByteSize);
            }
        } else {
            if (cacheNum < reduceAxisCnt * GetMajorCacheNum()) {
                // k not full load
                cacheSize_ = 0;
                tpipePtr->InitBuffer(qid_, cacheNum, matrixByteSize);
            } else if (reduceAxisCnt == 1 && cacheNum == DOUBLE_QUE) {
                // k full load, db on m axis
                cacheSize_ = 0;
                tpipePtr->InitBuffer(qid_, DOUBLE_QUE, matrixByteSize);
            } else {
                // k full load
                cacheSize_ = cacheNum;
                tpipePtr->InitBuffer(qidCache_, SINGLE_QUE, cacheSize_ * matrixByteSize);
            }
        }
    }

    __aicore__ inline void Destroy()
    {
        if (cacheProc_ > 0) {
            ASCENDC_ASSERT((qidCache_.GetState(cacheHead_) != TBufState::FREE),
                        { KERNEL_LOG(KERNEL_ERROR, "cacheHead_ state can not be TBufState::FREE"); });
            qidCache_.FreeTensor(cacheHead_);
            cacheProc_ = 0;
        }
        qid_.FreeAllEvent();
        qidCache_.FreeAllEvent();
        cacheAlloc_ = false;
    }

    __aicore__ inline LocalTensor<TransT> AllocTensor(int32_t bufferPos = -1)
    {
        ASCENDC_ASSERT(bufferPos != -1,
                       { KERNEL_LOG(KERNEL_ERROR, "bufferPos in AllocTensor for normal version should not be -1."); });
        if (bufferPos >= cacheSize_) {
            cacheAlloc_ = false;
            return qid_.template AllocTensor<TransT>();
        } else if (cacheProc_ == 0) {
            cacheHead_ = qidCache_.template AllocTensor<TransT>(); // To use que to insert events
        } else if (cacheProc_ >= cacheSize_) {
            ASCENDC_ASSERT((false), { // Logically, it shouldn't be entered.
                KERNEL_LOG(KERNEL_ERROR, "illegal branch");
            });
            qidCache_.FreeTensor(cacheHead_);
            cacheHead_ = qidCache_.template AllocTensor<TransT>(); // To use que to insert events
        }
        ++cacheProc_;
        cacheAlloc_ = true;
        return cacheHead_[bufferPos * baseBlockSize_];
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>)
    {
        ASCENDC_ASSERT(bufferPos != -1,
                       { KERNEL_LOG(KERNEL_ERROR, "bufferPos in FreeTensor for normal version should not be -1."); });
        if (bufferPos >= cacheSize_) {
            qid_.FreeTensor(const_cast<LocalTensor<TransT>&>(tensor));
        }
    }

    __aicore__ inline void Reset()
    {
        if (cacheProc_ > 0) {
            qidCache_.FreeTensor(cacheHead_);
            cacheProc_ = 0;
        }
    }

    __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1)
    {
        (void) bufferPos;
        return (iterIndex < cacheSize_ && iterIndex < cacheProc_);
    }

    __aicore__ inline LocalTensor<TransT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1)
    {
        (void) bufferPos;
        return cacheHead_[iterIndex * baseBlockSize_];
    }

    __aicore__ inline void EnQue(LocalTensor<TransT>& tensor)
    {
        if (cacheAlloc_) {
            qidCache_.EnQue(tensor);
        } else {
            qid_.EnQue(tensor);
        }
    }

    __aicore__ inline void DeQue()
    {
        if (cacheAlloc_) {
            (void) qidCache_.DeQue();
        } else {
            (void) qid_.DeQue();
        }
    }

private:

    __aicore__ inline int32_t GetMajorCacheNum()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            if constexpr (DoMatmulSpecialBasicBlock(MM_CFG)) {
                return ToMatmulConfig(MM_CFG).stepM;
            } else {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM();
            }
        } else {
            if constexpr (DoMatmulSpecialBasicBlock(MM_CFG)) {
                return ToMatmulConfig(MM_CFG).stepN;
            } else {
                return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN();
            }   
        }
    }

    typename CubeInQueType<INPUT_TYPE>::QUE qid_;
    typename CubeInQueType<INPUT_TYPE>::QUE qidCache_;
    LocalTensor<TransT> cacheHead_; // Allocate and release using qidCache_
    int32_t baseBlockSize_;
    int32_t cacheSize_;
    int32_t cacheProc_ { 0 };
    bool cacheAlloc_ { false };
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _CUBE_IN_BUFFER_NORMAL_H_
