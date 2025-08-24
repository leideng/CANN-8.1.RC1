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
* \file cube_in_buffer_double_buffer.h
* \brief
*/

#ifndef IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_DOUBLE_BUFFER_H
#define IMPL_MATMUL_RESOURCE_CUBE_IN_BUFFER_CUBE_IN_BUFFER_DOUBLE_BUFFER_H

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
GetCubeInBufferType<INPUT_TYPE, MM_CFG>() == CubeInBufferType::DOUBLE_BUFFER>> {
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    using TransT = typename INPUT_TYPE::TRANS_T;
public:
    __aicore__ inline CubeInBuffer() {}
    __aicore__ inline ~CubeInBuffer() {}
    __aicore__ inline void Init(int32_t baseBlockSize, int32_t cacheNum)
    {
        int32_t matrixByteSize =  baseBlockSize * AscendC::GetBitSize<TransT>() / ONE_BYTE_BIT_SIZE;
        int32_t stepSize = GetTotalCacheNum();
        cacheFactor_ = (cacheNum / stepSize - 1) & 1;
        int32_t queDepth = cacheFactor_ == 0 ? SINGLE_QUE : DOUBLE_QUE;
        GetTPipePtr()->InitBuffer(qid_, queDepth, matrixByteSize * stepSize + GetBankConflictSize());
#if __CCE_AICORE__ == 200
        if (IsFromUB()) {
            eventIDMte3ToMte1_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE1>());
        }
#endif
    }

    __aicore__ inline void Destroy()
    {
        isCachingPing_ = false;
        isCachingPong_ = false;
        qid_.FreeAllEvent();
#if __CCE_AICORE__ == 200
        if (IsFromUB()) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE1>(eventIDMte3ToMte1_);
        }
#endif
    }

    __aicore__ inline LocalTensor<TransT> AllocTensor(int32_t bufferPos = -1)
    {
        ASCENDC_ASSERT(bufferPos != -1,
            { KERNEL_LOG(KERNEL_ERROR, "bufferPos in AllocTensor for only db version should not be -1."); });
        LocalTensor<TransT> tensor = qid_.template AllocTensor<TransT>();
        int32_t cachePos = bufferPos & cacheFactor_;
        SetCache(cachePos, tensor);
        SetBufferCaching(cachePos, true);
        return tensor;
    }

    __aicore__ inline void FreeTensor(int32_t bufferPos = -1, const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>)
    {
        ASCENDC_ASSERT(bufferPos != -1,
            { KERNEL_LOG(KERNEL_ERROR, "bufferPos in FreeTensor for only db version should not be -1."); });
        int32_t cachePos = bufferPos & cacheFactor_;
        if (IsBufferCaching(cachePos)) {
            qid_.FreeBuffer(GetCache(cachePos));
            SetBufferCaching(cachePos, false);
        }
    }

    __aicore__ inline void Reset()
    {
        if (IsBufferCaching(0)) {
            qid_.FreeBuffer(GetCache(0));
            SetBufferCaching(0, false);
        }
        if (IsBufferCaching(1)) {
            qid_.FreeBuffer(GetCache(1));
            SetBufferCaching(1, false);
        }
    }

    __aicore__ inline bool Hit(int32_t iterIndex, int32_t bufferPos = -1)
    {
        ASCENDC_ASSERT(bufferPos != -1,
            { KERNEL_LOG(KERNEL_ERROR, "bufferPos in Hit for only db version should not be -1."); });
        return iterIndex != 0 || IsBufferCaching(bufferPos & cacheFactor_);
    }

    __aicore__ inline LocalTensor<TransT> GetBuffer(int32_t iterIndex, int32_t bufferPos = -1)
    {
        (void) iterIndex;
        ASCENDC_ASSERT(bufferPos != -1,
            { KERNEL_LOG(KERNEL_ERROR, "bufferPos in Hit for only db version should not be -1."); });
        LocalTensor<TransT> tensor;
        tensor.SetAddr(qid_.GetBufferAddr(GetCache(bufferPos & cacheFactor_)));
        return tensor;
    }

    __aicore__ inline void EnQue(LocalTensor<TransT>& tensor)
    {
#if __CCE_AICORE__ == 200
        if (IsFromUB()) {
            SetFlag<HardEvent::MTE3_MTE1>(eventIDMte3ToMte1_);
        } else {
            qid_.EnQue(tensor);
        }
#else
        qid_.EnQue(tensor);
#endif
    }

    __aicore__ inline void DeQue()
    {
#if __CCE_AICORE__ == 200
        if (IsFromUB()) {
            WaitFlag<HardEvent::MTE3_MTE1>(eventIDMte3ToMte1_);
        } else {
            (void) qid_.DeQue();
        }
#else
        (void) qid_.DeQue();
#endif
    }

private:
    __aicore__ inline int32_t GetTotalCacheNum()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM();
        } else {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN();
        }
    }

    __aicore__ inline int32_t GetMajorCacheNum()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa();
        } else {
            return MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb();
        }
    }

    __aicore__ inline int32_t GetBankConflictSize()
    {
        if constexpr (MatmulFeatureTrait<MM_CFG>::IsNeedUB() && ToMatmulConfig(MM_CFG).enVecND2NZ) {
            if constexpr (INPUT_TYPE::format == CubeFormat::ND) {
                constexpr int32_t c0Size = AuxGetC0Size<typename INPUT_TYPE::TRANS_T>();
                if constexpr (INPUT_TYPE::isTrans) {
                    if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
                        bool isBankConflict = Ceil(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM(), c0Size) *
                            ONE_BLK_SIZE % BANK_CONFLICT_SIZE == 0;
                        return isBankConflict ? MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK() * c0Size *
                            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa() * sizeof(typename INPUT_TYPE::TRANS_T) : 0;
                    } else {
                        bool isBankConflict = Ceil(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK(), c0Size) *
                            ONE_BLK_SIZE % BANK_CONFLICT_SIZE == 0;
                        return isBankConflict ? MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN() * c0Size *
                            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN() * sizeof(typename INPUT_TYPE::TRANS_T) : 0;
                    }
                } else {
                    if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
                        bool isBankConflict = Ceil(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK(), c0Size) *
                            ONE_BLK_SIZE % BANK_CONFLICT_SIZE == 0;
                        return isBankConflict ? MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseM() * c0Size *
                            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepM() * sizeof(typename INPUT_TYPE::TRANS_T) : 0;
                    } else {
                        bool isBankConflict = Ceil(MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepN() * MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseN(), c0Size) *
                            ONE_BLK_SIZE % BANK_CONFLICT_SIZE == 0;
                        return isBankConflict ? MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK() * c0Size *
                            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb() * sizeof(typename INPUT_TYPE::TRANS_T) : 0;
                    }
                }
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    }

    __aicore__ inline auto& GetCache(bool isPong)
    {
        return isPong ? cachePong_ : cachePing_;
    }

    __aicore__ inline bool IsBufferCaching(bool isPong)
    {
        return isPong ? isCachingPong_ : isCachingPing_;
    }

    __aicore__ inline void SetCache(bool isPong, const LocalTensor<TransT>& cacheTensor)
    {
        if (isPong) {
    #if __CCE_AICORE__ == 220
            Barrier();
    #endif
            cachePong_ = cacheTensor.GetBufferHandle();
        } else {
            cachePing_ = cacheTensor.GetBufferHandle();
        }
    }

    __aicore__ inline void SetBufferCaching(bool isPong, bool isCaching)
    {
        if (isPong) {
    #if __CCE_AICORE__ == 220
            Barrier();
    #endif
            isCachingPong_ = isCaching;
        } else {
            isCachingPing_ = isCaching;
        }
    }

    __aicore__ inline bool IsFromUB()
    {
        return IsSameType<typename INPUT_TYPE::T, int8_t>::value &&
            ((INPUT_TYPE::TAG == InputTypeTag::A && MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA()) ||
            (INPUT_TYPE::TAG == InputTypeTag::B && !MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB()));
    }

private:
    TBufHandle cachePing_;
    typename CubeInQueType<INPUT_TYPE>::QUE qid_;
    TBufHandle cachePong_;
    int32_t cacheFactor_;
    bool isCachingPing_ { false };
    bool isCachingPong_ { false };
#if __CCE_AICORE__ == 200
    event_t eventIDMte3ToMte1_;
#endif
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _CUBE_IN_BUFFER_DOUBLE_BUFFER_H_
