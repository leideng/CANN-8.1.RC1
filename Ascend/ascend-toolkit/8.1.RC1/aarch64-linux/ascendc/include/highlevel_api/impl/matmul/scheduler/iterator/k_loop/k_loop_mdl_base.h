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
* \file k_loop_mdl_base.h
* \brief
*/
#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_MDL_BASE_H_
#define IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_MDL_BASE_H_

#include "k_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    KLoopMDLBase is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    KLoopMDLBase is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG>
class KLoopMDLBase
{
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    __aicore__ inline KLoopMDLBase() = default;
    __aicore__ inline ~KLoopMDLBase() = default;

    __aicore__ inline void Init(int32_t singleShape)
    {
        SetSingleShape(singleShape);
    }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        const auto& tiling = MATMUL_MODULE(MatmulShapeTiling)->GetTiling();
        int32_t stepKa = tiling.GetStepKa();
        int32_t stepKb = tiling.GetStepKb();
        int32_t baseK = tiling.GetBaseK();
        if constexpr (IsBasicK(MM_CFG)) {
            kIter_ = 1;
        } else {
            kIter_ = Ceil(singleShape, baseK);
        }
        ASCENDC_ASSERT((kIter_ > 0),
            { KERNEL_LOG(KERNEL_ERROR, "kIter_ is %d , which should be larger than 0", kIter_); });
        if (kIter_ > stepKa) {
            if constexpr (!DoMatmulSpecialMDL(MM_CFG)) {
                ASCENDC_ASSERT((tiling.GetStepM() == 1),
                        { KERNEL_LOG(KERNEL_ERROR, "stepM is %d which can only be 1", tiling.GetStepM()); });
            }
        }
        if (kIter_ > stepKb) {
            if constexpr (!DoMatmulSpecialMDL(MM_CFG)) {
                ASCENDC_ASSERT((tiling.GetStepN() == 1),
                    { KERNEL_LOG(KERNEL_ERROR, "stepN is %d which can only be 1", tiling.GetStepN()); });
            }
        }
        if constexpr (NoTailK(MM_CFG)) {
            tailK_ = baseK;
        } else {
            tailK_ = singleShape % baseK;
            if (tailK_ == 0) {
                tailK_ = baseK;
            }
        }
        // get outer loop params
        minStepK_ = stepKa > stepKb ? stepKb : stepKa;
        kaStepFactor_ = stepKa / minStepK_;
        kbStepFactor_ = stepKb / minStepK_;
        ASCENDC_ASSERT((kaStepFactor_ >= 1), { KERNEL_LOG(KERNEL_ERROR,
            "kaStepFactor_ is %d, which should be no less than 1", kaStepFactor_); });
        ASCENDC_ASSERT((kbStepFactor_ >= 1), { KERNEL_LOG(KERNEL_ERROR,
            "kbStepFactor_ is %d, which should be no less than 1", kbStepFactor_); });
        outerKaIter_ = Ceil(singleShape, baseK * stepKa);
        outerKbIter_ = Ceil(singleShape, baseK * stepKb);
        ASCENDC_ASSERT((outerKaIter_ % outerKbIter_ == 0 || outerKbIter_ % outerKaIter_ == 0), {
            KERNEL_LOG(KERNEL_ERROR, "outerKaIter_ %d ,  outerKbIter_ is %d, "
            "outerKaIter_ and outerKbIter_ should be in multiple relationship.", outerKaIter_, outerKbIter_);
        });
        outIter_ = outerKaIter_ > outerKbIter_ ? outerKaIter_ : outerKbIter_;
        tailStepKa_ = singleShape % (baseK * stepKa);
        tailStepKb_ = singleShape % (baseK * stepKb);
        if (tailStepKa_ == 0) {
            tailStepKa_ = baseK * stepKa;
        }
        if (tailStepKb_ == 0) {
            tailStepKb_ = baseK * stepKb;
        }
        isA1KFullLoad_ = stepKa >= kIter_;
        isB1KFullLoad_ = stepKb >= kIter_;
    }

    __aicore__ inline void OuterStart()
    {
        outerIdx_ = 0;
        UpdateOuterParams();
    }

    __aicore__ inline bool OuterNext()
    {
        outerIdx_++;
        if (OuterEnd()) {
            return false;
        } else {
            UpdateOuterParams();
            return true;
        }
    }

    __aicore__ inline bool OuterEnd()
    {
        return outerIdx_ >= outIter_;
    }

    __aicore__ inline bool FirstOuterIter() const
    {
        return outerIdx_ == 0;
    }

    __aicore__ inline bool LastOuterIter() const
    {
        return outerIdx_ + 1 == outIter_;
    }

    __aicore__ inline void InnerStart()
    {
        innerIdx_ = innerStartIdx_;
        UpdateInnerParams();
    }

    __aicore__ inline bool InnerNext()
    {
        innerIdx_++;
        if (InnerEnd()) {
            return false;
        } else {
            UpdateInnerParams();
            return true;
        }
    }

    __aicore__ inline bool InnerEnd()
    {
        return innerIdx_ >= innerStartIdx_ + innerIter_;
    }

    __aicore__ inline bool FirstInnerIter() const
    {
        return innerIdx_ == 0;
    }

    __aicore__ inline int32_t GetTotalIter() const
    {
        return kIter_;
    }

    __aicore__ inline bool IsAKL1FullLoad() const
    {
        return isA1KFullLoad_;
    }

    __aicore__ inline bool IsBKL1FullLoad() const
    {
        return isB1KFullLoad_;
    }

    __aicore__ inline int32_t GetInnerStartIdx() const
    {
        return innerStartIdx_;
    }

    __aicore__ inline int32_t GetOuterIter() const
    {
        return outIter_;
    }

    __aicore__ inline int32_t GetInnerIter() const
    {
        return innerIter_;
    }

    __aicore__ inline int32_t GetOuterIdx() const
    {
        return outerIdx_;
    }

    /**
     * @description: Get current ka outer loop index, used for GetBufferPos in CopyCubeIn
     * @param: void
     * @return: return current ka outerIdx
     */
    __aicore__ inline int32_t GetOuterKaIdx() const
    {
        return outerIdx_ / kaStepFactor_;
    }

    /**
     * @description: Get current kb outer loop index, used for GetBufferPos in CopyCubeIn
     * @param: void
     * @return: return current kb outerIdx
     */
    __aicore__ inline int32_t GetOuterKbIdx() const
    {
        return outerIdx_ / kbStepFactor_;
    }

    /**
     * @description: Get next ka outer loop index, used for ClearL1BufferCache in SchedulerMDL
     * @param: void
     * @return: return next ka outerIdx
     */
    __aicore__ inline int32_t GetNextOuterKaIdx() const
    {
        return (outerIdx_ + 1) / kaStepFactor_;
    }

    /**
     * @description: Get next kb outer loop index, used for ClearL1BufferCache in SchedulerMDL
     * @param: void
     * @return: return next kb outerIdx
     */
    __aicore__ inline int32_t GetNextOuterKbIdx() const
    {
        return (outerIdx_ + 1) / kbStepFactor_;
    }

    __aicore__ inline int32_t GetInnerIdx() const
    {
        return innerIdx_;
    }

    __aicore__ inline int32_t GetTileShapeA() const
    {
        return tileShapeA_;
    }

    /**
     * @description: Get specified loop index's kaL1 length, used when Preload is enabled
     * @param: curOuterIdx: specified outer loop index
     * @return: return kaL1 length
     */
    __aicore__ inline int32_t GetTileShapeAOf(int32_t curOuterIdx) const
    {
        return (curOuterIdx + 1 >= outerKaIter_) ? tailStepKa_ :
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa() *
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
    }

    __aicore__ inline int32_t GetTileShapeB() const
    {
        return tileShapeB_;
    }

    /**
     * @description: Get specified loop index's kbL1 length, used when Preload is enabled
     * @param: curOuterIdx: specified outer loop index
     * @return: return kbL1 length
     */
    __aicore__ inline int32_t GetTileShapeBOf(int32_t curOuterIdx) const
    {
        return (curOuterIdx + 1 >= outerKbIter_) ? tailStepKb_ :
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb() *
            MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
    }

    __aicore__ inline int32_t GetTileBlockShapeA() const
    {
        return tileBlockShapeA_;
    }

    __aicore__ inline int32_t GetTileBlockShapeB() const
    {
        return tileBlockShapeB_;
    }

    __aicore__ inline int32_t GetBaseShape() const
    {
        return baseShape_;
    }

    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        return baseBlockShape_;
    }

protected:
    __aicore__ inline void UpdateOuterParams()
    {
        auto tilingStepKa = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKa();
        auto tilingStepKb = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetStepKb();
        innerStartIdx_ = outerIdx_ * minStepK_;
        int32_t curKaOuterIdx = innerStartIdx_ / tilingStepKa;
        int32_t curKbOuterIdx = innerStartIdx_ / tilingStepKb;
        ASCENDC_ASSERT((innerStartIdx_ >= curKaOuterIdx * tilingStepKa), {
            KERNEL_LOG(KERNEL_ERROR, "k is %d , minStepK_ is %d, curKaOuterIdx is %d, stepKa is %d,"
            "(k * minStepK_) should >= (curKaOuterIdx * stepKa)", outerIdx_, minStepK_, curKaOuterIdx, tilingStepKa);
        });
        ASCENDC_ASSERT((innerStartIdx_ >= curKbOuterIdx * tilingStepKb), {
            KERNEL_LOG(KERNEL_ERROR, "k is %d , minStepK_ is %d, curKbOuterIdx is %d, stepKb is %d,"
            "(k * minStepK_) should >= (curKbOuterIdx * stepKb)", outerIdx_, minStepK_, curKbOuterIdx, tilingStepKb);
        });

        auto tilingBaseK = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        tileShapeA_ =
            (curKaOuterIdx + 1 >= outerKaIter_) ? tailStepKa_ : tilingStepKa * tilingBaseK;
        tileShapeB_ =
            (curKbOuterIdx + 1 >= outerKbIter_) ? tailStepKb_ : tilingStepKb * tilingBaseK;
        tileBlockShapeA_ = Ceil(tileShapeA_, c0Size_);
        tileBlockShapeB_ = Ceil(tileShapeB_, c0Size_);

        // update inner loop common params
        baseSize_ = (outerIdx_ + 1 == kIter_) ? tailK_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        baseBlockShape_ = Ceil(baseSize_, c0Size_);
        int32_t baseBlockSize = baseBlockShape_ * c0Size_;
        int32_t tileShape = tileShapeA_ > tileShapeB_ ? tileShapeB_ : tileShapeA_;
        innerIter_ = tileShape / baseBlockSize;
        innerTailK_ = tileShape - innerIter_ * baseBlockSize;
        if (innerTailK_ == 0) {
            innerTailK_ = baseBlockSize;
        } else {
            innerIter_ = innerIter_ + 1;
        }
    }

    __aicore__ inline void UpdateInnerParams()
    {
        if constexpr (IsStaticPaddingEnable(MM_CFG)) {
            baseShape_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
        } else {
            baseShape_ = (innerIdx_ == innerStartIdx_ + innerIter_ -1) ? innerTailK_ : baseSize_;
        }
    }

protected:
    int32_t tailK_;
    int32_t tailStepKa_;
    int32_t tailStepKb_;
    int32_t minStepK_;         // lesser value of stepKa and stepKb
    int32_t baseSize_;         // kL1 base size, used for updating baseShape_
    int32_t innerTailK_;       // kL1 tail size of current outer loop, used for updating baseShape_

    int32_t tileShapeA_;       // kaL1 length
    int32_t tileShapeB_;       // kbL1 length
    int32_t tileBlockShapeA_;  // kaL1 block num
    int32_t tileBlockShapeB_;  // kbL1 block num
    int32_t kaStepFactor_;     // indicates the coefficient of stepka and minStepK_
    int32_t kbStepFactor_;     // indicates the coefficient of stepkb and minStepK_

    int32_t baseShape_;        // kL0 length
    int32_t baseBlockShape_;   // kL0 block num

    int32_t kIter_;           // total iterations counts
    int32_t outIter_;         // outer loop counts, greater value of outerKaIter_ and outerKbIter_;
    int32_t innerIter_;       // inner loop counts
    int32_t outerKaIter_;     // outer ka loop counts
    int32_t outerKbIter_;     // outer kb loop counts
    int32_t outerIdx_ {0};        // current outer loop index
    int32_t innerStartIdx_;   // inner loop start index of current outer loop, used for indicating k's index
                               // when load to L1 and calculating L1 offset when load to l0 
    int32_t innerIdx_ {0};        // current inner loop index
    bool isA1KFullLoad_, isB1KFullLoad_;

    constexpr static int32_t c0Size_ = AuxGetC0Size<typename A_TYPE::T>();
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _K_LOOP_MDL_BASE_H_
