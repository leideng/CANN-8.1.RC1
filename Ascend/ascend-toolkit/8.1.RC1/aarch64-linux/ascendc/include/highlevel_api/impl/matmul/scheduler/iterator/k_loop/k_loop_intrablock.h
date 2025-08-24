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
* \file k_loop_intrablock.h
* \brief
*/
#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_INTRABLOCK_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_INTRABLOCK_H

#include "k_loop_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    KLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    KLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG>
class KLoop<IMPL, TRANS_T, A_TYPE, MM_CFG, enable_if_t<IsIntrablock<MM_CFG>>>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);
    MATMUL_USE_MODULE(MatmulSubBlockInfo);

public:
    __aicore__ inline KLoop() = default;
    __aicore__ inline ~KLoop() = default;

    __aicore__ inline void Init(int32_t singleShape)
    {
        SetSingleShape(singleShape);
    }

    __aicore__ inline void SetSingleShape(int32_t singleShape)
    {
        if (MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx() == 0) {
            kIter_ = Ceil(singleShape, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK());
            tailK_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            if (tailK_ == 0) {
                tailK_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            }
            ASCENDC_ASSERT((kIter_ > 0),
                    { KERNEL_LOG(KERNEL_ERROR, "kIter_ is %d , which should be larger than 0", kIter_); });
        } else {
            intrablockKIter_ = Ceil(singleShape, MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK());
            intrablockTailK_ = singleShape % MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            if (intrablockTailK_ == 0) {
                intrablockTailK_ = MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            }
            ASCENDC_ASSERT((intrablockKIter_ > 0),
                    { KERNEL_LOG(KERNEL_ERROR, "kIter_ is %d , which should be larger than 0", intrablockKIter_); });
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void OuterStart()
    {
        if constexpr (IS_INTRA_BLOCK) {
            intrablockOuterIdx_ = 0;
        } else {
            outerIdx_ = 0;
        }
        UpdateOuterParams<IS_INTRA_BLOCK>();
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool OuterNext()
    {
        if constexpr (IS_INTRA_BLOCK) {
            intrablockOuterIdx_++;
        } else {
            outerIdx_++;
        }
        UpdateOuterParams<IS_INTRA_BLOCK>();
        return !OuterEnd<IS_INTRA_BLOCK>();
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool OuterEnd()
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockOuterIdx_ == intrablockKIter_;
        } else {
            return outerIdx_ == kIter_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool FirstOuterIter() const
    {
        return GetOuterIdx<IS_INTRA_BLOCK>() == 0;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool LastOuterIter() const
    {
        return GetOuterIdx<IS_INTRA_BLOCK>() + 1 == GetOuterIter<IS_INTRA_BLOCK>();
    }

    __aicore__ inline void InnerStart() {};

    __aicore__ inline bool InnerNext() {};

    __aicore__ inline bool InnerEnd() {};

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool FirstInnerIter() const
    {
        return GetInnerIdx<IS_INTRA_BLOCK>() == 0;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetTotalIter() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockKIter_;
        } else {
            return kIter_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOuterIter() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockKIter_;
        } else {
            return kIter_;
        }
    }

    __aicore__ inline uint32_t GetInnerIter() const
    {
        return 1;
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetOuterIdx() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockOuterIdx_;
        } else {
            return outerIdx_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline uint32_t GetInnerIdx() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockOuterIdx_;
        } else {
            return outerIdx_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline int32_t GetTileShapeA() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockTileShape_;
        } else {
            return tileShape_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline int32_t GetTileShapeB() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockTileShape_;
        } else {
            return tileShape_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline int32_t GetTileBlockShapeA() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockTileBlockShape_;
        } else {
            return tileBlockShape_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline int32_t GetTileBlockShapeB() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockTileBlockShape_;
        } else {
            return tileBlockShape_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline int32_t GetBaseShape() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockTileShape_;
        } else {
            return tileShape_;
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        if constexpr (IS_INTRA_BLOCK) {
            return intrablockTileBlockShape_;
        } else {
            return tileBlockShape_;
        }
    }

private:
    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void UpdateOuterParams()
    {
        if constexpr (IS_INTRA_BLOCK) {
            intrablockTileShape_ = (intrablockOuterIdx_ == intrablockKIter_ - 1) ?
                intrablockTailK_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            intrablockTileBlockShape_ = Ceil(intrablockTileShape_, c0Size_);
        } else {
            tileShape_ = (outerIdx_ == kIter_ - 1) ? tailK_ : MATMUL_MODULE(MatmulShapeTiling)->GetTiling().GetBaseK();
            tileBlockShape_ = Ceil(tileShape_, c0Size_);
        }
    }

private:
    int32_t tailK_;
    int32_t tileShape_;       // kL1 length, in normal version, kaL1 = kbL1
    int32_t tileBlockShape_;  // kL1 block num
    int32_t intrablockTailK_;
    int32_t intrablockTileShape_;       // kL1 length, in normal version, kaL1 = kbL1
    int32_t intrablockTileBlockShape_;  // kL1 block num
    uint32_t kIter_;          // total iteration counts
    uint32_t outerIdx_;       // current outer loop index
    uint32_t intrablockKIter_;          // total iteration counts
    uint32_t intrablockOuterIdx_;       // current outer loop index

    constexpr static int32_t c0Size_ = AuxGetC0Size<TRANS_T>();
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _K_LOOP_INTRA_BLOCK_H_
