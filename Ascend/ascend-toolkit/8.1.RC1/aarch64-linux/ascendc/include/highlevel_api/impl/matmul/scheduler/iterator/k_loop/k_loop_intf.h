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
* \file k_loop_intf.h
* \brief
*/
#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_INTF_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_INTF_H

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    KLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    KLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG, typename = void>
class KLoop {
public:
    __aicore__ inline KLoop() = default;
    __aicore__ inline ~KLoop() = default;

    /**
     * @description: Init KLoop params, should be called when matmul is inited.
     * @param: singleShape: singleCoreK
     * @return: void
     */
    __aicore__ inline void Init(int32_t singleShape) {}

    /**
     * @description: Update KLoop params, should be called when singleCoreK is update.
     * @param: singleShape: singleCoreK
     * @return: void
     */
    __aicore__ inline void SetSingleShape(int32_t singleShape) {}

    /**
     * @description: Start outer loop for KLoop
     * @param: void
     * @return: void
     */
    __aicore__ inline void OuterStart() {}

    /**
     * @description: Proceed outer loop for KLoop
     * @param: void
     * @return: return false if outer loop is end
     */
    __aicore__ inline bool OuterNext()
    {
        return false;
    }

    /**
     * @description: Judge if outer loop is end
     * @param: void
     * @return: return true if outer loop is end
     */
    __aicore__ inline bool OuterEnd()
    {
        return true;
    }

    /**
     * @description: Judge if first outer loop
     * @param: void
     * @return: return true if first outer loop
     */
    __aicore__ inline bool FirstOuterIter() const
    {
        return true;
    }

    /**
     * @description: Judge if last outer loop
     * @param: void
     * @return: return true if last outer loop
     */
    __aicore__ inline bool LastOuterIter() const
    {
        return true;
    }

    /**
     * @description: Start inner loop for KLoop
     * @param: void
     * @return: void
     */
    __aicore__ inline void InnerStart() {}

    /**
     * @description: Proceed inner loop for KLoop
     * @param: void
     * @return: return false if inner loop is end
     */
    __aicore__ inline bool InnerNext()
    {
        return false;
    }

    /**
     * @description: Judge if inner loop is end
     * @param: void
     * @return: return true if inner loop is end
     */
    __aicore__ inline bool InnerEnd()
    {
        return true;
    }

    /**
     * @description: Judge if first inner loop
     * @param: void
     * @return: return true if first inner loop
     */
    __aicore__ inline bool FirstInnerIter() const
    {
        return true;
    }

    /**
     * @description: Get total iteration counts for KLoop
     * @param: void
     * @return: return total iteration nums
     */
    __aicore__ inline uint32_t GetTotalIter() const
    {
        return 0;
    }

    /**
     * @description: Get outer loop counts for KLoop
     * @param: void
     * @return: return outer loop counts
     */
    __aicore__ inline uint32_t GetOuterIter() const
    {
        return 0;
    }

    /**
     * @description: Get inner loop counts for KLoop
     * @param: void
     * @return: return inner loop counts
     */
    __aicore__ inline uint32_t GetInnerIter() const
    {
        return 0;
    }

    /**
     * @description: Get outer loop index for KLoop
     * @param: void
     * @return: return current outer loop index
     */
        __aicore__ inline uint32_t GetOuterIdx() const
    {
        return 0;
    }

    /**
     * @description: Get inner loop index for KLoop
     * @param: void
     * @return: return current inner loop index
     */
    __aicore__ inline uint32_t GetInnerIdx() const
    {
        return 0;
    }

    /**
     * @description: Get Ka L1 length for current outer loop
     * @param: void
     * @return: return current Ka length
     */
    __aicore__ inline int32_t GetTileShapeA() const
    {
        return 0;
    }

    /**
     * @description: Get Kb L1 length for current outer loop
     * @param: void
     * @return: return current Kb length
     */
    __aicore__ inline int32_t GetTileShapeB() const
    {
        return 0;
    }

    /**
     * @description: Get Ka block number for current outer loop
     * @param: void
     * @return: return current Ka block number
     */
    __aicore__ inline int32_t GetTileBlockShapeA() const
    {
        return 0;
    }

    /**
     * @description: Get Kb block number for current outer loop
     * @param: void
     * @return: return current Kb block number
     */
    __aicore__ inline int32_t GetTileBlockShapeB() const
    {
        return 0;
    }

    /**
     * @description: Get K L0 length for current inner loop
     * @param: void
     * @return: return current K length
     */
    __aicore__ inline int32_t GetBaseShape() const
    {
        return 0;
    }

    /**
     * @description: Get K block number for current inner loop
     * @param: void
     * @return: return current K block number
     */
    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        return 0;
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _K_LOOP_INTF_H_