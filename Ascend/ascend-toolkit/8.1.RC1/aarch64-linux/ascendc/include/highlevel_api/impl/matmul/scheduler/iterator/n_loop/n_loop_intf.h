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
* \file n_loop_intf.h
* \brief
*/

#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_INTF_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_N_LOOP_N_LOOP_INTF_H

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    NLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    NLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, const auto &MM_CFG, typename = void>
class NLoop
{
public:
    __aicore__ inline NLoop() = default;
    __aicore__ inline ~NLoop() = default;
    /**
     * @description: Init of NLoop, should be called when matmul is inited.
     * @param: singleShape: singleCoreN
     * @return: void
     */
    __aicore__ inline void Init(int32_t singleShape) {}

    /**
     * @description: Set singleShape and update params of NLoop when singleCoreN changed.
     * @param: singleShape: singleCoreN
     * @return: void
     */
    __aicore__ inline void SetSingleShape(int32_t singleShape) {}

    /**
     * @description: Get total number of iteration on N-dimension.
     * @param: void
     * @return: total number of iteration on N-dimension
     */
    __aicore__ inline uint32_t GetTotalIter() const
    {
        return 0;
    }

    /**
     * @description: N-dimension outer loop start.
     * @param: void
     * @return: void
     */
    __aicore__ inline void OuterStart() {}

    /**
     * @description: N-dimension outer loop move next.
     * @param: void
     * @return: return false when outer loop is end.
     */
    __aicore__ inline bool OuterNext()
    {
        return false;
    }

    /**
     * @description: Judge whether N-dimension outer loop is end.
     * @param: void
     * @return: return true if outer loop is end.
     */
    __aicore__ inline bool OuterEnd()
    {
        return true;
    }

    /**
     * @description: Get current outer loop index.
     * @param: void
     * @return: current outer loop index.
     */
    __aicore__ inline uint32_t GetOuterIdx() const
    {
        return 0;
    }

    /**
     * @description: Get the iteration number of outer loop.
     * @param: void
     * @return: the iteration number of outer loop.
     */
    __aicore__ inline uint32_t GetOuterIter() const
    {
        return 0;
    }

    /**
     * @description: Get matrixB shape of current outer loop.
     * @param: void
     * @return: matrixB shape of current outer loop.
     */
    __aicore__ inline int32_t GetTileShape() const
    {
        return 0;
    }

    /**
     * @description: Get matrixB block shape of current outer loop.
     * @param: void
     * @return: matrixB block shape of current outer loop.
     */
    __aicore__ inline int32_t GetTileBlockShape() const
    {
        return 0;
    }

    /**
     * @description: Get matrixB tail shape of current outer loop.
     * @param: void
     * @return: matrixB tail shape of current outer loop.
     */
    __aicore__ inline int32_t GetTailShape() const
    {
        return 0;
    }

    /**
     * @description: N-dimension inner loop start.
     * @param: void
     * @return: void
     */
    __aicore__ inline void InnerStart() {}

    /**
     * @description: N-dimension inner loop move next.
     * @param: void
     * @return: return false when inner loop is end.
     */
    __aicore__ inline bool InnerNext()
    {
        return false;
    }

    /**
     * @description: Judge whether N-dimension inner loop is end.
     * @param: void
     * @return: return true if inner loop is end.
     */
    __aicore__ inline bool InnerEnd()
    {
        return true;
    }

    /**
     * @description: Get current inner loop index.
     * @param: void
     * @return: current inner loop index.
     */
    __aicore__ inline uint32_t GetInnerIdx() const
    {
        return 0;
    }

    /**
     * @description: Get the iteration number of inner loop.
     * @param: void
     * @return: the iteration number of inner loop.
     */
    __aicore__ inline uint32_t GetInnerIter() const
    {
        return 0;
    }

    /**
     * @description: Get matrixB shape of current inner loop.
     * @param: void
     * @return: matrixB shape of current inner loop.
     */
    __aicore__ inline int32_t GetBaseShape() const
    {
        return 0;
    }

    /**
     * @description: Get matrixB block shape of current inner loop.
     * @param: void
     * @return: matrixB block shape of current inner loop.
     */
    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        return 0;
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _N_LOOP_INTF_H_
