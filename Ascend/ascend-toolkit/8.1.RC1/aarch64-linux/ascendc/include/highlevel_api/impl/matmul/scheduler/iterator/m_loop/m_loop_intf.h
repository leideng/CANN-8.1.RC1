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
* \file m_loop_intf.h
* \brief
*/

#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_INTF_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_M_LOOP_M_LOOP_INTF_H

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    MLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    MLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, const auto &MM_CFG, typename = void>
class MLoop
{
public:
    __aicore__ inline MLoop() = default;
    __aicore__ inline ~MLoop() = default;
    /**
     * @description: Init of MLoop, should be called when matmul is inited.
     * @param: singleShape: singleCoreM
     * @return: void
     */
    __aicore__ inline void Init(int32_t singleShape) {}

    /**
     * @description: Set singleShape and update params of MLoop when singleCoreM changed.
     * @param: singleShape: singleCoreM
     * @return: void
     */
    __aicore__ inline void SetSingleShape(int32_t singleShape) {}

    /**
     * @description: Get total number of iteration on M-dimension.
     * @param: void
     * @return: total number of iteration on M-dimension
     */
    __aicore__ inline uint32_t GetTotalIter() const
    {
        return 0;
    }

    /**
     * @description: M-dimension outer loop start.
     * @param: void
     * @return: void
     */
    __aicore__ inline void OuterStart() {}

    /**
     * @description: M-dimension outer loop move next.
     * @param: void
     * @return: return false when outer loop is end.
     */
    __aicore__ inline bool OuterNext()
    {
        return false;
    }

    /**
     * @description: Judge whether M-dimension outer loop is end.
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
     * @description: Get matrixA shape of current outer loop.
     * @param: void
     * @return: matrixA shape of current outer loop.
     */
    __aicore__ inline int32_t GetTileShape() const
    {
        return 0;
    }

    /**
     * @description: Get matrixA block shape of current outer loop.
     * @param: void
     * @return: matrixA block shape of current outer loop.
     */
    __aicore__ inline int32_t GetTileBlockShape() const
    {
        return 0;
    }

    /**
     * @description: Get matrixA tail shape of current outer loop.
     * @param: void
     * @return: matrixA tail shape of current outer loop.
     */
    __aicore__ inline int32_t GetTailShape() const
    {
        return 0;
    }

    /**
     * @description: M-dimension inner loop start.
     * @param: void
     * @return: void
     */
    __aicore__ inline void InnerStart() {}

    /**
     * @description: M-dimension inner loop move next.
     * @param: void
     * @return: return false when inner loop is end.
     */
    __aicore__ inline bool InnerNext()
    {
        return false;
    }

    /**
     * @description: Judge whether M-dimension inner loop is end.
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
     * @description: Get matrixA shape of current inner loop.
     * @param: void
     * @return: matrixA shape of current inner loop.
     */
    __aicore__ inline int32_t GetBaseShape() const
    {
        return 0;
    }

    /**
     * @description: Get matrixA block shape of current inner loop.
     * @param: void
     * @return: matrixA block shape of current inner loop.
     */
    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        return 0;
    }
};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _M_LOOP_INTF_H_
