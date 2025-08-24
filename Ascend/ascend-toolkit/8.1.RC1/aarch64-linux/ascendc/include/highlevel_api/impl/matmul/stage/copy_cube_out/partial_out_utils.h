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
 * \file partial_out_utils.h
 * \brief
 */

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_OUT_PARTIAL_OUTPUT_PARTIAL_OUT_UTILS_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_OUT_PARTIAL_OUTPUT_PARTIAL_OUT_UTILS_H

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    PartialOutUtils is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    PartialOutUtils is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, const auto &MM_CFG, typename = void>
class PartialOutUtils
{
public:
    /**
     * @description: SetWorkspace for cacheworkspace
     * @return: void
     */
    template <class T>
    __aicore__ inline void SetWorkspace(__gm__ const T* addr) {}
    /**
     * @description: Copy data from L0C to gm
     * @return: void
     */
    __aicore__ inline void CopyOut() {}

    /**
     * @description: Destroy PartialOutUtils
     * @return: void
     */
    __aicore__ inline void Destroy() {}
};

template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, const auto &MM_CFG>
class PartialOutUtils<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG, enable_if_t<ToMatmulConfig(MM_CFG).isPartialOutput>>
{
    MATMUL_USE_MODULE(MLoop);
    MATMUL_USE_MODULE(NLoop);
    MATMUL_USE_MODULE(CubeOutBuffer);
    MATMUL_USE_MODULE(CopyCubeOut);

    using DstT = typename C_TYPE::T;

public:
    template <class T>
    __aicore__ inline void SetWorkspace(__gm__ const T* addr)
    {
        ASCENDC_ASSERT((addr != nullptr), { KERNEL_LOG(KERNEL_ERROR, "addr can not be nullptr"); });
        cacheWorkspaceAddr_ = reinterpret_cast<GM_ADDR>(const_cast<__gm__ T*>(addr));
    }
    /**
     * @description: Copy data from L0C to gm
     * @return: void
     */
    __aicore__ inline void CopyOut()
    {
        GlobalTensor<DstT> cacheWorkspace;
        cacheWorkspace.SetGlobalBuffer(
            reinterpret_cast<__gm__ DstT*>(cacheWorkspaceAddr_));
        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        MATMUL_MODULE(CubeOutBuffer)->EnQue(co1Local);
        MATMUL_MODULE(CubeOutBuffer)->DeQue();
        MATMUL_MODULE(CopyCubeOut)->template Copy<true>(cacheWorkspace[l0cOffsetSize_], co1Local,
            MATMUL_MODULE(MLoop)->GetInnerIdx(), MATMUL_MODULE(NLoop)->GetInnerIdx(),
            MATMUL_MODULE(MLoop)->GetBaseShape(), MATMUL_MODULE(NLoop)->GetBaseShape(),
            MATMUL_MODULE(MLoop)->GetBaseBlockShape(), MATMUL_MODULE(NLoop)->GetBaseBlockShape());
        event_t eventIDFIXToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_M));
        SetFlag<HardEvent::FIX_M>(eventIDFIXToM);
        WaitFlag<HardEvent::FIX_M>(eventIDFIXToM);
        l0cOffsetSize_ += MATMUL_MODULE(MLoop)->GetBaseShape() * MATMUL_MODULE(NLoop)->GetBaseShape();
    }

    __aicore__ inline void Destroy()
    {
        auto co1Local = MATMUL_MODULE(CubeOutBuffer)->GetTensor();
        MATMUL_MODULE(CubeOutBuffer)->FreeTensor(co1Local);
    }

private:
    uint64_t l0cOffsetSize_ { 0 };
    __gm__ uint8_t* cacheWorkspaceAddr_;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_OUT_PARTIAL_OUTPUT_PARTIAL_OUT_UTILS_H
