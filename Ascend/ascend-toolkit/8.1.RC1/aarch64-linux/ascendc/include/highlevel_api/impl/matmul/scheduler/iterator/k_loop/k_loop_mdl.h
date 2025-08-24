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
* \file k_loop_mdl.h
* \brief
*/
#ifndef IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_MDL_H
#define IMPL_MATMUL_SCHEDULER_ITERATOR_K_LOOP_K_LOOP_MDL_H

#include "k_loop_intf.h"
#include "k_loop_mdl_base.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    KLoop is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    KLoop is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG>
class KLoop<IMPL, TRANS_T, A_TYPE, MM_CFG,
            enable_if_t<(DoMatmulMDL(MM_CFG) && !IsKdimReorderLoad<MM_CFG>) || DoMatmulSpecialMDL(MM_CFG)>>
    : public KLoopMDLBase<IMPL, TRANS_T, A_TYPE, MM_CFG>
{
    MATMUL_USE_MODULE(MatmulShapeTiling);

public:
    using BASE_MODULE = AscendC::Impl::Detail::KLoopMDLBase<IMPL, TRANS_T, A_TYPE, MM_CFG>;
    __aicore__ inline KLoop() = default;
    __aicore__ inline ~KLoop() = default;

};

}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _K_LOOP_MDL_H_
