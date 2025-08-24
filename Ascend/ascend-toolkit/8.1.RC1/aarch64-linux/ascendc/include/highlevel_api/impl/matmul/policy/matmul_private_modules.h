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
* \file matmul_private_modules.h
* \brief
*/
#ifndef IMPL_MATMUL_POLICY_MATMUL_PRIVATE_MODULES_H
#define IMPL_MATMUL_POLICY_MATMUL_PRIVATE_MODULES_H

#include "../param/matmul_shape_info.h"
#include "../param/matmul_shape_tiling.h"
#include "../param/matmul_subblock_info.h"
#include "../param/matmul_tensor_info.h"
#include "../param/matmul_unit_flag.h"
#include "../param/matmul_usr_define_info.h"
#include "../resource/bias_buffer/c1_buffer/c1_buffer.h"
#include "../resource/bias_buffer/c2_buffer/c2_buffer.h"
#include "../resource/l0_buffer/tbuf_pool_l0.h"
#include "../resource/local_workspace/matmul_local_workspace.h"
#include "../scheduler/iterator/batch_loop/batch_loop.h"
#include "../scheduler/iterator/m_loop/m_loop.h"
#include "../scheduler/iterator/n_loop/n_loop.h"
#include "../scheduler/iterator/k_loop/k_loop.h"
#include "../stage/compute/mmad_compute.h"
#include "../stage/copy_cube_in/antiquant/antiquant_processor.h"
#include "../stage/copy_cube_in/batch/batch_copy_cube_in_intf.h"
#include "../stage/copy_cube_in/batch/batch_copy_cube_in_params.h"
#include "../stage/copy_cube_in/bias/copy_bias_in.h"
#include "../stage/copy_cube_in/copy_tile_to_cube/copy_tile_to_cube.h"
#include "../stage/copy_cube_in/copy_tile_to_cube/data_copy_wrapper.h"
#include "../stage/copy_cube_in/base/copy_cube_in_params.h"
#include "../stage/copy_cube_in/base/copy_cube_in_using_ub.h"
#include "../stage/copy_cube_out/copy_cube_out_datacopy_wrapper.h"
#include "../stage/copy_cube_out/partial_out_utils.h"
#include "../stage/copy_cube_out/quant/quant_processor.h"
#include "../stage/split/load_to_l0a/load_to_l0a.h"
#include "../stage/split/load_to_l0b/load_to_l0b.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
struct MatmulPrivateModules {
    using CopyCubeInParamsA = CopyCubeInParams<IMPL, MM_CFG, MatmulInputAType<A_TYPE, typename A_TYPE::T>>;
    using CopyCubeInParamsB = CopyCubeInParams<IMPL, MM_CFG, MatmulInputBType<B_TYPE, typename A_TYPE::T>>;
    using MatmulTensorInfoA = MatmulTensorInfo<IMPL, MM_CFG, MatmulInputAType<A_TYPE, typename A_TYPE::T>>;
    using MatmulTensorInfoB = MatmulTensorInfo<IMPL, MM_CFG, MatmulInputBType<B_TYPE, typename A_TYPE::T>>;
    using MatmulSubBlockInfo = AscendC::Impl::Detail::MatmulSubBlockInfo<IMPL, MM_CFG>;
    using MatmulShapeTiling = AscendC::Impl::Detail::MatmulShapeTiling<IMPL, MM_CFG>;
    using DataCopyUtilsA = CopyTileToCubeWrapper<IMPL, MM_CFG, MatmulInputAType<A_TYPE, typename A_TYPE::T>>;
    using DataCopyUtilsB = CopyTileToCubeWrapper<IMPL, MM_CFG, MatmulInputBType<B_TYPE, typename A_TYPE::T>>;
    using DataCopyWrapperA = DataCopyWrapper<IMPL, MM_CFG, MatmulInputAType<A_TYPE, typename A_TYPE::T>>;
    using DataCopyWrapperB = DataCopyWrapper<IMPL, MM_CFG, MatmulInputBType<B_TYPE, typename A_TYPE::T>>;
    using BatchCopyCubeInParamsA = BatchCopyCubeInParams<IMPL, MM_CFG, MatmulInputAType<A_TYPE, typename A_TYPE::T>>;
    using BatchCopyCubeInParamsB = BatchCopyCubeInParams<IMPL, MM_CFG, MatmulInputBType<B_TYPE, typename A_TYPE::T>>;
    using BatchCopyCubeInA = BatchCopyCubeIn<IMPL, MM_CFG, MatmulInputAType<A_TYPE, typename A_TYPE::T>>;
    using BatchCopyCubeInB = BatchCopyCubeIn<IMPL, MM_CFG, MatmulInputBType<B_TYPE, typename A_TYPE::T>>;
    using LocalWorkspace = MatmulLocalWorkspace<IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG>;
    using MatmulShapeInfo = AscendC::Impl::Detail::MatmulShapeInfo<IMPL, A_TYPE, MM_CFG>;
    using MatmulQuantProcessor = AscendC::Impl::Detail::MatmulQuantProcessor<IMPL, A_TYPE, C_TYPE, MM_CFG>;
    using MatmulAntiQuantProcessor = AscendC::Impl::Detail::MatmulAntiQuantProcessor<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    using MatmulUserDefineInfo = AscendC::Impl::Detail::MatmulUserDefineInfo<IMPL, MM_CFG>;
    using MatmulUnitFlag = AscendC::Impl::Detail::MatmulUnitFlag<IMPL, MM_CFG>;
    using BatchLoop = AscendC::Impl::Detail::BatchLoop<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeOutUtils = AscendC::Impl::Detail::CopyCubeOutWrapper<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG>;
    using PartialOutUtils = AscendC::Impl::Detail::PartialOutUtils<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG>;

    // using compute modules
    using L0cT = typename GetDstType<typename A_TYPE::T>::Type;
    using LoadToA2 = LoadToL0A<IMPL, A_TYPE, MM_CFG>;
    using LoadToB2 = LoadToL0B<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using TBufPoolL0 = AscendC::Impl::Detail::TBufPoolL0<IMPL, A_TYPE, B_TYPE, MM_CFG>;
    using MmadCompute = AscendC::Impl::Detail::MmadCompute<IMPL, L0cT, typename A_TYPE::T, typename A_TYPE::T, MM_CFG>;

    // using Bias modules
    using CopyBiasIn = AscendC::Impl::Detail::CopyBiasIn<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;
    using C1Buffer = AscendC::Impl::Detail::C1Buffer<IMPL, BIAS_TYPE, A_TYPE, MM_CFG>;
    using C2Buffer = AscendC::Impl::Detail::C2Buffer<IMPL, L0cT, A_TYPE, MM_CFG>;
    using LoadBias2C2 = AscendC::Impl::Detail::LoadBias2C2<IMPL, A_TYPE, BIAS_TYPE, MM_CFG>;

    // using MLoop, NLoop, KLoop modules
    using MLoop = AscendC::Impl::Detail::MLoop<IMPL, A_TYPE, MM_CFG>;
    using NLoop = AscendC::Impl::Detail::NLoop<IMPL, A_TYPE, MM_CFG>;
    using KLoop = AscendC::Impl::Detail::KLoop<IMPL, typename A_TYPE::T, A_TYPE, MM_CFG>;
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // _MATMUL_PRIVATE_MODULES_H_
