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
* \file copy_tile_to_cube_intf.h
* \brief
*/

#ifndef IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_TILE_TO_CUBE_INTF_H
#define IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_TILE_TO_CUBE_INTF_H

#include "../../../utils/matmul_module.h"
#include "../../../utils/matmul_param.h"
#include "../base/copy_cube_in_utils.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE, typename = void>
class CopyTileToCubeWrapper {
    using TransT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline CopyTileToCubeWrapper() = default;
    __aicore__ inline ~CopyTileToCubeWrapper() = default;

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyTileToCube(const LocalTensor<TransT>& dst, int32_t curRow, int32_t curCol,
        int32_t tileHeight, int32_t tileWidth) {}
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_TILE_TO_CUBE_INTF_H
