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
 * \file matmul_tiling.h
 * \brief
 */

#ifndef LIB_MATMUL_MATMUL_TILING_H
#define LIB_MATMUL_MATMUL_TILING_H

#include <cstdint>
#include <string>
#include <array>
#include "matmul_tiling_base.h"
#include "matmul_tilingdata.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace matmul_tiling {
// single core matmul tiling
class MatmulApiTiling : public MatmulApiTilingBase {
public:
    MatmulApiTiling() {};
    explicit MatmulApiTiling(const platform_ascendc::PlatformAscendC &ascendcPlatform)
        : MatmulApiTilingBase(ascendcPlatform){};
    explicit MatmulApiTiling(const PlatformInfo& platform) : MatmulApiTilingBase(platform) {};
    ~MatmulApiTiling() override = default;
    int64_t GetTiling(optiling::TCubeTiling &tiling) override;
    int64_t GetTiling(TCubeTiling &tiling) override;

protected:
    int64_t Compute() override;
};
} // namespace matmul_tiling

extern "C" {
int32_t MatmulGetTmpBufSize(optiling::TCubeTiling &tiling, matmul_tiling::SysTilingTempBufSize &bufSize);
int32_t MatmulGetTmpBufSizeV2(TCubeTiling &tiling, matmul_tiling::SysTilingTempBufSize &bufSize);
};

#endif // MATMUL_API_TILING_H