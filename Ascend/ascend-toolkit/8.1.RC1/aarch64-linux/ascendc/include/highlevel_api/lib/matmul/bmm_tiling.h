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
 * \file bmm_tiling.h
 * \brief
 */
#ifndef LIB_MATMUL_BMM_TILING_H
#define LIB_MATMUL_BMM_TILING_H

#include "matmul_tiling_base.h"
#include "matmul_tilingdata.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace matmul_tiling {
// matmu tiling for multi core
class MultiCoreMatmulTiling : public MatmulApiTilingBase {
public:
    MultiCoreMatmulTiling() {};
    explicit MultiCoreMatmulTiling(const platform_ascendc::PlatformAscendC &ascendcPlatform)
        : MatmulApiTilingBase(ascendcPlatform){};
    explicit MultiCoreMatmulTiling(const PlatformInfo &platform)
        : MatmulApiTilingBase(platform) {};
    int32_t SetDim(int32_t dim);                               // Sets the allowed block dim.
    int32_t SetShape(int32_t m, int32_t n, int32_t k) override;  // Sets the size of the original input
    virtual int32_t SetSingleShape(int32_t singleMIn = -1, int32_t singleNIn = -1, int32_t singleKIn = -1);
    int64_t GetTiling(optiling::TCubeTiling &tiling) override;
    int64_t GetTiling(TCubeTiling &tiling) override;
    virtual int32_t SetSingleRange(int32_t maxM = -1, int32_t maxN = -1, int32_t maxK = -1,
        int32_t minM = -1, int32_t minN = -1, int32_t minK = -1)
    {
        this->maxSingleM = maxM;
        this->maxSingleN = maxN;
        this->maxSingleK = maxK;
        this->minSingleM = minM;
        this->minSingleN = minN;
        this->minSingleK = minK;
        return 0;
    }
    int32_t SetAlignSplit(int32_t alignM, int32_t alignN, int32_t alignK);
    // Get the amount of data processed at a time.
    int32_t GetSingleShape(int32_t &shapeM, int32_t &shapeN, int32_t &shapeK);
    // Get the BlockDim used after multi core tiling.
    // It is carried by users to the kernel to control the service logic in the kernel.
    int32_t GetCoreNum(int32_t &dim, int32_t &mDim, int32_t &nDim);
    void EnableMultiCoreSplitK(bool flag)
    {
        enableSplitK_ = flag;
    }
    void SetSplitK(bool flag)
    {
        EnableMultiCoreSplitK(flag);
    }
protected:
    int64_t Compute() override;
};

// batch matul tiling
class BatchMatmulTiling : public MatmulApiTilingBase {
public:
    BatchMatmulTiling() {};
    explicit BatchMatmulTiling(const platform_ascendc::PlatformAscendC &ascendcPlatform)
        : MatmulApiTilingBase(ascendcPlatform){};
    int32_t GetCoreNum(int32_t &dim, int32_t &mDim, int32_t &nDim, int32_t &batchCoreM, int32_t &batchCoreN);
    int64_t GetTiling(optiling::TCubeTiling &tiling) override;
    int64_t GetTiling(TCubeTiling &tiling) override;
protected:
    int64_t Compute() override;
private:
    int32_t SetBatch(int32_t batchMIn = 1, int32_t batchNIn = 1);
    // Set the batch axis tiling mode.
    int32_t SetSingleBatch(int32_t singleMIn = -1, int32_t singleNIn = -1);
};
} // namespace matmul_tiling

extern "C" {
int32_t MultiCoreMatmulGetTmpBufSize(optiling::TCubeTiling &tiling, matmul_tiling::SysTilingTempBufSize &bufSize);
int32_t BatchMatmulGetTmpBufSize(optiling::TCubeTiling &tiling, matmul_tiling::SysTilingTempBufSize &bufSize);
int32_t MultiCoreMatmulGetTmpBufSizeV2(TCubeTiling &tiling, matmul_tiling::SysTilingTempBufSize &bufSize);
int32_t BatchMatmulGetTmpBufSizeV2(TCubeTiling &tiling, matmul_tiling::SysTilingTempBufSize &bufSize);
};

#endif // LIB_MATMUL_BMM_TILING_H
