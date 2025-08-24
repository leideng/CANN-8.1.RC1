/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of sample
 */

#ifndef _AICPU_RESHAPE_CUST_KERNELS_H_
#define _AICPU_RESHAPE_CUST_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class ReshapeCustCpuKernel : public CpuKernel {
public:
    ~ReshapeCustCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;
};
} // namespace aicpu
#endif
