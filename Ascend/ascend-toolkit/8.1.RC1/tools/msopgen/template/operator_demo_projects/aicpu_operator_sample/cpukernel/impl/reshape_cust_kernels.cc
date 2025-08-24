/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of sample
 */

#include "reshape_cust_kernels.h"
#include <cstring>
#include <cstdint>
#include "cpu_types.h"

namespace {
const char *RESHAPE_CUST = "ReshapeCust";
}

namespace aicpu {
uint32_t ReshapeCustCpuKernel::Compute(CpuKernelContext &ctx)
{
    Tensor *inputTensor = ctx.Input(0);
    if (inputTensor == nullptr) {
        return -1;
    }

    Tensor *outputTensor = ctx.Output(0);
    if (outputTensor == nullptr) {
        return -1;
    }
    auto inputData = inputTensor->GetData();
    if (inputData == nullptr) {
        return -1;
    }

    auto outputData = outputTensor->GetData();
    if (outputData == nullptr) {
        return -1;
    }

    uint64_t inputDataSize = inputTensor->GetDataSize();
    memcpy(outputData, inputData, inputDataSize);

    return 0;
}

REGISTER_CPU_KERNEL(RESHAPE_CUST, ReshapeCustCpuKernel);
} // namespace aicpu
