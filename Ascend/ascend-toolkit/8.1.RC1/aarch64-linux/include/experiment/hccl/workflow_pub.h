/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2023. All rights reserved.
 * Description: Workflow 模式配置管理
 */

#ifndef WORKFLOW_PUB_H
#define WORKFLOW_PUB_H

#include <hccl/hccl_types.h>

enum class HcclWorkflowMode {
    HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB = 0,
    HCCL_WORKFLOW_MODE_OP_BASE = 1,
    HCCL_WORKFLOW_MODE_RESERVED = 255
};


HcclResult InitWorkflowMode(HcclWorkflowMode mode);
HcclResult SetWorkflowMode(HcclWorkflowMode mode);
HcclWorkflowMode GetWorkflowMode();
void SetLaunchKernelMode(bool state);
bool IsLaunchKernelMode(void);
void SetTaskNumCalMode(bool state);
bool IsTaskNumCalMode(void);
#endif // WORKFLOW_PUB_H