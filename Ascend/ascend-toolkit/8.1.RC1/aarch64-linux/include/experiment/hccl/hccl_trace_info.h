/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: trace 管理类定义
 * Author: wuhaoyu
 * Create: 2025-02-10
 */

#ifndef HCCL_TRACE_INFO_H
#define HCCL_TRACE_INFO_H

#include "hccl_common.h"
#include "hccl/base.h"
#include <string>
namespace hccl {
enum class AtraceOption {
    Opbasekey,
    Algtype
};

class HcclTraceInfo {
public:
    enum class HcclTraceType {
        HostTraceType,
        DeviceTraceType
    };

    struct UtraceAttr {
        bool utraceStatusFlag;
        u32 pid;
        u32 deviceid;
    };

    HcclTraceInfo();
    HcclTraceInfo(const UtraceAttr &utraceAttr);
    ~HcclTraceInfo();
    HcclResult Init(std::string &logInfo);
    void DeInit();
    HcclResult Flush();
    HcclResult SaveTraceInfo(std::string &logInfo, AtraceOption op);
    HcclResult SavealgtypeTraceInfo(std::string &algtype, const std::string &tag);
    HcclTraceType hcclTraceType_ = HcclTraceType::HostTraceType;
    UtraceAttr utraceAttr_{0};
    uint32_t index{0};
    HcclTraHandle handle{0};
};
}
#endif // HCCL_TRACE_INFO_H