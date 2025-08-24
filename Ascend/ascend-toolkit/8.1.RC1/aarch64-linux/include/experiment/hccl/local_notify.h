/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: local notify interface
 */

#ifndef LOCAL_NOTIFY_H
#define LOCAL_NOTIFY_H

#include "stream_pub.h"
#include "dispatcher.h"
#include "hccl_common.h"

namespace hccl {
class LocalNotifyImpl;
class LocalNotify {
public:
    LocalNotify();
    ~LocalNotify();
    HcclResult Init(const NotifyLoadType type = NotifyLoadType::HOST_NOTIFY);
    HcclResult Init(const HcclSignalInfo &notifyInfo,
                    const NotifyLoadType type = NotifyLoadType::DEVICE_NOTIFY);

    HcclResult Wait(Stream& stream, HcclDispatcher dispatcher,
        s32 stage = INVALID_VALUE_STAGE, u32 timeOut = NOTIFY_INVALID_WAIT_TIME);
    HcclResult Post(Stream& stream, HcclDispatcher dispatcher,
        s32 stage = INVALID_VALUE_STAGE);
    virtual HcclResult Destroy();
    virtual HcclResult SetIpc();

    // mc2获取aicpu notify信息  local&remote
    HcclResult GetNotifyData(HcclSignalInfo &notifyInfo);
    // aicpu device 侧set notify信息
    HcclResult SetNotifyData(HcclSignalInfo &notifyInfo);

    // 使用时需判空
    inline HcclRtNotify ptr()
    {
        return notifyPtr;
    }

    // 获取offset
    HcclResult GetNotifyOffset(u64 &notifyOffset);

    static HcclResult Wait(Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify> &notify,
        s32 stage = INVALID_VALUE_STAGE, u32 timeOut = NOTIFY_INVALID_WAIT_TIME);
    static HcclResult Post(Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify> &notify,
        s32 stage = INVALID_VALUE_STAGE);

    u32 notifyId_{INVALID_UINT};
protected:
    std::unique_ptr<LocalNotifyImpl> pimpl_;
    HcclRtNotify notifyPtr = nullptr;
    /* 标记notify是否是本对象申请的，如果有notify不是用户传入, 而是代码申请的,
     * 在析构时需要销毁 */
    bool notifyOwner_ = true;
};
}

#endif // LOCAL_NOTIFY_H
