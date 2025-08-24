/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: remote notify interface
 */

#ifndef REMOTE_NOTIFY_H
#define REMOTE_NOTIFY_H

#include "stream_pub.h"
#include "dispatcher.h"

namespace hccl {
class RemoteNotifyImpl;
class RemoteNotify {
public:
    RemoteNotify();
    ~RemoteNotify();
    HcclResult Init(const std::vector<u8>& byteVector);
    HcclResult Init(const HcclSignalInfo &notifyInfo,
                    const NotifyLoadType type = NotifyLoadType::DEVICE_NOTIFY);

    HcclResult Open();
    HcclResult Close();
    HcclResult Post(Stream& stream, HcclDispatcher dispatcher, s32 stage = INVALID_VALUE_STAGE);
    HcclResult GetNotifyData(HcclSignalInfo &notifyInfo);
    HcclResult SetNotifyData(HcclSignalInfo &notifyInfo);

    // 仅限dispatcher使用，使用时需判空
    inline HcclRtNotify ptr()
    {
        return notifyPtr;
    }
    // 获取offset
    HcclResult GetNotifyOffset(u64 &notifyOffset);

private:
    std::unique_ptr<RemoteNotifyImpl> pimpl_;
    HcclRtNotify notifyPtr;
};

}

#endif // REMOTE_NOTIFY_H
