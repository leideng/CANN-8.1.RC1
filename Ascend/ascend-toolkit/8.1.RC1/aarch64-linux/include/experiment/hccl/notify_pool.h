/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: local notify pool interface
 */

#ifndef NOTIFY_POOL_H
#define NOTIFY_POOL_H


#include "dispatcher.h"

namespace hccl {
class NotifyPoolImpl;
class LocalIpcNotify ;
class NotifyPool {
public:
    NotifyPool();
    ~NotifyPool();
    HcclResult Init(const s32 devicePhyId);
    HcclResult Destroy();
    HcclResult RegisterOp(const std::string &tag);
    HcclResult UnregisterOp(const std::string &tag);
    // local notify申请
    HcclResult Alloc(const std::string &tag, const RemoteRankInfo &info,
        std::shared_ptr<LocalIpcNotify> &localNotify, const NotifyLoadType type = NotifyLoadType::HOST_NOTIFY);
    HcclResult ResetNotify();
    HcclResult ResetNotifyForDestRank(s64 destRank);

protected:
    std::unique_ptr<NotifyPoolImpl> pimpl_;
};
}
#endif //  NOTIFY_POOL_H