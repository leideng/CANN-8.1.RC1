/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: HCCL的profiler虚基类头文件，用于派生task记录类
 */

#ifndef DISPATCHER_TASK_TYPES_H
#define DISPATCHER_TASK_TYPES_H

#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <thread>

#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "workflow_pub.h"

namespace hccl {
enum class TaskType {
    TASK_SDMA = 0,
    TASK_RDMA,
    TASK_REDUCE_INLINE,
    TASK_REDUCE_TBE,
    TASK_NOTIFY_RECORD,
    TASK_NOTIFY_WAIT,
    TASK_HOST,
    TASK_GRAPH_LAUNCH
};

enum class SimpleTaskType {
    SDMA = 0,
    RDMA = 1,
    LOCAL = 2,
    RESERVED = 255
};

enum class TaskRole {
    DST = 0,
    SRC = 1,
    RESERVED = 255
};

enum class LinkType {
    LINK_ONCHIP = 0,
    LINK_HCCS = 1,
    LINK_PCIE = 2,
    LINK_ROCE = 3,
    LINK_SIO = 4,
    LINK_HCCS_SW = 5,
    LINK_STANDARD_ROCE = 6,
    LINK_RESERVED = 255
};

enum class RdmaType {
    RDMA_SEND_NOTIFY = 0,
    RDMA_SEND_PAYLOAD = 1,
    RDMA_TYPE_RESERVED = 255
};

enum class ProfilerType {
    TASK_PROFILING = 0,
    TASK_EXCEPTION = 1,
    TASK_OVERFLOW,
    TASK_ALL,
    TASK_RESERVE
};

struct StepData {
    s32 streamID;
    s32 planeID; // bit[31..28] = 节点内8P-ring对应的物理环, bit[27..16] = rank_size, bit[15..0] = rank_id
    s32 stage;
    s32 step;

    StepData() : streamID(0), planeID(0), stage(0), step(-1) {}
};

struct TaskParaDMA {
    const void *src{nullptr};
    const void *dst{nullptr};
    std::size_t size{0};
    u64 notifyID{INVALID_U64};
    LinkType linkType{LinkType::LINK_ONCHIP};
    u32 remoteUserRank{INVALID_VALUE_RANKID};
    RdmaType rdmaType{RdmaType::RDMA_TYPE_RESERVED};
    u32 ctxId{INVALID_UINT}; // 子图 ctxId信息
    TaskParaDMA() {}

    TaskParaDMA(const void *inputSrc, const void *inputDst, std::size_t inputSize)
        : src(inputSrc),
          dst(inputDst),
          size(inputSize),
          notifyID(INVALID_U64),
          linkType(LinkType::LINK_ONCHIP),
          remoteUserRank(INVALID_VALUE_RANKID),
          rdmaType(RdmaType::RDMA_TYPE_RESERVED),
          ctxId(INVALID_UINT)
    {}

    TaskParaDMA(const void *inputSrc, const void *inputDst, std::size_t inputSize, u64 inputNotifyID)
        : src(inputSrc),
          dst(inputDst),
          size(inputSize),
          notifyID(inputNotifyID),
          linkType(LinkType::LINK_ONCHIP),
          remoteUserRank(INVALID_VALUE_RANKID),
          rdmaType(RdmaType::RDMA_TYPE_RESERVED),
          ctxId(INVALID_UINT)
    {}

    TaskParaDMA(const void *inputSrc, const void *inputDst, std::size_t inputSize, u64 inputNotifyID,
        LinkType inputLinkType, RdmaType inputRdmaType)
        : src(inputSrc),
          dst(inputDst),
          size(inputSize),
          notifyID(inputNotifyID),
          linkType(inputLinkType),
          remoteUserRank(static_cast<u32>(inputNotifyID >> 32)),  // notifyID的高32位为对端usrrank
          rdmaType(inputRdmaType),
          ctxId(INVALID_UINT)
    {}
    TaskParaDMA(const void *inputSrc, const void *inputDst, std::size_t inputSize, LinkType inputLinkType,
        u32 remoteUserRank) // 无notifyID时需要传入remoteUserRank
        : src(inputSrc),
          dst(inputDst),
          size(inputSize),
          notifyID(INVALID_U64),
          linkType(inputLinkType),
          remoteUserRank(remoteUserRank),
          rdmaType(RdmaType::RDMA_TYPE_RESERVED),
          ctxId(INVALID_UINT)
    {}
    TaskParaDMA(const void *inputSrc, const void *inputDst, std::size_t inputSize, u64 inputNotifyID,
        LinkType inputLinkType, RdmaType inputRdmaType, u32 inputCtxId)
        : src(inputSrc),
          dst(inputDst),
          size(inputSize),
          notifyID(inputNotifyID),
          linkType(inputLinkType),
          remoteUserRank(static_cast<u32>(inputNotifyID >> 32)),  // notifyID的高32位为对端usrrank
          rdmaType(inputRdmaType),
          ctxId(inputCtxId)
    {}
    TaskParaDMA(const void *inputSrc, const void *inputDst, std::size_t inputSize, LinkType inputLinkType,
        u32 remoteUserRank, RdmaType inputRdmaType, u32 inputCtxId)
        : src(inputSrc),
          dst(inputDst),
          size(inputSize),
          notifyID(INVALID_U64),
          linkType(inputLinkType),
          remoteUserRank(remoteUserRank),
          rdmaType(inputRdmaType),
          ctxId(inputCtxId)
    {}
};

struct TaskParaReduce {
    const void *src;
    const void *dst;
    std::size_t size;

    HcclReduceOp op;
    HcclDataType dataType;
    LinkType linkType;
    u32 remoteUserRank;
    u32 ctxId; // 子图 ctxId信息
    TaskParaReduce()
        : src(nullptr),
          dst(nullptr),
          size(0),
          op(HCCL_REDUCE_RESERVED),
          dataType(HCCL_DATA_TYPE_RESERVED),
          linkType(LinkType::LINK_ONCHIP),
          remoteUserRank(INVALID_UINT),
          ctxId(INVALID_UINT)
    {}

    TaskParaReduce(const void *inputSrc, const void *inputDst, std::size_t inputSize, HcclReduceOp inputOp,
        HcclDataType inputDataType)
        : src(inputSrc),
          dst(inputDst),
          size(inputSize),
          op(inputOp),
          dataType(inputDataType),
          linkType(LinkType::LINK_ONCHIP),
          remoteUserRank(INVALID_UINT),
          ctxId(INVALID_UINT)
    {}

    TaskParaReduce(const void *inputSrc, const void *inputDst, std::size_t inputSize, HcclReduceOp inputOp,
        HcclDataType inputDataType, LinkType inputLinkType)
        : src(inputSrc),
          dst(inputDst),
          size(inputSize),
          op(inputOp),
          dataType(inputDataType),
          linkType(inputLinkType),
          remoteUserRank(INVALID_UINT),
          ctxId(INVALID_UINT)
    {}
    TaskParaReduce(const void *inputSrc, const void *inputDst, std::size_t inputSize, HcclReduceOp inputOp,
        HcclDataType inputDataType, LinkType inputLinkType, u32 remoteUserRank)
        : src(inputSrc),
          dst(inputDst),
          size(inputSize),
          op(inputOp),
          dataType(inputDataType),
          linkType(inputLinkType),
          remoteUserRank(remoteUserRank),
          ctxId(INVALID_UINT)
    {}
    TaskParaReduce(const void *inputSrc, const void *inputDst, std::size_t inputSize, HcclReduceOp inputOp,
        HcclDataType inputDataType, LinkType inputLinkType, u32 remoteUserRank, u32 inputCtxId)
        : src(inputSrc),
          dst(inputDst),
          size(inputSize),
          op(inputOp),
          dataType(inputDataType),
          linkType(inputLinkType),
          remoteUserRank(remoteUserRank),
          ctxId(inputCtxId)
    {}
};

struct TaskParaNotify {
    u64 notifyID;
    s32 stage; // 用于标识stream间同步时所在的stage， 非用于stream同步的默认为-1
    u32 remoteUserRank;
    u32 ctxId; // 子图 ctxId信息
    TaskParaNotify() : notifyID(0), stage(INVALID_VALUE_STAGE), remoteUserRank(INVALID_UINT), ctxId(INVALID_UINT) {}
    explicit TaskParaNotify(u64 notifyIDInput)
        : notifyID(notifyIDInput),
          stage(INVALID_VALUE_STAGE),
          remoteUserRank(static_cast<u32>(notifyIDInput >> 32)), // 无remoteRank时，使用notify的高32位为remote rank
          ctxId(INVALID_UINT)
    {}
    TaskParaNotify(u64 notifyIDInput, s32 stageIn)
        : notifyID(notifyIDInput),
          stage(stageIn),
          remoteUserRank(static_cast<u32>(notifyIDInput >> 32)), // 无remoteRank时，使用notify的高32位为remote rank
          ctxId(INVALID_UINT)
    {}
    TaskParaNotify(u64 notifyIDInput, s32 stageIn, u32 remoteUserRank)
        : notifyID(notifyIDInput), stage(stageIn), remoteUserRank(remoteUserRank), ctxId(INVALID_UINT)
    {}
    TaskParaNotify(u64 notifyIDInput, s32 stageIn, u32 remoteUserRank, u32 ctxIdInput)
        : notifyID(notifyIDInput), stage(stageIn), remoteUserRank(remoteUserRank), ctxId(ctxIdInput)
    {}
};

struct TaskParaHost {
    u32 streamID;
    u32 taskID;
    u64 len;
    std::chrono::microseconds duration;
    std::string tag;
    TaskParaHost(u32 streamID, u32 taskID, u64 len, std::chrono::microseconds duration, std::string &tag)
        : streamID(streamID),
          taskID(taskID),
          len(len),
          duration(duration),
          tag(tag)
    {}
};

struct TaskParaGraphLaunch {
    u32 ctxNum{0};
};

struct TaskPara {
    TaskType type{TaskType::TASK_NOTIFY_RECORD};
    ProfilerType profilerType{ProfilerType::TASK_ALL};
    void *stream{nullptr};
    bool isMainStream{false};
    u64 beginTime{0};
    bool isFftsDispatcher{false};
    union {
        struct TaskParaDMA dma;
        struct TaskParaReduce reduce;
        struct TaskParaNotify notify;
        struct TaskParaHost host;
        struct TaskParaGraphLaunch graphLaunch;
    };

    TaskPara()
        : type(TaskType::TASK_NOTIFY_RECORD), profilerType(ProfilerType::TASK_ALL), stream(nullptr),
          isMainStream(false), beginTime(0), isFftsDispatcher(false), dma()
    {}

    TaskPara(TaskType type, TaskParaReduce reduce)
        : type(type), profilerType(ProfilerType::TASK_ALL), stream(nullptr), isMainStream(false), beginTime(0),
          isFftsDispatcher(false), reduce(reduce)
    {}

    TaskPara(TaskType type, TaskParaNotify notify)
        : type(type), profilerType(ProfilerType::TASK_ALL), stream(nullptr), isMainStream(false), beginTime(0),
          isFftsDispatcher(false), notify(notify)
    {}

    TaskPara(TaskType type, TaskParaHost host)
        : type(type), profilerType(ProfilerType::TASK_ALL), stream(nullptr), isMainStream(false), beginTime(0),
          isFftsDispatcher(false), host(host)
    {}

    TaskPara(TaskType type, TaskParaGraphLaunch graphLaunch)
        : type(type), profilerType(ProfilerType::TASK_ALL), stream(nullptr), isMainStream(false), beginTime(0),
          isFftsDispatcher(false), graphLaunch(graphLaunch)
    {}

    ~TaskPara() {}
};

} // namespace hccl

#endif /* DISPATCHER_TASK_TYPES_H */
