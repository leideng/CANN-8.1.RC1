/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: TransportPub 头文件
 */

#ifndef TRANSPORT_PUB_H
#define TRANSPORT_PUB_H

#include <initializer_list>
#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "sal_pub.h"
#include "adapter_pub.h"
#include "stream_pub.h"
#include "dispatcher.h"
#include "hccl_socket.h"
#include "notify_pool.h"
#include "local_notify.h"
#include "remote_notify.h"

enum class DBMode : s32 {
    INVALID_DB = -1,
    HW_DB = 0,
    SW_DB
};
 
struct HcclAiRMAWQ {
    u32 wqn;
    u64 bufAddr;
    u32 wqeSize;
    u32 depth;
    u64 headAddr;
    u64 tailAddr;
    DBMode dbMode; // 0-hw/1-sw
    u64 dbAddr;
    u32 sl;
    HcclAiRMAWQ() : wqn(0), bufAddr(0), wqeSize(0), depth(0), headAddr(0), tailAddr(0), dbMode(DBMode::INVALID_DB), dbAddr(0), sl(0) {}
};
 
struct HcclAiRMACQ {
    u32 cqn;
    u64 bufAddr;
    u32 cqeSize;
    u32 depth;
    u64 headAddr;
    u64 tailAddr;
    DBMode dbMode; // 0-hw/1-sw
    u64 dbAddr;
    HcclAiRMACQ() : cqn(0), bufAddr(0), cqeSize(0), depth(0), headAddr(0), tailAddr(0), dbMode(DBMode::INVALID_DB), dbAddr(0) {}
};
 
struct HcclAiRMAQueueInfo {
    struct HcclAiRMAWQ sq;
    struct HcclAiRMAWQ rq;
    struct HcclAiRMACQ scq;
    struct HcclAiRMACQ rcq;
};

struct HcclQpInfoV2 {
    u64 qpPtr;
    u32 sqIndex;
    u32 dbIndex;

    HcclQpInfoV2() : qpPtr(0), sqIndex(0), dbIndex(0)
    {}
    HcclQpInfoV2(const HcclQpInfoV2 &other) : qpPtr(other.qpPtr), sqIndex(other.sqIndex), dbIndex(other.dbIndex)
    {}
    HcclQpInfoV2(HcclQpInfoV2 &&other) : qpPtr(other.qpPtr), sqIndex(other.sqIndex), dbIndex(other.dbIndex)
    {}
    HcclQpInfoV2 &operator=(const HcclQpInfoV2 &other)
    {
        if(&other != this) {
            qpPtr = other.qpPtr;
            sqIndex = other.sqIndex;
            dbIndex = other.dbIndex;
        }
        return *this;
    }
    HcclQpInfoV2 &operator=(HcclQpInfoV2 &&other)
    {
        if(&other != this) {
            qpPtr = other.qpPtr;
            sqIndex = other.sqIndex;
            dbIndex = other.dbIndex;
        }
        return *this;
    }
};

struct AddrKey {
    u64 addr;
    u32 key;
};

struct MemDetails {
    u64 size = 0;
    u64 addr = 0;
    u32 key = 0;

    MemDetails()
    {}
    MemDetails(const MemDetails &that) : size(that.size), addr(that.addr),
        key(that.key)
    {}

    MemDetails(MemDetails &&that) : size(that.size), addr(that.addr),
        key(that.key)
    {}
    MemDetails &operator=(const MemDetails &that)
    {
        if (&that != this) {
            size = that.size;
            addr = that.addr;
            key = that.key;
        }
        return *this;
    }

    MemDetails operator=(MemDetails &&that)
    {
        if (&that != this) {
            size = that.size;
            addr = that.addr;
            key = that.key;
        }
        return *this;
    }
};

namespace hccl {

class TransportBase;
struct RxMemoryInfo;
struct TxMemoryInfo;
enum class UserMemType;
class DeviceMem;
class MemNameRepository;

enum class MachineType {
    MACHINE_SERVER_TYPE,
    MACHINE_CLIENT_TYPE,
    MACHINE_RESERVED_TYPE
};
enum class LinkMode {
    LINK_SIMPLEX_MODE,
    LINK_DUPLEX_MODE,
    LINK_RESERVED_MODE
};

// signal record 使用的value的内存信息
// 使用SDMDA或者RDMA进行notify record时需要将该内存copy到远端 notify 寄存器
using HcclSignalRecordBuff = struct HcclSignalRecordBuffDef {
    u64 address{0};     //  signal 地址
    u64 length{0};      //  signal 长度
};

constexpr u32 HCCL_TRANSPORT_RELATIONSHIP_SAME_CHIP = 0x1U << 0;        // transport 的两端rank位于同一个NPU芯片内
constexpr u32 HCCL_TRANSPORT_RELATIONSHIP_SAME_SERVER = 0x1U << 1;      // transport 的两端rank位于同一个服务器内
constexpr u32 HCCL_TRANSPORT_RELATIONSHIP_SAME_SUPERPOD = 0x1U << 2;    // transport 的两端rank位于同一个超节点内

// transport 使用的基础信息，包括链路类型、和远端的位置关系等
using TransportAttr = struct TransportAttrDef {
    hccl::LinkType linkType{hccl::LinkType::LINK_RESERVED};  // 链路类型，HCCS,
    u32 relationship{0}; // 和remote的位置关系{同芯片，同节点，跨节点}
    HcclSignalRecordBuff signalRecordBuff;
};

constexpr u64 MAX_EXCHANGE_DATA_LEN = 2ULL * 1024 * 1024; // 自定义交换数据限制2MB

// 传参数的时候都填充，link自己使用的时候区分
// RDMA:  machineType+serverId+local_rank_id+remote_rank_id+collectiveId
// TCP:   machineType+serverId+local_rank_id+remote_rank_id+collectiveId
// PCIE:  local_rank_id+remote_rank_id+collectiveId+localDeviceId+remoteDeviceId
using MachinePara = struct TagMachinePara {
public:
    MachineType machineType{MachineType::MACHINE_RESERVED_TYPE};  // client或者server
    LinkMode linkMode{LinkMode::LINK_RESERVED_MODE};
    std::string collectiveId{""};  // 本节点所在的通信域ID
    std::string tag{""};
    std::string serverId;                       // 本端server id

    HcclIpAddress localIpAddr;     // 本端rank ip
    HcclIpAddress remoteIpAddr;    // 对端rank ip

    u32 localSocketPort;
    u32 remoteSocketPort;

    s32 localDeviceId{-1};                      // 本端device physical id
    s32 remoteDeviceId{-1};                     // 对端device physical id
    s32 deviceLogicId{0};

    u32 localUserrank{INVALID_VALUE_RANKID};    // 本端user rank
    u32 remoteUserrank{INVALID_VALUE_RANKID};   // 对端user rank

    u32 localWorldRank{INVALID_VALUE_RANKID};   // 本端world group rank
    u32 remoteWorldRank{INVALID_VALUE_RANKID};  // 对端world group rank

    NICDeployment nicDeploy{NICDeployment::NIC_DEPLOYMENT_DEVICE};
    DevType deviceType{DevType::DEV_TYPE_COUNT};

    std::vector<std::shared_ptr<HcclSocket> > sockets;
    std::vector<u8> exchangeInfo; // 自定义交换数据,限制MAX_EXCHANGE_DATA_LEN = 2MB

    DeviceMem inputMem{DeviceMem()};
    DeviceMem outputMem{DeviceMem()};
    std::vector<DeviceMem> mem{};

    // link特性位图: bit0:0x1支持WRITE操作（源端发起数据传输，优选项）
    // bit1:0x2支持READ操作（目的端发起数据传输）。如果同时支持link优先选用目的端发起数据传输
    u64 linkAttribute{0x1}; // 初始设置为WRITE操作，从源端发起数据传输;

    bool supportDataReceivedAck{false};
    bool isAicpuModeEn{false};
    std::vector<u32> srcPorts; // 多qp配置的源端口号
    u32 notifyNum{0};
    QPMode qpMode{QPMode::INVALID}; // 是否为普通QP模式
    u32 tc { HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET };
    u32 sl { HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET };
    TagMachinePara() {}

    TagMachinePara(const struct TagMachinePara &that)
    {
        machineType = (that.machineType);
        linkMode = (that.linkMode);
        serverId = (that.serverId);
        localIpAddr = (that.localIpAddr);
        remoteIpAddr = (that.remoteIpAddr);
        localDeviceId = (that.localDeviceId);
        remoteDeviceId = (that.remoteDeviceId);
        localUserrank = (that.localUserrank);
        remoteUserrank = (that.remoteUserrank);
        localWorldRank = (that.localWorldRank);
        remoteWorldRank = (that.remoteWorldRank);
        collectiveId = (that.collectiveId);
        deviceType = (that.deviceType);
        tag = (that.tag);
        inputMem = (that.inputMem);
        outputMem = (that.outputMem);
        mem = (that.mem);
        linkAttribute = (that.linkAttribute);
        sockets = (that.sockets);
        exchangeInfo = (that.exchangeInfo);
        supportDataReceivedAck = (that.supportDataReceivedAck);
        nicDeploy = (that.nicDeploy);
        localSocketPort = that.localSocketPort;
        remoteSocketPort = that.remoteSocketPort;
        isAicpuModeEn = that.isAicpuModeEn;
        deviceLogicId = that.deviceLogicId;
        srcPorts = that.srcPorts;
        notifyNum = that.notifyNum;
        qpMode = that.qpMode;
        tc = that.tc;
        sl = that.sl;
    }

    struct TagMachinePara &operator=(struct TagMachinePara &that)
    {
        if (&that != this) {
            machineType = (that.machineType);
            linkMode = (that.linkMode);
            serverId = (that.serverId);
            localIpAddr = (that.localIpAddr);
            remoteIpAddr = (that.remoteIpAddr);
            localDeviceId = (that.localDeviceId);
            remoteDeviceId = (that.remoteDeviceId);
            localUserrank = (that.localUserrank);
            remoteUserrank = (that.remoteUserrank);
            localWorldRank = (that.localWorldRank);
            remoteWorldRank = (that.remoteWorldRank);
            collectiveId = (that.collectiveId);
            deviceType = (that.deviceType);
            tag = (that.tag);
            inputMem = (that.inputMem);
            outputMem = (that.outputMem);
            mem = (that.mem);
            linkAttribute = (that.linkAttribute);
            sockets = (that.sockets);
            exchangeInfo = (that.exchangeInfo);
            supportDataReceivedAck = (that.supportDataReceivedAck);
            localSocketPort = that.localSocketPort;
            remoteSocketPort = that.remoteSocketPort;
            isAicpuModeEn = that.isAicpuModeEn;
            deviceLogicId = that.deviceLogicId;
            srcPorts = that.srcPorts;
            notifyNum = that.notifyNum;
            qpMode = that.qpMode;
            tc = that.tc;
            sl = that.sl;
        }

        return *this;
    }
};

struct TransportPara {
    std::chrono::milliseconds timeout;
    NICDeployment nicDeploy;
    u32 localDieID;
    u32 dstDieID;
    HcclIpAddress* selfIp;
    HcclIpAddress* peerIp;
    u32 peerPort;
    u32 selfPort;
    const void* transportResourceInfoAddr;
    size_t transportResourceInfoSize;
    u32 index;
    bool isRootRank;
    u32 devLogicId;
    u32 proxyDevLogicId;
    s32 qpMode = 0;
    bool isHdcMode = false;
    bool remoteIsHdc = false;
    bool isESPs = false;
    bool virtualFlag = false;
};

struct TransportDeviceP2pData {
    void *inputBufferPtr;
    void *outputBufferPtr;
    std::shared_ptr<LocalIpcNotify> ipcPreWaitNotify;
    std::shared_ptr<LocalIpcNotify> ipcPostWaitNotify;
    std::vector<std::shared_ptr<LocalIpcNotify>> userLocalNotify;
    std::shared_ptr<RemoteNotify> ipcPreRecordNotify;
    std::shared_ptr<RemoteNotify> ipcPostRecordNotify;
    std::vector<std::shared_ptr<RemoteNotify>> userRemoteNotify;
    TransportAttr transportAttr;

    TransportDeviceP2pData() {}
    TransportDeviceP2pData(void *inputBufferPtr,
                           void *outputBufferPtr,
                           std::shared_ptr<LocalIpcNotify> ipcPreWaitNotify,
                           std::shared_ptr<LocalIpcNotify> ipcPostWaitNotify,
                           std::vector<std::shared_ptr<LocalIpcNotify>> userLocalNotify,
                           std::shared_ptr<RemoteNotify> ipcPreRecordNotify,
                           std::shared_ptr<RemoteNotify> ipcPostRecordNotify,
                           std::vector<std::shared_ptr<RemoteNotify>> userRemoteNotify,
                           TransportAttr &transportAttr)
        : inputBufferPtr(inputBufferPtr),
          outputBufferPtr(outputBufferPtr),
          ipcPreWaitNotify(ipcPreWaitNotify),
          ipcPostWaitNotify(ipcPostWaitNotify),
          userLocalNotify(userLocalNotify),
          ipcPreRecordNotify(ipcPreRecordNotify),
          ipcPostRecordNotify(ipcPostRecordNotify),
          userRemoteNotify(userRemoteNotify),
          transportAttr(transportAttr)
    {}
};

struct TransportDeviceIbverbsData {
    void *inputBufferPtr;
    void *outputBufferPtr;
    MemDetails localInputMem;
    MemDetails localOutputMem;
    std::shared_ptr<LocalIpcNotify> ackNotify;
    std::shared_ptr<LocalIpcNotify> dataAckNotify;
    std::shared_ptr<LocalIpcNotify> dataNotify;
    std::vector<std::vector<std::shared_ptr<LocalIpcNotify>>> userLocalNotify;
    uint64_t localNotifyValueAddr;
    uint64_t remoteAckNotifyAddr;
    uint64_t remoteDataNotifyAddr;
    uint64_t remoteDataAckNotifyAddr;
    std::vector<std::vector<uint64_t>> userRemoteNotifyAddr;
    uint32_t notifyValueKey;
    uint32_t remoteNotifyKey;
    std::vector<uint32_t> userRemoteNotifyKey;
    std::vector<struct HcclQpInfoV2> qpInfo;
    uint32_t remoteInputKey;
    uint32_t remoteOutputKey;
    uint32_t notifySize;
    s64 chipId;
    u32 multiQpThreshold;
    u32 qpsPerConnection;

    TransportDeviceIbverbsData()
    {}
    TransportDeviceIbverbsData(void *inputBufferPtr,
                               void *outputBufferPtr,
                               MemDetails localInputMem,
                               MemDetails localOutputMem,
                               std::shared_ptr<LocalIpcNotify> ackNotify,
                               std::shared_ptr<LocalIpcNotify> dataAckNotify,
                               std::shared_ptr<LocalIpcNotify> dataNotify,
                               std::vector<std::vector<std::shared_ptr<LocalIpcNotify>>> userLocalNotify,
                               uint64_t localNotifyValueAddr,
                               uint64_t remoteAckNotifyAddr,
                               uint64_t remoteDataNotifyAddr,
                               uint64_t remoteDataAckNotifyAddr,
                               std::vector<std::vector<uint64_t>> userRemoteNotifyAddr,
                               uint32_t notifyValueKey,
                               uint32_t remoteNotifyKey,
                               std::vector<uint32_t> userRemoteNotifyKey,
                               std::vector<struct HcclQpInfoV2> qpInfo,
                               uint32_t remoteInputKey,
                               uint32_t remoteOutputKey,
                               uint32_t notifySize,
                               s64 chipId,
                               u32 multiQpThreshold,
                               u32 qpsPerConnection)
        : inputBufferPtr(inputBufferPtr),
          outputBufferPtr(outputBufferPtr),
          localInputMem(localInputMem),
          localOutputMem(localOutputMem),
          ackNotify(ackNotify),
          dataAckNotify(dataAckNotify),
          dataNotify(dataNotify),
          userLocalNotify(userLocalNotify),
          localNotifyValueAddr(localNotifyValueAddr),
          remoteAckNotifyAddr(remoteAckNotifyAddr),
          remoteDataNotifyAddr(remoteDataNotifyAddr),
          remoteDataAckNotifyAddr(remoteDataAckNotifyAddr),
          userRemoteNotifyAddr(userRemoteNotifyAddr),
          notifyValueKey(notifyValueKey),
          remoteNotifyKey(remoteNotifyKey),
          userRemoteNotifyKey(userRemoteNotifyKey),
          qpInfo(qpInfo),
          remoteInputKey(remoteInputKey),
          remoteOutputKey(remoteOutputKey),
          notifySize(notifySize),
          chipId(chipId),
          multiQpThreshold(multiQpThreshold),
          qpsPerConnection(qpsPerConnection)
    {}

    TransportDeviceIbverbsData(const TransportDeviceIbverbsData &that)
        : inputBufferPtr(that.inputBufferPtr),
          outputBufferPtr(that.outputBufferPtr),
          localInputMem(that.localInputMem),
          localOutputMem(that.localOutputMem),
          ackNotify(that.ackNotify),
          dataAckNotify(that.dataAckNotify),
          dataNotify(that.dataNotify),
          userLocalNotify(that.userLocalNotify),
          localNotifyValueAddr(that.localNotifyValueAddr),
          remoteAckNotifyAddr(that.remoteAckNotifyAddr),
          remoteDataNotifyAddr(that.remoteDataNotifyAddr),
          remoteDataAckNotifyAddr(that.remoteDataAckNotifyAddr),
          userRemoteNotifyAddr(that.userRemoteNotifyAddr),
          notifyValueKey(that.notifyValueKey),
          remoteNotifyKey(that.remoteNotifyKey),
          userRemoteNotifyKey(that.userRemoteNotifyKey),
          qpInfo(that.qpInfo),
          remoteInputKey(that.remoteInputKey),
          remoteOutputKey(that.remoteOutputKey),
          notifySize(that.notifySize),
          chipId(that.chipId),
          multiQpThreshold(that.multiQpThreshold),
          qpsPerConnection(that.qpsPerConnection)
    {}
};
using CqeInfo =  struct tagCqeInfo {
    struct timeval time;
    uint32_t status = 0;
    HcclIpAddress remoteIp;
    char reserved[32];
    tagCqeInfo() {}
    tagCqeInfo(const struct timeval &time, const uint32_t status, const HcclIpAddress &remoteIp)
    : time(time), status(status), remoteIp(remoteIp)
    {}
};


class Transport {
public:
    struct Buffer {
        const void *addr{nullptr};
        u32 size{0};

        Buffer() : addr(nullptr), size(0) {}
        Buffer(const void *addr, u32 size) : addr(addr), size(size) {}
    };

    Transport() {};
    explicit Transport(TransportBase *pimpl): pimpl_(pimpl) {};
    Transport(TransportType type, TransportPara& para,
              const HcclDispatcher dispatcher,
              const std::unique_ptr<NotifyPool> &notifyPool,
              MachinePara &machinePara,
              const TransportDeviceP2pData &transDevP2pData = TransportDeviceP2pData(),
              const TransportDeviceIbverbsData &transDevIbverbsData = TransportDeviceIbverbsData());

    ~Transport();

    HcclResult Stop();
    HcclResult Resume();
    HcclResult Init();
    HcclResult DeInit();

    HcclResult TxDataSignal(Stream &stream);
    HcclResult RxDataSignal(Stream &stream);

    HcclResult TxAsync(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream);
    HcclResult TxAsync(std::vector<TxMemoryInfo>& txMems, Stream &stream);

    HcclResult TxWithReduce(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len,
                                    const HcclDataType datatype, HcclReduceOp redOp, Stream &stream);
    HcclResult TxWithReduce(const std::vector<TxMemoryInfo> &txWithReduceMems, const HcclDataType datatype,
        HcclReduceOp redOp, Stream &stream);
    HcclResult RxWithReduce(UserMemType recvSrcMemType, u64 recvSrcOffset, void *recvDst, u64 recvLen,
        void *reduceSrc, void *reduceDst, u64 reduceDataCount, HcclDataType reduceDatatype,
        HcclReduceOp reduceOp, Stream &stream, const u64 reduceAttr);
    HcclResult RxWithReduce(const std::vector<RxWithReduceMemoryInfo> &rxWithReduceMems,
        HcclDataType reduceDatatype, HcclReduceOp reduceOp, Stream &stream,
        const u64 reduceAttr);
    bool IsSupportTransportWithReduce();

    HcclResult RxAsync(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream);
    HcclResult RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream &stream);
    HcclResult DataReceivedAck(Stream &stream);

    HcclResult TxAck(Stream &stream);
    HcclResult RxAck(Stream &stream);

    HcclResult TxPrepare(Stream &stream);
    HcclResult RxPrepare(Stream &stream);

    HcclResult TxDone(Stream &stream);
    HcclResult RxDone(Stream &stream);

    HcclResult TxData(UserMemType dstMemType, u64 dstOffset, const void *src, u64 len, Stream &stream);
    HcclResult RxData(UserMemType srcMemType, u64 srcOffset, void *dst, u64 len, Stream &stream);

    // 保证send语义完成
    HcclResult TxWaitDone(Stream &stream);
    // 保证recv语义完成
    HcclResult RxWaitDone(Stream &stream);
    // TxWaitDone、RxWaitDone共同出现保证sendrecv语义完成

    HcclResult Post(u32 notifyIdx, Stream &stream);
    HcclResult Wait(u32 notifyIdx, Stream &stream);

    HcclResult GetLocalNotify(std::vector<HcclSignalInfo> &localNotify);
    HcclResult GetRemoteNotify(std::vector<HcclSignalInfo> &localNotify);
    HcclResult GetRemoteMem(UserMemType memType, void **remotePtr);
    HcclResult GetRemoteMem(std::vector<void *> *remotePtrVec);
    HcclResult GetRemoteMemKey(UserMemType memType, uint32_t *remoteMemKey);
    HcclResult GetLocalRdmaNotify(std::vector<HcclSignalInfo> &rdmaNotify);
    HcclResult GetRemoteRdmaNotifyAddrKey(std::vector<AddrKey> &rdmaNotifyAddr);
    HcclResult GetLocalNotifyValueAddrKey(std::vector<AddrKey> &notifyValue);
    HcclResult GetLocalMemDetails(UserMemType memType, MemDetails &memDetails);
    HcclResult GetAiQpInfo(std::vector<HcclQpInfoV2> &aiQpInfo);
    HcclResult GetAiRMAQueueInfo(std::vector<HcclAiRMAQueueInfo> &aiRMAQueueInfo);
    HcclResult GetTransportId(u32 &id);
    HcclResult GetChipId(s64 &chipId);
    virtual HcclResult GetRemoteMemSize(UserMemType memType, u64 &size);
    HcclResult GetTxAckDevNotifyInfo(HcclSignalInfo &notifyInfo);
    HcclResult GetRxAckDevNotifyInfo(HcclSignalInfo &notifyInfo);
    HcclResult GetTxDataSigleDevNotifyInfo(HcclSignalInfo &notifyInfo);
    HcclResult GetRxDataSigleDevNotifyInfo(HcclSignalInfo &notifyInfo);

    hccl::LinkType GetLinkType() const;
    bool IsSpInlineReduce() const;
    bool GetSupportDataReceivedAck() const;
    void SetSupportDataReceivedAck(bool supportDataReceivedAck);
    u32 GetRemoteRank();

    HcclResult ConnectAsync(u32& status);
    HcclResult ConnectQuerry(u32& status);
    void Break();

    void EnableUseOneDoorbell();

    bool GetUseOneDoorbellValue();

    HcclResult GetTransportAttr(TransportAttr &attr);

    HcclResult TxEnv(const void *ptr, const u64 len, Stream &stream);
    HcclResult RxEnv(Stream &stream);
    bool IsTransportRoce();

    HcclResult WriteAsync(struct Buffer &remoteBuf, struct Buffer &localBuf, Stream &stream);
    HcclResult WriteSync(struct Buffer &remoteBuf, struct Buffer &localBuf, Stream &stream);

    HcclResult WriteReduceAsync(struct Buffer &remoteBuf, struct Buffer &localBuf,
        const HcclDataType datatype, HcclReduceOp redOp, Stream &stream);

    HcclResult ReadAsync(struct Buffer &localBuf, struct Buffer &remoteBuf, Stream &stream);
    HcclResult ReadSync(struct Buffer &localBuf, struct Buffer &remoteBuf, Stream &stream);
    HcclResult ReadReduceSync(struct Buffer &localBuf, struct Buffer &remoteBuf,
        const HcclDataType datatype, HcclReduceOp redOp, Stream &stream);

    HcclResult PostReady(Stream &stream);
    HcclResult WaitReady(Stream &stream);

    HcclResult PostFin(Stream &stream);
    HcclResult WaitFin(Stream &stream);

    HcclResult PostFinAck(Stream &stream);
    HcclResult WaitFinAck(Stream &stream);

    HcclResult SetStopFlag(bool value);
    HcclResult UpdateRemoteAddr(void *remoteIn, void *remoteOut);
    static HcclResult GetTransportErrorCqe(const HcclNetDevCtx netDevCtx,
        std::vector<std::pair<Transport*, CqeInfo>> &infos, u32 &num);
    inline TransportType GetTransportType() const
    {
        return type_;
    }

    std::vector<u8> GetExchangeInfo();

private:
    void CreateTransportRoce(TransportType type, TransportPara& para, const HcclDispatcher dispatcherPtr,
        const std::unique_ptr<NotifyPool> &notifyPool, MachinePara &machinePara);
    TransportBase *pimpl_ = nullptr;
    const TransportType type_ = TransportType::TRANS_TYPE_RESERVED;

    static std::mutex mapMutex_;
    static std::unordered_map<TransportBase*, Transport*> transportMap_;
};

using LINK = std::shared_ptr<Transport>;
}  // namespace hccl

#endif /* TRANSPORT_BASE_H */
