/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: aicpu算子头文件
 */

#ifndef AICPU_OPERATOR_PUB_H
#define AICPU_OPERATOR_PUB_H
#include <stdlib.h>
#include <chrono>
#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "common.h"
#include "hdc_pub.h"
#include "transport_pub.h"
#include "dispatcher_task_types.h"
#include "externalinput_pub.h"

using RankId = u32;

namespace {
constexpr u32 RANK_NUM = 32;
}
struct HcclStreamInfo {
    s32 streamIds;
    u32 sqIds;
    u32 cqIds;   // 记录物理cqId
    u32 logicCqids; // 记录逻辑cqId
};

struct HcclMC2WorkSpace {
    u64 workSpace;
    u64 workSpaceSize;
};

struct HcclOpConfig {
    u8 deterministic; // 确定性计算开关
    u8 retryEnable;  // 是否重执行
    u8 highPerfEnable;
    u8 padding[5];  // 大小需要64By对齐，未来添加参数时减小padding
    std::chrono::milliseconds linkTimeOut; //发送超时时长
    u64 notifyWaitTime;  // 超时时长，同HCCL_EXEC_TIMEOUT
    u32 retryHoldTime;
    u32 retryIntervalTime;
    bool interHccsDisable = false;  // 使能rdma开关
    rtFloatOverflowMode_t floatOverflowMode = RT_OVERFLOW_MODE_UNDEF;
    u32 multiQpThreshold{HCCL_MULTI_QP_THRESHOLD_DEFAULT};  // 多QP每个QP分担数据量最小阈值
};

// TP8卡
struct HcclCombinOpSignalParam {
    HcclSignalInfo noIpcNotifys[RANK_NUM*2];
    HcclSignalInfo ipcNotifys[RANK_NUM*4];
    HcclSignalInfo noIpcEvents[RANK_NUM];
    HcclSignalInfo aicpuNotify;
    HcclSignalInfo aicpuOpNotify[2]; // 集合通信AICPU展开资源
};

struct BatchSendRecvInfo {
    uint8_t bsrTag[128];
    uint32_t index;
    uint32_t tpQpn;
    uint32_t srcRank;
    uint32_t detRank;
    uint32_t streamId = ~0u;
};
using HcclOpIdentifier = struct HcclOpIdentifierDef {
    uint8_t tag[128];   // 通信算子的标签
    uint8_t newTag[256];
    uint32_t index;     // 集合通信算子在通信域内的编号
    uint32_t srcRank;
    uint32_t detRank;
    bool isSendRecv = false;
    u32 streamId = ~0u;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    BatchSendRecvInfo bsrInfo[2];
    bool isBsrTaskStart = false; //batchsendrecv 主流start task是否完成
};

struct HcclCombinOpParam {
    HcclMC2WorkSpace mc2WorkSpace;
    u32 rankId = 0; // 当前卡rankId
    u32 rankNum = 0;
    u64 winSize = 0; // 每个win大小
    u64 windowsIn[RANK_NUM];
    u64 windowsOut[RANK_NUM];
    char hcomId[128] = "\0";
    HcclStreamInfo streamInfo[RANK_NUM];
    HcclCombinOpSignalParam signalInfo;
    HcclOpConfig config; // 配置参数
    u64 overFlowAddr = 0;
    u8 onlyRead = 0;  // 只使用读模式，不使用写模式

    // communicate retry
    hccl::HDCommunicateParams kfcControlTransferH2DParams;
    hccl::HDCommunicateParams kfcStatusTransferD2HParams;

    u8 padding[16]; // 大小需要64By对齐，未来添加参数时减小padding
    u64 winExpSize = 0;
    u64 windowsExp[RANK_NUM]; 

    u8 multiServerFlag = 0;
    u64 ibverbsData = 0; // TransportDeviceNormalIbverbsData数组的首地址
    u64 ibverbsDataSize = 0; // TransportDeviceNormalIbverbsData数组的字节长度
};

struct HcclKFCTilingData {
    u32 preparePosition;  // 任务准备位置：0表示在host完成所有通信任务的准备，1表示在kernel侧完成
    u64 sendOff;        // 发送数据地址偏移，count * dataTypeSize
    u64 recvOff;        // 接收数据地址偏移, count * dataTypeSize
    u64 tailSendOff;    // 尾块发送数据地址偏移，count * dataTypeSize
    u64 tailRecvOff;    // 尾块发送数据地址偏移，count * dataTypeSize
    u64 sendCnt;        // 整块发送数据个数
    u64 recvCnt;        // 整块接收数据个数
    u64 tailSendCnt;    // 尾块发送数据个数
    u64 tailRecvCnt;    // 尾块接收数据个数
    u64 totalCnt;       // 总数据个数
    u32 turnNum;         // 总轮次
    u32 tailNum;         // 尾块的轮次
    u32 stride;          // 跳写间隔
    u32 workspaceOff;    // 使用workspace作为recvbuf时的workspace偏移
    u32 notifyOff;       // device notify write/read value偏移
    u16 notifyBeginCnt;  // notift write value的使用个数
    u16 notifyEndCnt;    // notift read value的使用个数
    u8 useBufferType;    // 是否使用workspace作为recvbuf
    u8 funID;            // funtion ID
    u8 dataType;         // hccl 数据类型
    u8 groupNum;         // groupNum
    u8 reuseMode;        // tiling调试，填msgCnt，内存优化选择复用的内存块个数
    u8 commType;         // 通信类型
    u8 reduceOp;         // reduce op type
    u8 commOrder;        // 通信顺序，0表示通信在前，1表示通信在后
    u8 waitPolicy;       // 等待任务启动的阻塞策略
    // 2、首轮等待，1、每轮等待。KFC根据此标记在主流任务前面加wait，AIC需要按策略发对应record才能触发执行
    u8 rspPolicy;  // 任务执行结束时的响应策略， 2、最后通知一次，
    // 1、每轮通知一次。KFC根据此标记在主流任务后面加record
    u8 exitPolicy;  // 退出策略，0，一次通信任务下发完成直接退出；1. 通信任务执行完成退出；2.
    // 等待AIC通知退出(可以多次执行任务)。
    u8 commAlg;    // 用于指定具体通信算法。
    u8 taskType;   // 用于识别不同任务。参考KfcTaskType定义
    u8 debugMode;  // 调测模式
    // 1:单独执行CUBE
    // 2:单独执行Vector
    // 4:单独执行AICPU KFC算子
    // 8:KFC等待通信结束
    // 16:KFC统计各阶段耗时
    u8 stepSize;         // 用于指定通算频率步长
    u8 sendArgIndex;     // 发送数据参数索引，对应算子原型的参数顺序
    u8 recvArgIndex;     // 接收数据参数索引，对应算子原型的参数顺序
    u8 commOutArgIndex;  // 通信输出参数索引，对应算子原型的参数顺序
    u8 hasCommOut;       // 是否有通信输出
};

enum HcclKfcTaskType {
    HCCL_KFC_TASK_HCC_RES_INIT = 1,      // 集合通信资源下发&校验
    HCCL_KFC_TASK_HCC_START_SERVER = 2,  // 只启动KFC server，从msg queue接收任务
    HCCL_KFC_TASK_HCC_TASK_PREPARE = 3,  // 从参数获取通信任务，完成准备，待AIC通知消息再激活
    HCCL_KFC_TASK_HCC_TASK_DELIVER = 4,  // 从参数获取通信任务，直接下发。AIC自己发Record激活
    HCCL_KFC_TASK_COMMON_CMD_EXE = 5,    // 预留，通用任务执行，具体任务信息可以在参数中描述。
    HCCL_KFC_TASK_HCCL_ONLY_EXE = 6,     // 通信任务单独执行
    KFC_TASK_TYPE_END
};

#define list_entry(ptr, type, member) (reinterpret_cast<type *>(reinterpret_cast<char *>(ptr) - offsetof(type, member)))

constexpr u64 COMM_P2P = 1UL << 0;
constexpr u64 COMM_RDMA = 1UL << 1;

constexpr u32 LINK_P2P_MAX_NUM = 64;
constexpr u32 MEM_DETAILS_NUM = 2; // link信息中memDetails的大小：0对应INPUT，1对应OUTPUT
constexpr u32 RDMA_QP_MAX_NUM = 33;
constexpr u32 RDMA_NOTIFY_MAX_NUM = 8 * 3;
constexpr u32 LOCAL_STREAM_MAX_NUM = 19;
constexpr u32 LOCAL_NOTIFY_MAX_NUM = 64;
using RANK_TYPE = u32;
using TAG_TYPE = u32;
using LENGTH_TYPE = u32;
constexpr u32 TOP_COMM_LEVEL0_SHIFT = 16;
constexpr u32 TOP_COMM_LEVEL0_LOCATION = 0xFFFF0000;
constexpr u32 TOP_COMM_LEVEL1_LOCATION = 0xFFFF;
constexpr u32 TOP_HIERARCHICAL_COMM_LEVEL0_SHIFT = 12;
constexpr u32 TOP_HIERARCHICAL_COMM_LEVEL1_SHIFT = 16;
constexpr u32 TOP_HIERARCHICAL_COMM_LEVEL0_LOCATION = 0xF0000000;
constexpr u32 TOP_HIERARCHICAL_COMM_LEVEL1_LOCATION = 0xFFF0000;
constexpr u32 TOP_HIERARCHICAL_COMM_LEVEL2_LOCATION = 0xFFFF;
constexpr u32 TOP_COMM_RING_LOCATION = 0;
constexpr u32 TAG_MAX_LENGTH = 256;
constexpr u32 AICPU_OP_NOTIFY_MAX_NUM = 2;
constexpr u32 AICPU_MAX_RANK_NUM = 128 * 1024;


struct HcclOpConfigV2 {
    u8 deterministic;  // 确定性计算开关
};

struct HcclMC2WorkSpaceV2 {
    u64 workSpace;
    u64 workSpaceSize;
};

struct HcclStreamInfoV2 {
    s32 streamIds;
    u32 sqIds;
};

struct ListCommon {
    u64 nextHost;
    u64 preHost;
    u64 nextDevice;
    u64 preDevice;
};

static inline void ListCommonInit(struct ListCommon *deviceList, struct ListCommon *hostList)
{
    if (hostList == nullptr) {
        HCCL_ERROR("hostList[%p] is nullptr", hostList);
        return;
    }
    hostList->nextHost = reinterpret_cast<u64>(hostList);
    hostList->preHost = reinterpret_cast<u64>(hostList);
    hostList->nextDevice = reinterpret_cast<u64>(deviceList);
    hostList->preDevice = reinterpret_cast<u64>(deviceList);
}

static inline void ListCommonAddHead(struct ListCommon *newDeviceL, struct ListCommon *newHostL,
    struct ListCommon *headHostL, struct ListCommon *headDeviceL)
{
    if (newHostL == nullptr || headHostL == nullptr) {
        HCCL_ERROR("intput ptr is nullptr, newHostL[%p], headHostL[%p]", newHostL, headHostL);
        return;
    }
    ListCommon *headHostLNextHost = reinterpret_cast<ListCommon *>(headHostL->nextHost);
    headHostLNextHost->preHost = reinterpret_cast<u64>(newHostL);
    newHostL->nextHost = headHostL->nextHost;
    newHostL->preHost = reinterpret_cast<u64>(headHostL);
    headHostL->nextHost = reinterpret_cast<u64>(newHostL);

    headHostLNextHost->preDevice = reinterpret_cast<u64>(newDeviceL);
    newHostL->nextDevice = headHostL->nextDevice;
    newHostL->preDevice = reinterpret_cast<u64>(headDeviceL);
    headHostL->nextDevice = reinterpret_cast<u64>(newDeviceL);
}

// KFC控制命令
enum class KfcCommand : int64_t {
    kNone = 0,		// 空命令
    kStopLaunch,	// 停止算子下发
    kStopExec,		// 停止算子执行
    kClear,			// 清理算子状态&资源
    kChangeLink,    // 切换主备链路
    kRetry,			// 重新执行算子
    kExit,			// 退出算子执行
    NsStopLaunch,   // N秒快恢下的停止算子下发
    NsStopExec,     // N秒快恢下的停止算子执行
    NsClear,        // N秒快恢下的清理算子状态&资源
    kDestroyComm,   // 销毁AICPU通信域
};

// AICPU背景线程控制
enum class BackgroundCommand : int64_t {
    kNone = 0,			// 空命令
    kStop,
};

// device侧感知目前通信域的状态
enum class HcclComSuspendingFlag : int64_t {
    isNull = 0,
    isResume,      // device侧感知是resume状态
    isSuspending,  // device侧感知是suspending状态
};

// host向aicpu发送link状态
using ChangeLinkInfo = struct ChangeLinkInfoDef {
    u32 remoteRankNum = 0;
    u32 remoteRankList[AICPU_MAX_RANK_NUM] = {};
    bool isUseDefaultPort[AICPU_MAX_RANK_NUM] = {};
    bool isChangeLinkFlag = false;
};

// host向aicpu发送的命令
using KfcExecControl = struct KfcExecControlDef {
    KfcCommand kfcCmd;				// 控制KFC执行
    BackgroundCommand bgCmd;		// 控制背景线程执行
    HcclComSuspendingFlag suspendingStatus;  // KFC状态
    HcclOpIdentifier targetOp;
    ChangeLinkInfo changeLinkInfo;
};

// KFC的执行状态
enum class KfcStatus : int64_t {
    kNull = 0 , // 算子的执行状态还没有initOpStatus前的状态
    kRuning,
    kEnd,
    kStoplaunch,
    kStopExec,
    kClear,
    kChanged,
    kError,
    kDestroyComm,
};

// KFC的执行错误码
enum class KfcError : int64_t {
    kNone = 0,        // 无异常
    kSdma,            // SDMA task执行失败
    kRdma,            // RDMA task执行失败
    kTimeout,         // task执行超时
    kInner,           // 通信算子展开、task下发失败
    kExec,            // 算子执行失败
    kExit,            // 算子强行终止
};

enum class PollStatus : int64_t {
    kDefault = 0,
    kStopAsException,
};

enum class BackgroundStatus : int64_t {
    kNone = 0,			// 正常状态
    kStop,              //背景线程停止状态
};

using KfcTaskException = struct KfcTaskExceptionDef {
    uint16_t streamId = 0;
    uint16_t taskId = 0;
    uint32_t errorCode = 0;
    uint8_t errorType = 0;
    uint8_t sqeType = 0;
    uint16_t sqId = 0;
};
using KfcRetryInfo = struct KfcRetryStatusDef {
    uint32_t retryCount = 0;         // 已重试次数;
};

struct ErrorMessageReport {
    char tag[TAG_MAX_LENGTH] = {0};
    char group[GROUP_NAME_MAX_LEN + 1] = {0};
    u32 remoteUserRank = 0;
    s32 streamId = 0;
    u32 taskId = 0;
    u32 notifyId = 0;
    s32 stage = 0;
    u32 rankId = 0;
    u32 rankSize = 0;
    AlgType algType;
    hccl::TaskType taskType = hccl::TaskType::TASK_SDMA;
};

struct ExecStatusDef {
        KfcStatus kfcStatus;		// KFC状态
        KfcError kfcError;			// KFC错误码
        KfcRetryInfo retryInfo;
        KfcTaskException taskException;
        PollStatus pollStatus;		// 背景线程状态
        BackgroundStatus backgroundStatus;
};

#pragma pack(push)
#pragma pack(1)
// aicpu向host提供的状态查询
using KfcExecStatus = struct KfcExecStatusDef {
    HcclOpIdentifier opId;
    ExecStatusDef execStatus;
    ErrorMessageReport emrReport;
};
#pragma pack(pop)

// P2P同步资源
struct HcclLinkP2pV2 {
    MemDetails localMem[MEM_DETAILS_NUM];
    MemDetails remoteMem[MEM_DETAILS_NUM];
    HcclSignalInfo localIpcSignal[LINK_P2P_MAX_NUM];  // localnotify
    HcclSignalInfo remoteIpcSignal[LINK_P2P_MAX_NUM];
    hccl::TransportAttr transportAttr;
};

struct HcclLinkRoceV2 {
    MemDetails localMem[MEM_DETAILS_NUM];
    MemDetails remoteMem[MEM_DETAILS_NUM];
    u64 notifyValue{0};  // notify record的src地址value存储位置
    u32 notifyValueKey{0};
    u32 singleQPNotifyNum{0};  // one link multi notify resources nums obtained via alg
    u64 localNotifyList{0};    // devicePtrValue    type:HcclSignalInfo  size: [3,8192]
    u64 remoteNotifyList{0};   // devicePtrValue    type:u64             size: [3,8192]
    u32 remoteNotifyKey{0};
    s64 chipId{LLONG_MAX};
    HcclQpInfoV2 QpInfo[RDMA_QP_MAX_NUM];  // ayyay size: [1, 1 + HCCL_RDMA_QPS_PER_CONNECTION_MAX]
    u32 qpsPerConnection{1};               // current link actual qp num obtained via env、workflowMode、qpMode
};

struct HccltagRemoteResV2 {
    ListCommon nextTagRes; 
    char tag[TAG_MAX_LENGTH]; // newtag
    HcclLinkP2pV2 linkP2p;
    HcclLinkRoceV2 linkRoce[4];
};

struct HccltagRemoteResV3 {
   HccltagRemoteResV2 *tagRemoteResPtr;
   u64 p2pNotifyNum = 0;            // 用于linkp2p添加notify信息，不能超过Max个，超过报错（不区分local&remote）
   u64 roceNotifyNum = 0;           // 主链路：用于linkroce添加notify信息，不能超过Max个，超过报错（不区分local&remote）
   u64 roceNotifyNumBackup = 0;     // 备链路：用于linkroce添加notify信息，不能超过Max个，超过报错（不区分local&remote）
   u64 qpNum = 0;                   // 主链路：QP计数，支持多个，用于linkroce添加QP信息
   u64 qpNumBackup = 0;             // 备链路：QP计数，支持多个，用于linkroce添加QP信息
};

struct HcclRankRelationResV2 {
    u32 remoteUsrRankId = 0;
    u32 remoteWorldRank = 0;
    u64 windowsIn = 0;
    u64 windowsOut = 0;
    u64 windowsExp = 0;
    ListCommon nextTagRes = { 0 }; // HccltagRemoteResV2(根据tag区分)
};

struct HccltagLocalResV2 {     // TAG专用,scratchmem，
    ListCommon nextTagRes;     // 为0时，为尾节点
    char tag[TAG_MAX_LENGTH];  // 通信域内共享的资源则与通信域的hcomid相同
    u64 ScratchmemSize;        // 为0时代表没有使用Scratchmem
    u64 Scratchmem;
};

// 卡内主从流同步，下沉图，单算子，通信域内复用
struct LocalResInfoV2 {
    u32 streamNum;
    u32 signalNum;
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM];  // 集合通信AICPU展开资源
    ListCommon nextTagRes;                                  // HccltagLocalResV2
};
struct CommonTlv {
    TAG_TYPE type;
    LENGTH_TYPE length;  // 本TLV总字节数，包括Type和length
    RANK_TYPE value;
};

struct AlgoTopoInfo {
    u32 userRank;      // 通信域 RankID
    u32 userRankSize;  // 通信域的 Rank数量
    s32 deviceLogicId;
    bool isSingleMeshAggregation;
    u32 deviceNumPerAggregation;  // 每个module中的Device数量
    u32 superPodNum;           // 集群中总的超节点数
    u32 devicePhyId;
    u32 topoType;  // TopoType
    u32 deviceType;
    u32 serverNum;
    u32 meshAggregationRankSize;
    u32 multiModuleDiffDeviceNumMode;
    u32 multiSuperPodDiffServerNumMode;
    u32 realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    u32 gcdDeviceNumPerAggregation;
    u32 moduleNum;
    u32 isUsedRdmaRankPairNum;
    u64 isUsedRdmaRankPair;
    u32 pairLinkCounterNum;
    u64 pairLinkCounter;
    u32 nicNum;
    u64 nicList;            // niclist数组指针
    u64 complanRankLength;  // complanRank占用的字节数
    u64 complanRank;        // 指针
    u64 bridgeRankNum;      // bridgeRank占用的个数
    u64 bridgeRank;         // 指针
    u64 serverAndsuperPodRankLength;  // serverAndsuperPodRank占用的字节数
    u64 serverAndsuperPodRank;    // 指针
};
struct RemoteResPtr {
    u64 nextHostPtr;
    u64 nextDevicePtr;
};

constexpr u32 ZERO_COPY_BUFFER_MAX_MAP_COUNT = 16 * 32 * 1024; // 16卡，一次每卡最多32k个映射修改
constexpr u32 ZERO_COPY_IPC_BUFFER_LENGTH = 4096;              // 跨进程交换地址所需buffer长度
struct LocalIpc2RemoteAddr {
    u32 devicePhyId = 0;
    u64 localIpcAddr = 0;
    u64 remoteAddr = 0;
    u64 length = 0;
};

enum class ZeroCopyItemType : u32 {
    SET_MEMORY,
    UNSET_MEMORY,
    ACTIVATE_MEMORY,
    DEACTIVATE_MEMORY
};

struct ZeroCopyRingBufferItem {
    ZeroCopyItemType type;
    LocalIpc2RemoteAddr addr;
};
// AICPU算子计数信息
struct OpCounterInfo {
    u64 headCountMem = 0;
    u64 tailCountMem = 0;
    u64 addOneMem = 0;
    u32 memSize = 0;
    bool isEnableCounter = false;
};

// 层次化算法信息
struct HierarchicalAlgInfo {
    u64 commplaneSubGroupRankLength;  // complanSubGroupRank占用的字节数
    u64 commplaneSubGroupRank;  // 指针
    u32 hierarchicalAlgOptionNum;
    u64 hierarchicalAlgOptionVec;    // hierarchicalAlgOptionVec数组指针
};

struct HcclOpResParam {
    // 本地资源
    HcclMC2WorkSpace mc2WorkSpace;
    u32 localUsrRankId;  // usrrankid
    u32 rankSize;        // 通信域内total rank个数
    u64 winSize;  // 每个win大小，静态图时，可能为0，如果通信域内也有动态图，则可能为非0
    u64 localWindowsIn;   // 全F为无效值
    u64 localWindowsOut;  // 全F为无效值
    char hcomId[128];
    u64 winExpSize;
    u64 localWindowsExp;
    // aicore识别remote window
    u32 rWinStart;   // 为HcclRankRelationRes起始位置
    u32 rWinOffset;  // 为HcclRemoteRes的大小
    u64 version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;

    // 外部配置参数
    HcclOpConfig config;
    u64 hostStateInfo;
    u64 aicpuStateInfo;
    u64 lockAddr;
    u32 rsv[16];
    u32 notifysize;                              // RDMA场景使用，910B/910_93为4B，其余芯片为8B
    u32 remoteResNum;                            // 有效的remoteResNum
    RemoteResPtr remoteRes[AICPU_MAX_RANK_NUM];  // 数组指针，指向HcclRankRelationResV2，下标为remoteUserRankId

    // communicate retry
    hccl::HDCommunicateParams kfcControlTransferH2DParams;
    hccl::HDCommunicateParams kfcStatusTransferD2HParams;

    u64 tinyMem;   // for all2all
    u64 tinyMemSize;

    // 零拷贝场景使用
    u64 zeroCopyHeadPtr;
    u64 zeroCopyTailPtr;
    u64 zeroCopyRingBuffer;
    u64 zeroCopyIpcPtrs[MAX_MODULE_DEVICE_NUM];               // 保存集合通信时每个对端的输入输出内存地址
    u32 zeroCopyDevicePhyId[MAX_MODULE_DEVICE_NUM];           // 保存每个rank对应的物理卡Id

    bool utraceStatusFlag;

    OpCounterInfo opCounterInfo;

    HierarchicalAlgInfo hierarchicalAlgInfo;
};

struct OpTilingData {
    char tag[128];
    char newTag[256];
    char algName[128];
    u32 index; // 集合通信算子在通信域内的编号
    u64 algType;
    u8 floatOverflowMode;
    u8 dumpDebug;
    u8 debugMode;
    u8 workflowMode;
    u64 inputPtr;
    u64 outputPtr;
    u8 reduceType;  // HcclReduceOp ::HCCL_REDUCE_RESERVED
    u8 syncMode;    // SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    RankId root = INVALID_VALUE_RANKID;
    RankId dstRank = 0;
    RankId srcRank = 0;
    u8 opType; // HcclCMDType::HCCL_CMD_INVALID;
    u8 inplaceSupportRetry;
    u8 retryEnable;
    u8 inPlaceSupportRetryStatus;
    u8 isInplacePreSync;
    u8 isPostSync;
    u8 isZeroCopy = 0;
    u64 version = 0;

    /******************可变长度数据区，如需新增字段请在这之前增加*******************/
    u64 length;   // 可变长度数据区长度
 
    /* 不同算子，长度不同，依据opType决定选择使用
    * (1)batchsendrcv
    * struct {
    *     u32 itemNum;
    *     HcclSendRecvItem orderedList[itemNum];
    * } OpTilingBatchSendRecvDataDes;
    * (2)alltoallv
    * struct {
    *     u8 sendType;  // HcclDataType
    *     u8 recvType;  // HcclDataType
    *     u32 rankSize;
    *     u64 sendCounts[rankSize];
    *     u64 recvCounts[rankSize];
    *     u64 sdispls[rankSize];
    *     u64 rdispls[rankSize];
    * };
    * (3)alltoallvc
    * struct {
    *     u8 sendType;  // HcclDataType
    *     u8 recvType;  // HcclDataType
    *     u32 rankSize;
    *     u64 sendCountMatrix[rankSize * rankSize];
    * };
    *  (4)alltoall
    * struct {
    *     u8 sendType;  // HcclDataType
    *     u8 recvType;  // HcclDataType
    *     u64 sendCount;
    *     u64 recvCount;
    * };
    *  (5)other operators
    * struct {
    *     u64 count;
    *     u8 datatype;  // HcclDataType
    * }; */
};
struct OpTilingDataDes {
    u64 count;
    u8 dataType; 
};

struct OpTilingBatchSendRecvDataDes {
    u32 itemNum;
    HcclSendRecvItem batchSendRecvItem[];
};
struct OpTilingAllToAllDataDes{
    u8 sendType;  // HcclDataType
    u8 recvType;  // HcclDataType
    u64 sendCount;
};
struct OpTilingAlltoallvDataDes {
    u8 sendType;  // HcclDataType
    u8 recvType;  // HcclDataType
    u64 sendRecvInfos[];
};
struct OpTilingAlltoallvcDataDes {
    u8 sendType;  // HcclDataType
    u8 recvType;  // HcclDataType
    u64 sendCountMatrix[];
};

struct TransportDeviceNormalData {
    MemDetails remoteInputMem{};
    MemDetails remoteOutputMem{};
    MemDetails localInputMem{};
    MemDetails localOutputMem{};
    struct HcclQpInfoV2 qpInfo{};
    QPMode qpMode{QPMode::INVALID};
    void Print() {
        HCCL_DEBUG("remoteInputMem: addr[%llu], size[%llu], key[%u]", remoteInputMem.addr, remoteInputMem.size, remoteInputMem.key);
        HCCL_DEBUG("remoteOutputMem: addr[%llu], size[%llu], key[%u]", remoteOutputMem.addr, remoteOutputMem.size, remoteOutputMem.key);
        HCCL_DEBUG("remoteInputMem: addr[%llu], size[%llu], key[%u]", localInputMem.addr, localInputMem.size, localInputMem.key);
        HCCL_DEBUG("remoteOutputMem: addr[%llu], size[%llu], key[%u]", localOutputMem.addr, localOutputMem.size, localOutputMem.key);
        HCCL_DEBUG("qpInfo: qpPtr[%llu], sqIndex[%u], dbIndex[%u]", qpInfo.qpPtr, qpInfo.sqIndex, qpInfo.dbIndex);
        HCCL_DEBUG("qpMode[%d]", static_cast<int32_t>(qpMode));
        return ;
    }
};

#endif
