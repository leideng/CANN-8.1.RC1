/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: the definition of stars
 */

#ifndef CCE_RUNTIME_RT_STARS_DEFINE_H
#define CCE_RUNTIME_RT_STARS_DEFINE_H

#include "base.h"

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

#pragma pack(push)
#pragma pack (1)

typedef struct tagStarsSqeHeader {
    uint8_t type : 6;
    uint8_t l1Lock : 1;
    uint8_t l1Unlock : 1;

    uint8_t ie : 2;
    uint8_t preP : 2;
    uint8_t postP : 2;
    uint8_t wrCqe : 1;
    uint8_t reserved : 1;

    uint16_t blockDim;

    uint16_t rtStreamId;
    uint16_t taskId;
} rtStarsSqeHeader_t;

typedef struct tagDavidStarsSqeHeader {
    /* word0 */
    uint8_t type : 6;
    uint8_t lock : 1;
    uint8_t unlock : 1;
    uint8_t ie : 1;
    uint8_t preP : 1;
    uint8_t postP : 1;
    uint8_t wrCqe : 1;
    uint8_t ptrMode : 1;
    uint8_t rttMode : 1;
    uint8_t headUpdate : 1;
    uint8_t res0 : 1;
    uint16_t blockDim;

    /* word1 */
    uint16_t rtStreamId;
    uint16_t taskId;
} rtDavidStarsSqeHeader_t;

typedef struct tagRtStarsCommonSqe {
    rtStarsSqeHeader_t sqeHeader;  // word 0-1
    uint32_t commandCustom[14];       // word 2-15 is custom define by command.
} rtStarsCommonSqe_t;

typedef struct tagRtDavidStarsCommonSqe {
    rtDavidStarsSqeHeader_t sqeHeader;  // word 0-1
    uint32_t commandCustom[14];       // word 2-15 is custom define by command.
} rtDavidStarsCommonSqe_t;

typedef struct tagStarsDsaSqe {
    // 0-7 bytes
    rtStarsSqeHeader_t sqeHeader;
    // 8-11 bytes
    uint32_t start : 1;
    uint32_t functionType : 3;
    uint32_t dataType : 3;
    uint32_t algoType : 3;
    uint32_t paramVldBitmap : 5;
    uint32_t paramAddrValBitmap : 7;
    uint32_t reserved0 : 10;
    // 12-15 bytes
    uint16_t sqeIndex;
    uint8_t kernelCredit;
    uint8_t reserved1;
    // 16-31 bytes
    uint32_t dsaCfgResultAddrLow;
    uint32_t dsaCfgResultAddrHigh;
    uint32_t dsaCfgStateAddrLow;
    uint32_t dsaCfgStateAddrHigh;
    // 32-47 bytes
    uint32_t dsaCfgParamAddrLow;
    uint32_t dsaCfgParamAddrHigh;
    uint32_t dsaCfgSeedLow;
    uint32_t dsaCfgSeedHigh;
    // 48-63 bytes
    uint32_t dsaCfgNumberLow;
    uint32_t dsaCfgNumberHigh;
    uint32_t reserved2[2];
} rtStarsDsaSqe_t;

// ffts+ type
typedef enum tagFftsPlusType {
    RT_FFTS_PLUS_TYPE_RES1 = 2,   // Reserved
    RT_FFTS_PLUS_TYPE_RES2 = 3,   // Reserved
    RT_FFTS_PLUS_TYPE = 4,        // FFTS+ mode
} rtFftsPlusType_t;

typedef struct tagStarsFftsPlusHeader {
    uint8_t type : 6;
    uint8_t l1Lock : 1;
    uint8_t l1Unlock : 1;

    uint8_t ie : 2;
    uint8_t preP : 2;
    uint8_t postP : 2;
    uint8_t wrCqe : 1;
    /* tell mcu if this subgraph is overflow-enabled and mcu will send this flag to aicpu when aicpu ctx is excuted */
    uint8_t overflowEn : 1;

    uint16_t blockDim;

    uint16_t rtStreamId;
    uint16_t taskId;
} rtStarsFftsPlusHeader_t;
// ffts+ sqe
typedef struct tagFftsPlusSqe {
    // 0-7 bytes
    rtStarsSqeHeader_t sqeHeader; // use rtStarsFftsPlusHeader_t instead
    // 8-11 bytes
    uint16_t fftsType : 3;
    uint16_t cmo : 1;
    uint16_t scheduleDfxFlag : 1;
    uint16_t reserved1 : 7;
    uint16_t wrrRatio : 4;
    uint16_t dsaSqId : 11;
    uint16_t reserved2 : 5;
    // 12-15 bytes
    uint16_t sqeIndex;
    uint8_t  kernelCredit;
    uint8_t  subType;
    // 16-23 bytes
    uint32_t stackPhyBaseL;
    uint32_t stackPhyBaseH;
    // 24-31 bytes
    uint16_t  totalContextNum;
    uint16_t  readyContextNum;
    uint16_t  preloadContextNum;
    uint16_t  timeout;
    // 32-35 bytes
    uint16_t  reserved6;
    uint16_t  prefetchOstNum : 5;
    uint16_t  reserved9 : 3;
    uint16_t  cmaintOstNum : 5;
    uint16_t  reserved10 : 3;
    // 36-39 bytes
    uint16_t  aicPrefetchLower : 5;
    uint16_t  reserved11 : 3;
    uint16_t  aicPrefetchUpper : 5;
    uint16_t  reserved12 : 3;
    uint16_t  aivPrefetchLower : 5;
    uint16_t  reserved13 : 3;
    uint16_t  aivPrefetchUpper : 5;
    uint16_t  reserved14 : 3;
    // 40-47 bytes
    uint32_t contextAddressBaseL;
    uint32_t contextAddressBaseH : 17;
    uint32_t reserved15 : 15;
    // 48-63 bytes:48-51 use for pid
    // use reserved16[1] bit 4 for l2cache
    uint32_t reserved16[4];
} rtFftsPlusSqe_t;

typedef enum {
    RECORD_STORE_MODE = 0x0U,
    RECORD_ADD_MODE = 0x1U,
    RECORD_WRITE_BIT_MODE = 0x2U,
    RECORD_INVALID_MODE = 0x3U, // invalid, cannot set
    RECORD_CLEAR_BIT_MODE = 0x4U,
    RECORD_MODE_BUT
} rtCntNotifyRecordMode_t;

typedef enum {
    WAIT_LESS_MODE = 0x0U,
    WAIT_EQUAL_MODE = 0x1U,
    WAIT_BIGGER_MODE = 0x2U,
    WAIT_BIGGER_OR_EQUAL_MODE = 0x3U,
    WAIT_BITMAP_MODE = 0x4U,
    WAIT_MODE_BUT
} rtCntNotifyWaitMode_t;

typedef enum {
    NOTIFY_TABLE_SLICE  = 0U,
    NOTIFY_CNT_ST_SLICE  = 1U,
    NOTIFY_CNT_ADD_SLICE  = 2U,
    NOTIFY_CNT_BIT_WR_SLICE  = 3U,
    NOTIFY_CNT_BIT_CLR_SLICE  = 4U,
    NOTIFY_TYPE_BUFF
} rtNotifyType_t;

typedef struct tagCmoTaskInfo {
    uint8_t  qos;
    uint8_t  partId;
    uint8_t  pmg;
    uint8_t  reserved;
    uint16_t cmoType;
    uint16_t opCode; // 6: Preload; 7: Prewriteback; 8: invalid; 9: flush;
    uint16_t numInner;
    uint16_t numOuter;
    uint32_t logicId;
    uint32_t lengthInner;
    uint64_t sourceAddr;
    uint32_t striderOuter;
    uint32_t striderInner;
} rtCmoTaskInfo_t;

typedef struct tagBarrierCmoInfo {
    uint16_t cmoType; // 0 is barrier, 1 is invalid, Prefetch is 2, Write_back is 3, FE/GE only use invalid type.
    uint32_t logicId;
} rtBarrierCmoInfo_t;

#define RT_CMO_MAX_BARRIER_NUM 6U // 6U is max support
typedef struct tagBarrierTaskInfo {
    uint8_t logicIdNum;
    rtBarrierCmoInfo_t cmoInfo[RT_CMO_MAX_BARRIER_NUM];
} rtBarrierTaskInfo_t;

typedef enum tagRtCmoType {
    RT_CMO_PREFETCH = 6, // Preload
    RT_CMO_WRITEBACK, // Prewriteback
    RT_CMO_INVALID, // invalid
    RT_CMO_FLUSH, // flush
    RT_CMO_RESERVED,
} rtCmoOpCode_t;

typedef rtCmoOpCode_t rtCmoOpCode;

typedef struct {
    uint32_t resv0;
    uint32_t resv1;
    uint16_t num_outer;
    uint16_t num_inner;
    uint32_t len_inner;
    uint64_t src;
    uint32_t stride_outer;
    uint32_t stride_inner;
} rtCmoAddrInfo;

#pragma pack(pop)

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif
#endif // CCE_RUNTIME_RT_STARS_DEFINE_H
