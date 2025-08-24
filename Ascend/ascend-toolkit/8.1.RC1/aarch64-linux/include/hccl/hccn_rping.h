/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: rping功能对外头文件
 */

#ifndef HCCN_RPING_H_
#define HCCN_RPING_H_

#include <stdint.h>
 
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void* HccnRpingCtx;

#define HCCN_RPING_PAYLOAD_LEN_MAX 1500

typedef enum {
    HCCN_SUCCESS = 0,    /* success */
    HCCN_E_AGAIN,        /* try again */
    HCCN_E_FAIL,         /* fail */
    HCCN_E_PARA,         /* wrong parameter */
    HCCN_E_MEM,          /* memory */
    HCCN_E_RESERVED      /* reserved */
} HccnResult;

typedef enum {
    HCCN_RPING_MODE_ROCE = 0,    /* RoCE */
    HCCN_RPING_MODE_RESERVED     /* reserved */
} HccnRpingMode;

typedef enum {
    HCCN_RPING_ADDTARGET_STATE_DONE = 0,        /* add success */
    HCCN_RPING_ADDTARGET_STATE_DOING,           /* adding */
    HCCN_RPING_ADDTARGET_STATE_FAIL,            /* add fail */
    HCCN_RPING_ADDTARGET_STATE_TIMEOUT,         /* connect target timeout */
    HCCN_RPING_ADDTARGET_STATE_RESERVED         /* reserved */
} HccnRpingAddTargetState;

typedef enum {
    HCCN_RPING_RESULT_STATE_NOT_FOUND = 0,    /* not found */
    HCCN_RPING_RESULT_STATE_INVALID,          /* invalid */
    HCCN_RPING_RESULT_STATE_VALID,            /* valid */
    HCCN_RPING_RESULT_STATE_RESERVED          /* reserved */
} HccnRpingResultState;

typedef struct HccnRpingInitAttrDef {
    HccnRpingMode mode;  /* Link type: RoCE : HCCN_RPING_MODE_ROCE / others */
    uint32_t port;       /* Port to listen when device being target */
    uint32_t npuNum;     /* Numbers of all the devices in the net */
    uint32_t bufferSize; /* Size of resource that device need to allocate when device being client */
    uint32_t sl;         /* service level, range: 0~7, need set as 4 when no use */
    uint32_t tc;         /* traffic class, range: 0~255, need set as 132 when no use */
    char *ipAddr;        /* IP address of device */
} HccnRpingInitAttr;
 
typedef struct HccnRpingTargetInfoDef {
    uint32_t srcPort;              /* udp src port, hash lag needed */
    uint32_t reserved;
    uint32_t sl;                   /* service level, range: 0~7, need set as 4 when no use */
    uint32_t tc;                   /* traffic class, range: 0~255, need set as 132 when no use */
    uint32_t port;                 /* port to connect target */
    uint32_t payloadLen;
    char payload[HCCN_RPING_PAYLOAD_LEN_MAX]; /* user defined payload */
    char *srcIp;                   /* local(client) ip */
    char *dstIp;                   /* remote(target) ip */
} HccnRpingTargetInfo;
 
typedef struct HccnRpingResultInfoDef {
    uint32_t txPkt;       /* send pkt num */
    uint32_t rxPkt;       /* receive pkt num */
    uint32_t minRTT;      /* minimum round-trip time / usec */
    uint32_t maxRTT;      /* maximum round-trip time / usec */
    uint32_t avgRTT;      /* average round-trip time / usec */
    HccnRpingResultState state; /* ping result state: valid | invalid */
    uint32_t reserved[5U];
} HccnRpingResultInfo;

typedef struct HccnRpingTimestampDef {
    uint64_t sec;  /* time sec */
    uint64_t usec; /* time usec */
} HccnRpingTimestamp;

/**
 * @brief struct of every payload header
 */
typedef struct HccnRpingPayloadHeadDef {
    char srcIp[64];        /* local(client) ip */
    char dstIp[64];        /* remote(target) ip */
    uint32_t payloadLen;   /* user defined payload length */
    uint32_t resvd[3];
    HccnRpingTimestamp t1; /* client send timestamp */
    HccnRpingTimestamp t2; /* target recv timestamp */
    HccnRpingTimestamp t3; /* target send timestamp */
    HccnRpingTimestamp t4; /* client recv timestamp */
    uint32_t rpingBatchId; /* batch ping task id */
    uint8_t reserved[44];
} HccnRpingPayloadHead;


/**
 * @brief Init rping resource on a device.
 * @param devLogicId : Device logic ID.
 * @param initAttr: init attribute.
 * @param rpingCtx: context of rping resource.
 * @return HccnResult
 */
extern HccnResult HccnRpingInit(uint32_t devLogicId, HccnRpingInitAttr *initAttr, HccnRpingCtx *rpingCtx);
 
/**
 * @brief Release rping resource on a device.
 * @param rpingCtx: context of rping resource.
 * @return HccnResult
 */
extern HccnResult HccnRpingDeinit(HccnRpingCtx rpingCtx);
 
/**
 * @brief Add targets to client.
 * @param rpingCtx: context of rping resource.
 * @param targetNum: Number of NPUs need probe.
 * @param target: Infoes of NPU need probe, this is an array.
 * @return HccnResult
 */
extern HccnResult HccnRpingAddTarget(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo *target);
 
/**
 * @brief Remove targets from targets.
 * @param rpingCtx: context of rping resource.
 * @param targetNum: Number of NPUs need probe.
 * @param target: Infoes of NPU need probe, this is an array.
 * @return HccnResult
 */
extern HccnResult HccnRpingRemoveTarget(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo *target);
 
/**
 * @brief Get adding target's state.
 * @param rpingCtx: context of rping resource.
 * @param targetNum: Number of NPUs need probe.
 * @param target: Infoes of NPU need probe, this is an array.
 * @param targetState: target state, this is an array.
 * @return HccnResult
 */
extern HccnResult HccnRpingGetTarget(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo *target,
                                     HccnRpingAddTargetState *targetState);
 
/**
 * @brief Start batch ping task.
 * @param rpingCtx: context of rping resource.
 * @param pktNum: Number of packet send to target.
 * @param interval: Interval between two sends of ping packet.
 * @param timeout: Time threshold between ping & pong packet.
 * @return HccnResult
 */
extern HccnResult HccnRpingBatchPingStart(HccnRpingCtx rpingCtx, uint32_t pktNum, uint32_t interval, uint32_t timeout);
 
/**
 * @brief Stop batch ping task.
 * @param rpingCtx: context of rping resource.
 * @return HccnResult
 */
extern HccnResult HccnRpingBatchPingStop(HccnRpingCtx rpingCtx);
 
/**
 * @brief Get batch ping results.
 * @param rpingCtx: context of rping resource.
 * @param targetNum: Number of NPUs need probe.
 * @param target: Infoes of NPU need probe, this is an array.
 * @param result: probe result, this is an array.
 * @return HccnResult
 */
extern HccnResult HccnRpingGetResult(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo *target,
                                     HccnRpingResultInfo *result);

/**
 * @brief Get batch ping packet payload.
 * @param rpingCtx: context of rping resource.
 * @param payload: packet payload pointer, contain all payload, every payload head struct as HccnRpingPayloadHeader.
 * @param payloadLen: length of all payload.
 * @return HccnResult
 */
extern HccnResult HccnRpingGetPayload(HccnRpingCtx rpingCtx, void **payload, uint32_t *payloadLen);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCN_RPING_H_