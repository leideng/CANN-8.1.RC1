/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: QOS
 * Author: Huawei Technologies Co., Ltd.
 * Create: 2023-06-29
 * Notes:
 * History:
 */
#ifndef LIBQOSMANAGER_H
#define LIBQOSMANAGER_H

#include <string>
#include <cstdint>
enum class QosErrorCode : int {
    QOS_SUCCESS = 0,
    QOS_UNINIT_ERROR,
    QOS_INIT_ERROR,
    QOS_ILLEGAL_PARA,
    QOS_NOT_FOUND,
    QOS_UNSUPPORTED,
    QOS_DSMI_ERROR,
    QOS_NOMATCH_MPAMID,
    QOS_NULL_FUNC,
};

enum class QosStreamType : int {
    STREAM_FORWARD_COMPUTE = 0,                          // 前向计算
    STREAM_BACKWARD_COMPUTE = 1,                         // 后向计算
    STREAM_PARAMETER_UPDATE = 2,                         // 参数更新
    STREAM_GRADUATION_AGGREGATION = 3,                   // 梯度聚合
    STREAM_HCCL_MODEL_LAY_PARALLEL_FEATURE_MAP = 4,      // 模型层内并行Feature Map通讯
    STREAM_HCCL_MODEL_PIPELINE_PARALLEL_FEATURE_MAP = 5, // Pipeline模型并行Feature Map通讯
    STREAM_HCCL_PARAMETER_PREFETCH = 6,                  // 数据并行参数预取
    STREAM_HCCL_FEATURE_MAP_PREFETCH = 7,                // 数据并行Feature Map预取
    STREAM_HCCL_FEATURE_MAP_SHARE = 8,                   // 数据并行Feature Map共享
    STREAM_HCCL_EMBEDDING_READ_WRITE = 9,                // 数据并行Embedding Table读写
    STREAM_DVPP_COMPUTE = 10,                            // DVPP计算
    STREAM_L2CACHE_PREFETCH = 11,                        // L2 CACHE预取
    STREAM_L2CACHE_INV_WRB_FLUSH = 12,                   // L2 CACHE Invalid/Writeback/Flush操作
    STREAM_AIV_H2D_COPY = 13,                            // 使用AIV从HOST搬移数据到DEVICE
    STREAM_OTHERS,                                       // AI取指令, STARS读SQE, STARS写CQE, 同步通信, SMMU查页表
    STREAM_INVALID,
    STREAM_MAX
};

enum class QosStreamGroupId : uint32_t {
    STREAM_L2CACHE_PREFETCH_GROUP_ID = 0x30000U,             // L2 CACHE预取的parrallel_group_id
    STREAM_L2CACHE_INV_WRB_FLUSH_GROUP_ID = 0x30000U,        // L2 CACHE Invalid/Writeback/Flush操作parrallel_group_id
    STREAM_MAX_GROUP_ID = 0x7F007FU,
    STREAM_DEFAULT_GROUP_ID = 0xFFFFFFFFU                    // 获取默认QoS配置的parrallel_group_id
};

enum class QosMasterType : int {
    MASTER_DVPP_VENC,                                 // DVPP中的VENC加速器
    MASTER_DVPP_VDEC,                                 // DVPP中的VDEC加速器
    MASTER_DVPP_VPC,                                  // DVPP中的VPC加速器
    MASTER_DVPP_JPEGE,                                // DVPP中的JPEGE加速器
    MASTER_DVPP_JPEGD,                                // DVPP中的JPEGD加速器
    MASTER_ROCE,                                      // ROCE引擎
    MASTER_NIC,                                       // NIC
    MASTER_PCIE,                                      // PCIE DMA控制器
    MASTER_AICPU,                                     // AI CPU
    MASTER_AIC_DAT,                                   // AICUBE
    MASTER_AIC_INS,                                   // AIVEC
    MASTER_AIV_DAT,                                   // AICUBE
    MASTER_AIV_INS,                                   // AIVEC
    MASTER_SDMA,                                      // SDMA
    MASTER_STARS,                                     // STARS
    MASTER_ISP_VICAP,
    MASTER_ISP_VIPROC,
    MASTER_ISP_VIPE,
    MASTER_ISP_GDC,
    MASTER_ISP_VGS,
    MASTER_USB,
    MASTER_MAX,                                       // 以此为界, master失效
};

enum QosSqeCfgEntry {
    CONFIG_TYPE_AI_SYS,
    CONFIG_TYPE_HCOM_MODEL_PARALLE,
    CONFIG_TYPE_HCOM_DATA_PARALLE,
    CONFIG_TYPE_HCOM_PIPELIE_PARALLE,
    CONFIG_TYPE_HCOM_OTHERS,
    CONFIG_TYPE_AICPU,
    CONFIG_TYPE_SDMA,
    CONFIG_TYPE_CMO,
    CONFIG_TYPE_BUTT,
};

enum QosEngType {
    ENGINE_AI,
    ENGINE_HCCL,
    ENGINE_AICPU,
    ENGINE_MEMCPYS,
    ENGINE_CMO,
    AI,
    HCCL_T,
    AICPU_T,
    MEMCPYS,
    CMO,
    ENGINE_MAX
};

enum class QosEngineType : int {
    ENGINE_AI,
    ENGINE_HCCL,
    ENGINE_AICPU,
    ENGINE_MEMCPYS,
    ENGINE_CMO,
    AI,
    HCCL,
    AICPU,
    MEMCPYS,
    CMO,
    ENGINE_MAX,
};

struct QosConfig {
    unsigned int mpamId = 0;            // MPAMID ，取值范围： 0~127
    unsigned int bwHigh = 100;          // 带宽的高水线
    unsigned int bwLow = 0;             // 带宽的低水线
    unsigned int qos = 0;               // qos优先级，取值范围：0~7
    unsigned int hardlimit = 0;         // 是否使能hardlimit，1：使能，0：不使能
    unsigned int pmg = 0;
    unsigned int ns = 0;
    unsigned int mode = 0;              // AIC/AIV/SDMA support, 0--reg, 1--smmu, 2--sqe
};

const unsigned int GROUPID_MASK = 0xFFFF0000U;

// 返回Qos功能是否使能
// 未初始化，返回初始化错误，*en为false
QosErrorCode QosIsEnabled(bool* en);

// 根据业务流标签/引擎类型和算子名称，获取该业务流标签对应的MPMAID和Qos优先级
// label: 业务流名称，调优场景下使用
// engine: 引擎名称，默认配置场景下使用
// op: 引擎下对应的算子名称，如HCCl引擎下的算子：allreduce/allgather
// devId: 当前的device id
// info: 以上label、engine、op条件下对应的配置信息
// ret: SUCCESS or others
QosErrorCode QosGetStreamEngineQos(QosStreamType label, QosEngineType engine, const std::string &op,
                                   int devId, QosConfig* info);

#define RT_GET_QOS_ENABLE_STATUS(en) \
        ((rtGetQosEnableStatus != NULL) ? rtGetQosEnableStatus(en) : QosErrorCode::QOS_NULL_FUNC)

#define RT_INIT_QOS_CONFIG() \
        ((rtInitQosConfig != NULL) ? rtInitQosConfig() : QosErrorCode::QOS_NULL_FUNC)

#define RT_GET_QOS_CONFIG(parrallel_group_id, dev_id, info) \
        ((rtGetQosConfig != NULL) ? rtGetQosConfig(parrallel_group_id, dev_id, info) : QosErrorCode::QOS_NULL_FUNC)

#define RT_GET_MASTER_QOS_CONFIG(label, dev_id, info) \
        ((rtGetMasterQosConfig != NULL) ? rtGetMasterQosConfig(label, dev_id, info) : QosErrorCode::QOS_NULL_FUNC)

#define RT_IS_PAR_GID_RESERVED(parallelGroupId, reserved) \
        ((rtIsParGidReserved != NULL) ? rtIsParGidReserved(parallelGroupId, reserved) : QosErrorCode::QOS_NULL_FUNC)

// 对外提供的rt接口：
// 返回Qos功能是否使能
// 未初始化，返回初始化错误，*en为false
__attribute__((weak)) QosErrorCode rtGetQosEnableStatus(bool *status);

// 对外提供的rt接口：
// 初始化QoS,加载xml配置
__attribute__((weak)) QosErrorCode rtInitQosConfig(void);

// 对外提供的rt接口：
// 根据业务流标签获parrallel_group_id, 取对应的MPAMID和QoS优先级
// parrallel_group_id: 业务流标签，高16bit为groupid，低16bit为ruleid
// devId: 当前的device id
// info: 当前目标加速器对应的配置信息
// ret: SUCCESS or others
__attribute__((weak)) QosErrorCode rtGetQosConfig(uint32_t parrallelGroupId, int32_t devId, QosConfig* info);

// 对外提供的rt接口：
// 根据加速器类型，获取该加速器对应的MPMAID和Qos优先级
// label: 加速器类型
// devId: 当前的device id
// info: 当前目标加速器对应的配置信息
// ret: SUCCESS or others
__attribute__((weak)) QosErrorCode rtGetMasterQosConfig(QosMasterType label, int devId, struct QosConfig* info);

// 对外提供的rt接口：
// 判断入参parallelGroupId是否为预留id
// parallelGroupId: 业务流标签，高16bit为groupid，低16bit为ruleid
// reserved：是否为预留的groupid, true: 是预留id, false: 不是预留id
// ret: SUCCESS or others
__attribute__((weak)) QosErrorCode rtIsParGidReserved(uint32_t parallelGroupId, bool* reserved);
#endif
