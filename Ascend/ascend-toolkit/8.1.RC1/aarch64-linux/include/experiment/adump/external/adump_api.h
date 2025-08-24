/**
 * @file adump_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.\n
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n
 *
 * 描述：算子dump接口头文件。\n
 */

/* * @defgroup dump dump接口 */
#ifndef ADUMP_API_H
#define ADUMP_API_H
#include <cstdint>
#include <string>
#include <map>
#include <vector>
#include "runtime/base.h"
#include "exe_graph/runtime/tensor.h"
#include "toolchain/prof_common.h"

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define ADX_API __declspec(dllexport)
#else
#define ADX_API __attribute__((visibility("default")))
#endif
namespace Adx {
constexpr int32_t ADUMP_SUCCESS = 0;
constexpr int32_t ADUMP_FAILED = -1;
constexpr uint32_t ADUMP_ARGS_EXCEPTION_HEAD = 2;

// AdumpGetSizeInfoAddr chunk size parameter
constexpr uint32_t RING_CHUNK_SIZE = 60000;
constexpr uint32_t MAX_TENSOR_NUM = 1000;

// AdumpGetDFXInfoAddr chunk size parameter
extern uint64_t *g_dynamicChunk;
extern uint64_t *g_staticChunk;
constexpr uint32_t DYNAMIC_RING_CHUNK_SIZE = 393216;  // 393216 * 8 = 3M
constexpr uint32_t STATIC_RING_CHUNK_SIZE = 131072;  // 131072 * 8 = 1M
constexpr uint32_t DFX_MAX_TENSOR_NUM = 4000;
constexpr uint16_t RESERVE_SPACE = 2;

enum class DumpType : int32_t {
    OPERATOR = 0x01,
    EXCEPTION = 0x02,
    ARGS_EXCEPTION = 0x03,
    OP_OVERFLOW = 0x04,
    AIC_ERR_DETAIL_DUMP = 0x05 // COREDUMP mode
};

// dumpSwitch bitmap
constexpr uint64_t OPERATOR_OP_DUMP = 1U << 0;
constexpr uint64_t OPERATOR_KERNEL_DUMP = 1U << 1;

struct DumpConfig {
    std::string dumpPath;
    std::string dumpMode;   // input/output/workspace
    std::string dumpStatus; // on/off
    std::string dumpData;   // tensor/stats
    uint64_t dumpSwitch{ OPERATOR_OP_DUMP | OPERATOR_KERNEL_DUMP };
    std::vector<std::string> dumpStatsItem;
};

/**
 * @ingroup dump
 * @par 描述: dump 开关状态(flag)查询
 *
 * @attention 无
 * @param  dumpType [IN] dump 类型（operator, exception）
 * @retval #false dump 开关状态(flag) off
 * @retval #true dump 开关状态(flag) on
 * @see 无
 * @since
 */
ADX_API bool AdumpIsDumpEnable(DumpType dumpType);

/**
 * @ingroup dump
 * @par 描述: dump 开关状态(flag)查询
 *
 * @attention 无
 * @param  dumpType [IN] dump 类型（operator, exception）
 * @param  dumpType [OUT] dumpSwitch开关
 * @retval #false dump 开关状态(flag) off
 * @retval #true dump 开关状态(flag) on
 * @see 无
 * @since
 */
ADX_API bool AdumpIsDumpEnable(DumpType dumpType, uint64_t &dumpSwitch);

/**
 * @ingroup dump
 * @par 描述: dump 开关设置
 *
 * @attention 无
 * @param  dumpType [IN] dump 类型（operator, exception）
 * @param  flag [IN] dump开关状态, 0: off, !0 on
 * @retval #0 dump 开关设置成功
 * @retval #!0 dump 开关设置失败
 * @see 无
 * @since
 */
ADX_API int32_t AdumpSetDumpConfig(DumpType dumpType, const DumpConfig &dumpConfig);

enum class TensorType : int32_t {
    INPUT,
    OUTPUT,
    WORKSPACE
};

enum class AddressType : int32_t {
    TRADITIONAL,
    NOTILING,
    RAW
};

struct TensorInfo {
    gert::Tensor *tensor;
    TensorType type;
    AddressType addrType;
    uint32_t argsOffSet;
};

/**
 * @ingroup dump
 * @par 描述: dump tensor
 *
 * @attention 无
 * @param  opType [IN] 算子类型
 * @param  opName [IN] 算子名称
 * @param  tensors [IN] 算子tensor信息
 * @param  stream [IN] 算子处理流句柄
 * @retval #0 dump tensor成功
 * @retval #!0 dump tensor失败
 * @see 无
 * @since
 */
ADX_API int32_t AdumpDumpTensor(const std::string &opType, const std::string &opName,
    const std::vector<TensorInfo> &tensors, rtStream_t stream);

constexpr char DUMP_ADDITIONAL_BLOCK_DIM[] = "block_dim";
constexpr char DUMP_ADDITIONAL_TILING_KEY[] = "tiling_key";
constexpr char DUMP_ADDITIONAL_TILING_DATA[] = "tiling_data";
constexpr char DUMP_ADDITIONAL_IMPLY_TYPE[] = "imply_type";
constexpr char DUMP_ADDITIONAL_ALL_ATTRS[] = "all_attrs";
constexpr char DUMP_ADDITIONAL_IS_MEM_LOG[] = "is_mem_log";
constexpr char DUMP_ADDITIONAL_IS_HOST_ARGS[] = "is_host_args";
constexpr char DUMP_ADDITIONAL_NODE_INFO[] = "node_info";
constexpr char DUMP_ADDITIONAL_DEV_FUNC[] = "dev_func";
constexpr char DUMP_ADDITIONAL_TVM_MAGIC[] = "tvm_magic";
constexpr char DUMP_ADDITIONAL_OP_FILE_PATH[] = "op_file_path";
constexpr char DUMP_ADDITIONAL_KERNEL_INFO[] = "kernel_info";
constexpr char DUMP_ADDITIONAL_WORKSPACE_BYTES[] = "workspace_bytes";
constexpr char DUMP_ADDITIONAL_WORKSPACE_ADDRS[] = "workspace_addrs";

constexpr char DEVICE_INFO_NAME_ARGS[] = "args before execute";

struct DeviceInfo {
    std::string name;
    void *addr;
    uint64_t length;
};

struct OperatorInfo {
    bool agingFlag{ true };
    uint32_t taskId{ 0U };
    uint32_t streamId{ 0U };
    uint32_t deviceId{ 0U };
    uint32_t contextId{ UINT32_MAX };
    std::string opType;
    std::string opName;
    std::vector<TensorInfo> tensorInfos;
    std::vector<DeviceInfo> deviceInfos;
    std::map<std::string, std::string> additionalInfo;
};

/**
 * @ingroup dump
 * @par 描述: 保存异常需要Dump的算子信息。
 *
 * @attention 无
 * @param  OperatorInfo [IN] 算子信息
 * @retval #0 保存成功
 * @retval #!0 保存失败
 * @see 无
 * @since
 */
extern "C" ADX_API int32_t AdumpAddExceptionOperatorInfo(const OperatorInfo &opInfo);

/**
 * @ingroup dump
 * @par 描述: 模型卸载时，删除异常需要Dump的算子信息。
 *
 * @attention 无
 * @param  deviceId [IN] 设备逻辑id
 * @param  streamId [IN] 执行流id
 * @retval #0 保存成功
 * @retval #!0 保存失败
 * @see 无
 * @since
 */
extern "C" ADX_API int32_t AdumpDelExceptionOperatorInfo(uint32_t deviceId, uint32_t streamId);

/**
 * @ingroup dump
 * @par 描述: 获取异常算子需要Dump的信息空间。
 *
 * @attention 无
 * @param  uint32_t space [IN] 待获取space大小
 * @param  uint64_t &atomicIndex [OUT] 返回获取space地址的index参数
 * @retval # nullptr 地址信息获取失败
 * @retval # !nullptr 地址信息获取成功
 * @see 无
 * @since
 */
extern "C" ADX_API void *AdumpGetSizeInfoAddr(uint32_t space, uint32_t &atomicIndex);

/**
 * @ingroup dump
 * @par 描述: 获取动态shape异常算子需要Dump的size信息空间。
 *
 * @attention 无
 * @param  uint32_t space [IN] 待获取space大小
 * @param  uint64_t &atomicIndex [OUT] 返回获取space地址的index参数
 * @retval # nullptr 地址信息获取失败
 * @retval # !nullptr 地址信息获取成功
 * @see 无
 * @since
 */
extern "C" ADX_API void *AdumpGetDFXInfoAddrForDynamic(uint32_t space, uint64_t &atomicIndex);

/**
 * @ingroup dump
 * @par 描述: 获取静态shape异常算子需要Dump的size信息空间。
 *
 * @attention 无
 * @param  uint32_t space [IN] 待获取space大小
 * @param  uint64_t &atomicIndex [OUT] 返回获取space地址的index参数
 * @retval # nullptr 地址信息获取失败
 * @retval # !nullptr 地址信息获取成功
 * @see 无
 * @since
 */
extern "C" ADX_API void *AdumpGetDFXInfoAddrForStatic(uint32_t space, uint64_t &atomicIndex);

/**
 * @ingroup dump
 * @par 描述: 打印workspace内存中的print信息
 *
 * @attention 无
 * @param  void* workspace首地址
 * @param  size_t dumpWorkSpaceSize workspace大小
 * @param  rtStream_t stream 流信息
 * @param  char *opType 算子类型
 * @see 无
 * @since
 */
ADX_API void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                                 rtStream_t stream, const char *opType);

ADX_API void AdumpPrintAndGetTimeStampInfo(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
    rtStream_t stream, const char *opType, std::vector<MsprofAicTimeStampInfo> &timeStampInfo);

struct AdumpPrintConfig{
   bool printEnable;
};
ADX_API void AdumpPrintSetConfig(const AdumpPrintConfig &config);
} // namespace Adx
#endif
