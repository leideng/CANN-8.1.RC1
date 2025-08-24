//
// Created by zhouqilong on 22/3/2023.
//

#ifndef OP_API_COMMON_INC_OPDEV_DATA_TYPE_UTILS_H_
#define OP_API_COMMON_INC_OPDEV_DATA_TYPE_UTILS_H_

#include <algorithm>
#include <complex>
#include <string>
#include <acl/acl_base.h>
#include "common_types.h"
#include "op_common/data_type_utils.h"

namespace op {
using opcommon::IsComplexType;
using opcommon::IsFloatingType;
using opcommon::IsIntegralType;
using opcommon::PromoteType;
using opcommon::CanCast;

constexpr uint32_t kDataTypeSizeBitOffset = 1000;

inline bool CheckType(const DataType dtype,
                      const std::initializer_list<DataType> &valid_types)
{
    return std::find(valid_types.begin(), valid_types.end(), dtype) != valid_types.end();
}

inline bool IsBasicType(const DataType dtype)
{
    return CheckType(dtype,
                     {DataType::DT_COMPLEX128, DataType::DT_COMPLEX64, DataType::DT_DOUBLE, DataType::DT_FLOAT,
                      DataType::DT_FLOAT16, DataType::DT_INT16, DataType::DT_INT32, DataType::DT_INT64,
                      DataType::DT_INT8, DataType::DT_QINT16, DataType::DT_QINT32, DataType::DT_QINT8,
                      DataType::DT_QUINT16, DataType::DT_QUINT8, DataType::DT_UINT16, DataType::DT_UINT32,
                      DataType::DT_UINT64, DataType::DT_UINT8, DataType::DT_BF16});
}

inline bool IsNumberType(const DataType dtype)
{
    return CheckType(dtype,
                     {DataType::DT_COMPLEX128, DataType::DT_COMPLEX64, DataType::DT_DOUBLE, DataType::DT_FLOAT,
                      DataType::DT_FLOAT16, DataType::DT_INT16, DataType::DT_INT32, DataType::DT_INT64,
                      DataType::DT_INT8, DataType::DT_QINT32, DataType::DT_QINT8, DataType::DT_QUINT8,
                      DataType::DT_UINT16, DataType::DT_UINT32, DataType::DT_UINT64, DataType::DT_UINT8,
                      DataType::DT_BF16});
}

inline bool IsRealNumberType(const DataType dtype)
{
    return CheckType(dtype,
                     {DataType::DT_DOUBLE, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT16,
                      DataType::DT_INT32, DataType::DT_INT64,
                      DataType::DT_INT8, DataType::DT_UINT16, DataType::DT_UINT32, DataType::DT_UINT64,
                      DataType::DT_UINT8, DataType::DT_BF16});
}

inline size_t TypeSize(DataType dataType)
{
    static int data_type_size[DataType::DT_MAX] = {
        4, // DT_FLOAT = 0,               float type
        2, // DT_FLOAT16 = 1,             fp16 type
        1, // DT_INT8 = 2,                int8 type
        4, // DT_INT32 = 3,
        1, // DT_UINT8 = 4,               uint8 type
        -1,
        2,                          // DT_INT16 = 6,               int16 type
        2,                          // DT_UINT16 = 7,              uint16 type
        4,                          // DT_UINT32 = 8,              unsigned int32
        8,                          // DT_INT64 = 9,               int64 type
        8,                          // DT_UINT64 = 10,             unsigned int64
        8,                          // DT_DOUBLE = 11,             double type
        1,                          // DT_BOOL = 12,               bool type
        -1,                         // DT_STRING = 13,             string type
        1,                          // DT_DUAL_SUB_INT8 = 14,      dual output int8 type
        1,                          // DT_DUAL_SUB_UINT8 = 15,     dual output uint8 type
        8,                          // DT_COMPLEX64 = 16,          complex64 type
        16,                         // DT_COMPLEX128 = 17,         complex128 type
        1,                          // DT_QINT8 = 18,              qint8 type
        2,                          // DT_QINT16 = 19,             qint16 type
        4,                          // DT_QINT32 = 20,             qint32 type
        1,                          // DT_QUINT8 = 21,             quint8 type
        2,                          // DT_QUINT16 = 22,            quint16 type
        8,                          // DT_RESOURCE = 23,           resource type
        -1,                         // DT_STRING_REF = 24,         string ref type
        5,                          // DT_DUAL = 25,               dual output type (float + int8)
        8,                          // DT_VARIANT                  variant type
        2,                          // DT_BF16 = 27,               bf16 type
        -1,                         // DT_UNDEFINED = 28           Used to indicate a DataType field has not been set.
        kDataTypeSizeBitOffset + 4, // DT_INT4 = 29,             int4 type
        kDataTypeSizeBitOffset + 1, // DT_UINT1 = 30,            uint1 type
        kDataTypeSizeBitOffset + 2, // DT_INT2 = 31,             int2 type
        kDataTypeSizeBitOffset + 2, // DT_UINT2 = 32,            uint2 type
        // DT_MAX
    };
    if (dataType < DataType::DT_FLOAT || dataType >= DataType::DT_MAX || data_type_size[dataType] <= 0) {
        return sizeof(complex128);
    }
    return static_cast<size_t>(data_type_size[dataType]);
}

DataType ToDataType(const std::string &dataTypeStr);
constexpr inline DataType ToOpDataType(aclDataType type)
{
    if (type != aclDataType::ACL_DT_UNDEFINED) {
        return static_cast<DataType>(type);
    }

    return DataType::DT_UNDEFINED;
}

inline aclDataType ToAclDataType(DataType type)
{
    static const std::vector<DataType> CAN_CONVERT_TO_ACL_DataType_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                           DataType::DT_INT8, DataType::DT_INT32,
                                                                           DataType::DT_UINT8, DataType::DT_INT16,
                                                                           DataType::DT_UINT16, DataType::DT_UINT32,
                                                                           DataType::DT_INT64, DataType::DT_DOUBLE,
                                                                           DataType::DT_BOOL, DataType::DT_STRING,
                                                                           DataType::DT_COMPLEX64,
                                                                           DataType::DT_COMPLEX128,
                                                                           DataType::DT_BF16, DataType::DT_UINT64,
                                                                           DataType::DT_INT4};
    auto iter = std::find(CAN_CONVERT_TO_ACL_DataType_LIST.begin(),
                          CAN_CONVERT_TO_ACL_DataType_LIST.end(),
                          type);
    if (iter == CAN_CONVERT_TO_ACL_DataType_LIST.end()) {
        return aclDataType::ACL_DT_UNDEFINED;
    }

    return static_cast<aclDataType>(type);
}

ge::AscendString ToString(const std::initializer_list<DataType> &dataTypes);
int64_t CalcShapeBytes(int64_t size, DataType dataType, bool ceil = false);
} //  namespace op

#endif // OP_API_COMMON_INC_OPDEV_DATA_TYPE_UTILS_H_
