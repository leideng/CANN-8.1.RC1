/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file kernel_utils_constants.h
 * \brief
 */
#ifndef ASCENDC_MODULE_UTILS_CONSTANTS_H
#define ASCENDC_MODULE_UTILS_CONSTANTS_H
#include "utils/kernel_utils_ceil_oom_que.h"

namespace AscendC {
const int32_t DEFAULT_BLK_NUM = 8;
const int32_t POWER_MASK_NUM = 8;
const int32_t HALF_FACTOR = 2;
const int32_t DOUBLE_FACTOR = 2;
const int32_t DEFAULT_BLK_STRIDE = 1;
const uint8_t DEFAULT_REPEAT_STRIDE = 8;
const uint8_t HALF_DEFAULT_REPEAT_STRIDE = 4;
const uint8_t ONE_FOURTH_DEFAULT_REPEAT_STRIDE = 2;
const uint64_t FULL_MASK = 0xffffffffffffffff;
const uint64_t CONST_MASK_VALUE = 0x8000000000000000;
const uint16_t MAX_HALF_MASK_LEN = 64;
const int32_t DEFAULT_C0_SIZE = 32;
const int32_t DEFAULT_BLOCK_SIZE = 256;
const int32_t MAX_REPEAT_TIMES = 255;
const int32_t MIN_REPEAT_TIMES = 0;
const bool DEFAULT_REPEAT_STRIDE_MODE = 0;
const bool STRIDE_SIZE_MODE = 0;
const int32_t ONE_BYTE_BIT_SIZE = 8;
const int32_t ONE_DUMP_BACKUP_SIZE = 1024;
const int32_t DUMP_UB_SIZE = 256;
const int32_t DUMP_EXC_FLAG = 7;
const uint32_t TOTAL_L0A_SIZE = 64 * 1024;
const uint32_t TOTAL_L0B_SIZE = 64 * 1024;
const uint32_t TMP_UB_SIZE = 8 * 1024;
const uint32_t MAX_SLICE_SIZE = 6 * 256;
const uint32_t F32_INF = 0x7f800000;
const uint32_t F32_NEG_INF = 0xff800000;
const uint32_t F32_NAN = 0x7fc00000;

const uint16_t VALUE_512 = 512;         // align with 512B / value range [0, 512]
const uint16_t UINT12_MAX = 4095;       // 12 bit range is [0, 4095]
const uint16_t UINT15_MAX = 32767;      // 15 bit range is [0, 32767]

// BlockInfo Pos
const uint32_t BLOCK_INFO_LEN_POS = 0;
const uint32_t BLOCK_INFO_CORE_POS = 1;
const uint32_t BLOCK_INFO_BLOCKNUM_POS = 2;
const uint32_t BLOCK_INFO_DUMPOFFSET_POS = 3;
const uint32_t BLOCK_INFO_MAGIC_POS = 4;
const uint32_t BLOCK_INFO_RSV_POS = 5;
const uint32_t BLOCK_INFO_DUMP_ADDR = 6;
const uint32_t BLOCK_INFO_MAGIC_NUM = 0x5aa5bccd;
// DUMP_META Pos 以uint8_t为单位计算位置
const uint32_t DUMP_META_TYPE_POS = 0;
const uint32_t DUMP_META_LEN_POS = 4;
const uint16_t DUMP_META_BLOCK_DIM_POS = 8;
const uint8_t DUMP_META_CORE_TYPE_POS = 10;
const uint8_t DUMP_META_TASK_RATION = 11;
const uint32_t DUMP_META_RSV_POS = 12;
// DumpMessageHead Pos
const uint32_t DUMP_MESSAGE_HEAD_TYPE_POS = 0;
const uint32_t DUMP_MESSAGE_HEAD_LEN_POS = 1;
const uint32_t DUMP_MESSAGE_HEAD_ADDR_POS = 2;
const uint32_t DUMP_MESSAGE_HEAD_DATA_TYPE_POS = 3;
const uint32_t DUMP_MESSAGE_HEAD_DESC_POS = 4;
const uint32_t DUMP_MESSAGE_HEAD_BUFFERID_POS = 5;
const uint32_t DUMP_MESSAGE_HEAD_POSITION_POS = 6;
const uint32_t DUMP_MESSAGE_HEAD_RSV_POS = 7;
const uint32_t DUMP_SCALAR_POS = 8;
const uint32_t DUMP_CORE_COUNT = 75;
const uint32_t DUMP_WORKSPACE_SIZE = DUMP_CORE_COUNT * ONE_CORE_DUMP_SIZE;
// DumpShapeMessageHead Pos
const uint32_t DUMP_SHAPE_MESSAGE_HEAD_TYPE_POS = 0;
const uint32_t DUMP_SHAPE_MESSAGE_HEAD_LEN_POS = 1;
const uint32_t DUMP_SHAPE_MESSAGE_HEAD_DIM_POS = 2;
const uint32_t DUMP_SHAPE_MESSAGE_HEAD_SHAPE_START_POS = 3;
const uint32_t DUMP_SHAPE_MESSAGE_HEAD_RSV_POS = 11;
const uint32_t DUMP_SHAPE_MESSAGE_TL_LEN = 8;
// DumpTimeStamp
const uint32_t DUMP_TIME_STAMP_LEN = 24; // desc_id(uint32_t)+rsv(uint32_t)+cycle(uint64_t)
const uint32_t DUMP_TIME_STAMP_TOTAL_LEN = 32; // 6 * 4
const uint32_t DUMP_TIME_STAMP_LEN_POS = 1;
const uint32_t DUMP_TIME_STAMP_ID_POS = 2;
const uint32_t DUMP_TIME_STAMP_CYCLE_POS = 4;
const uint32_t DUMP_TIME_STAMP_PTR_POS = 6;
// Ctrl bit Pos
constexpr int32_t CTRL_46_BIT = 46;
constexpr int32_t CTRL_47_BIT = 47;
constexpr int32_t CTRL_48_BIT = 48;
constexpr int32_t CTRL_53_BIT = 53;

// power param
constexpr uint32_t TENSOR_TENSOR_FLOAT_POWER_FACTOR = 4;
constexpr uint32_t TENSOR_TENSOR_INT_POWER_FACTOR = 6;
constexpr uint32_t TENSOR_TENSOR_HALF_POWER_FACTOR = 7;
constexpr uint32_t TENSOR_SCALAR_FLOAT_POWER_FACTOR = 5;
constexpr uint32_t TENSOR_SCALAR_INT_POWER_FACTOR = 7;
constexpr uint32_t TENSOR_SCALAR_HALF_POWER_FACTOR = 7;
constexpr uint32_t POWER_TWO = 2;
constexpr uint32_t POWER_THREE = 3;
constexpr uint32_t POWER_INT32_BITS = 32;

// int4b_t param
constexpr uint32_t INT4_TWO = 2;
constexpr uint32_t INT4_BIT_NUM = 4;

// AddDeqRelu param
constexpr int32_t DEQ_SHIFT_LEFT_17_BIT = 131072;
constexpr float DEQ_SHIFT_RIGHT_17_BIT = 1.0 / DEQ_SHIFT_LEFT_17_BIT;
constexpr int8_t ADDDEQRELU_MASK_MODE_ONE = 1;
constexpr int8_t ADDDEQRELU_MASK_MODE_TWO = 2;

#if (__CCE_AICORE__ <= 200)
const int32_t TOTAL_VEC_LOCAL_SIZE = 248 * 1024;
const uint32_t TOTAL_UB_SIZE = 256 * 1024;

const uint32_t TMP_UB_OFFSET = 248 * 1024;
const uint32_t TOTAL_L1_SIZE = 1024 * 1024;
const uint32_t TOTAL_L0C_SIZE = 256 * 1024;
#elif (__CCE_AICORE__ == 220)
const int32_t TOTAL_VEC_LOCAL_SIZE = 184 * 1024;
const uint32_t TOTAL_UB_SIZE = 192 * 1024;
const uint32_t TMP_UB_OFFSET = 184 * 1024;
#ifndef KFC_L1_RESERVER_SIZE
#define KFC_L1_RESERVER_SIZE 128
#endif
const uint32_t TOTAL_L1_SIZE = 512 * 1024 - KFC_L1_RESERVER_SIZE;
const uint32_t SINGLE_MSG_SIZE = 64;
const uint32_t CACHE_LINE_SIZE = 64;
const uint32_t TOTAL_L0C_SIZE = 128 * 1024;
#elif (__CCE_AICORE__ == 300)
const int32_t TOTAL_VEC_LOCAL_SIZE = 184 * 1024;
const uint32_t TOTAL_UB_SIZE = 248 * 1024;
const uint32_t TMP_UB_OFFSET = 248 * 1024;
const uint32_t TOTAL_L1_SIZE = 1024 * 1024;
const uint32_t SINGLE_MSG_SIZE = 64;
const uint32_t CACHE_LINE_SIZE = 64;
const uint32_t TOTAL_L0C_SIZE = 128 * 1024;
const uint32_t VECTOR_REG_WIDTH = 256;
#elif defined(__DAV_M310__)
const int32_t TOTAL_VEC_LOCAL_SIZE = 184 * 1024;
const uint32_t TOTAL_UB_SIZE = 256 * 1024;
const uint32_t TMP_UB_OFFSET = 248 * 1024;
const uint32_t TOTAL_L1_SIZE = 1024 * 1024;
const uint32_t SINGLE_MSG_SIZE = 64;
const uint32_t CACHE_LINE_SIZE = 64;
const uint32_t TOTAL_L0C_SIZE = 128 * 1024;
const uint32_t VECTOR_REG_WIDTH = 256;
#endif
const uint8_t PAD_SIZE = 4;
const uint8_t MRG_SORT_ELEMENT_LEN = 4;
const uint8_t DEFAULT_DATA_COPY_NBURST = 1;
const uint8_t DEFAULT_DATA_COPY_STRIDE = 0;
const int32_t BLOCK_CUBE = 16;
const int32_t CUBE_MAX_SIZE = 256;
const int32_t BYTE_PER_FRACTAL = 512;
const int32_t SRC_BURST_LEN_SIZE_ELE = 16;
const int32_t SRC_GAP_SIZE_BYTE = 32;
const int32_t DST_BURST_LEN_SIZE_ELE = 256;
const int32_t VREDUCE_PER_REP_OUTPUT = 2;
const uint16_t ONE_BLK_SIZE = 32;
const uint16_t ONE_PARAM_SIZE = 8;
const uint16_t AIV_CORE_NUM = 50;
const uint16_t DUMP_MSG_HEAD_SIZE = 24;
const int32_t ONE_REPEAT_BYTE_SIZE = 256;
const int32_t FULL_MASK_LEN = 128;
const int32_t HLAF_MASK_LEN = 64;
const int32_t DEFAULT_REDUCE_DST_REP_SRIDE = 1;
const uint8_t B64_BYTE_SIZE = 8;
const uint8_t B32_BYTE_SIZE = 4;
const uint8_t B16_BYTE_SIZE = 2;
const uint8_t B8_BYTE_SIZE = 1;
const uint8_t B32_DATA_NUM_PER_BLOCK = 8;
const uint8_t B16_DATA_NUM_PER_BLOCK = 16;
const int32_t B16_DATA_NUM_PER_REPEAT = 128;
const int32_t B32_DATA_NUM_PER_REPEAT = 64;
const int32_t BLOCK_STRIDE_POS_IN_SM = 16;
const int32_t PLD_BUFFER_SIZE = 2;
const uint8_t FIXPIPE_DEQ_TENSOR_SIZE = 16;
const uint8_t SET_DATA_EXP_ZERO = 0;
const uint8_t SET_DATA_EXP_ONE = 1;
const uint8_t SET_DATA_EXP_TWO = 2;
const uint8_t SET_DATA_EXP_THREE = 3;
const uint8_t VDEQ_TENSOR_SIZE = 16;
// workspace system reserve 16MB
#if (__CCE_AICORE__ == 100)
constexpr size_t RESERVED_WORKSPACE = 2 * 1024 * 1024;
#elif (__CCE_AICORE__ == 200)
constexpr size_t RESERVED_WORKSPACE = 2 * 1024 * 1024;
#elif (__CCE_AICORE__ == 220)
constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;
#elif (__CCE_AICORE__ == 300)
constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;
#elif defined(__DAV_M310__)
constexpr size_t RESERVED_WORKSPACE = 16 * 1024 * 1024;
#endif
// nchwconv address list size
const int32_t NCHW_CONV_ADDR_LIST_SIZE = 16;
const int32_t VA_REG_ARRAY_LEN = 8;
const uint8_t CONV2D_IMG_SIZE = 2;
const uint8_t CONV2D_KERNEL_SIZE = 2;
const uint8_t CONV2D_STRIDE = 2;
const uint8_t CONV2D_PAD = 4;
const uint8_t CONV2D_DILATION = 2;
const int32_t K_MAX_DIM = 8;

const uint32_t TWO_OF_STACK_BUFFER = 2;
const uint32_t THREE_OF_STACK_BUFFER = 3;
const uint32_t HALF_REPEAT_SIZE = ONE_REPEAT_BYTE_SIZE / B16_BYTE_SIZE;
const uint32_t FLOAT_REPEAT_SIZE = ONE_REPEAT_BYTE_SIZE / B32_BYTE_SIZE;
const uint32_t ONE_REPEAT_FLOAT_SIZE = ONE_REPEAT_BYTE_SIZE / B32_BYTE_SIZE;
const uint32_t ONE_REPEAT_HALF_SIZE = ONE_REPEAT_BYTE_SIZE / B16_BYTE_SIZE;
const uint32_t MAX_REPEAT_FLOAT_SIZE = ONE_REPEAT_FLOAT_SIZE * MAX_REPEAT_TIMES;
const uint32_t MAX_REPEAT_HALF_SIZE = ONE_REPEAT_HALF_SIZE * MAX_REPEAT_TIMES;
const uint32_t ONE_BLK_HALF_NUM = ONE_BLK_SIZE / B16_BYTE_SIZE;
const uint32_t ONE_BLK_FLOAT_NUM = ONE_BLK_SIZE / B32_BYTE_SIZE;
const uint32_t BRCB_BROADCAST_NUMBER = 8;
const uint32_t BRCB_MAX_REPEAT_SIZE = BRCB_BROADCAST_NUMBER * MAX_REPEAT_TIMES;
const int32_t MIN_BLOCK_LEN = 1;
const uint32_t PAIR_REDUCE_REPEAT_STRIDE_LEN = 128;
const uint32_t PAIR_REDUCE_SUM_MERGES = 2;
const uint32_t TWO_HUNDRED_FIFTY_TWO_REPEAT = 252;
const uint32_t TWO_HUNDRED_FIFTY_TWO_REPEAT_BYTE_SIZE = TWO_HUNDRED_FIFTY_TWO_REPEAT * ONE_REPEAT_BYTE_SIZE;
const uint32_t REDUCEV2_MODE_SEVEN = 7;
const uint32_t DROPOUT_MODE_BYTE_MISALIGN = 1;
const uint32_t DROPOUT_MODE_BYTE_ALIGN = 2;
const uint32_t DROPOUT_MODE_BIT_ALIGN = 3;
const uint32_t DROPOUT_MODE_BIT_MISALIGN = 4;
const uint32_t REDUCEV2_MODE_ONE = 1;
const uint32_t REDUCEV2_MODE_TWO = 2;
const uint32_t REDUCEV2_MODE_THREE = 3;

// 4dTrans param size
const int32_t B8_TMP_ELE_LEN = 1024;
const int32_t B16_TMP_ELE_LEN = 256;
const int32_t B32_TMP_ELE_LEN = 128;
const int32_t B8_TRANS_LEN = 1024;
const int32_t B8_TRANS_FRACTAL = 512;
const int32_t B8_TRANS_ROW = 32;
const int32_t B8_COPY_COL = 32;

// load3dPro config
const uint64_t LOAD_M_START_POSITION = 48;
const uint64_t LOAD_K_START_POSITION = 32;
const uint64_t LOAD_M_EXTENSION = 16;
const uint64_t LOAD_DILATION_FILTER_H = 40;
const uint64_t LOAD_DILATION_FILTER_W = 32;
const uint64_t LOAD_FILTER_H = 24;
const uint64_t LOAD_FILTER_W = 16;
const uint64_t LOAD_STRIDE_H = 8;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
// param check size
const int32_t MAX_BLOCK_COUNT = 4095;
const int32_t MIN_BLOCK_COUNT = 1;
const int32_t MAX_BLOCK_LEN = 65535;

const int32_t MAX_16BITS_STRIDE = 65535;
const int32_t MAX_8BITS_STRIDE = 255;
const int32_t MIN_BLOCK_NUM = 1;
const int32_t MAX_PROPOSAL_MODE_NUM = 5;
const int32_t MIN_PROPOSAL_MODE_NUM = 0;

// load2d param size
const int32_t MAX_LOAD2D_START_INDEX = 65535;
const int32_t MIN_LOAD2D_START_INDEX = 0;
const int32_t MAX_LOAD2D_SID = 15;
const int32_t MIN_LOAD2D_SID = 0;

// load3dv1 param size
const int32_t MAX_LOAD3D_PAD = 255;
const int32_t MIN_LOAD3D_PAD = 0;
const int32_t MAX_LOAD3D_L1 = 32767;
const int32_t MIN_LOAD3D_L1 = 1;
const int32_t MAX_LOAD3D_C1_IDX = 4095;
const int32_t MIN_LOAD3D_C1_IDX = 0;
const int32_t MAX_LOAD3D_LEFT_TOP = 32767;
const int32_t MIN_LOAD3D_LEFT_TOP = -255;
const int32_t MAX_LOAD3D_STRIDE = 63;
const int32_t MIN_LOAD3D_STRIDE = 1;
const int32_t MAX_LOAD3D_FILTER = 255;
const int32_t MIN_LOAD3D_FILTER = 1;
const int32_t MIN_LOAD3D_FETCH_FILTER = 0;
const int32_t MAX_LOAD3D_FETCH_FILTER = 254;
const int32_t MIN_LOAD3D_DILATION_FILTER = 1;
const int32_t MAX_LOAD3D_JUMP_STRIDE = 127;
const int32_t MIN_LOAD3D_JUMP_STRIDE = 1;
const int32_t MAX_LOAD3D_REPEAT_MODE = 1;
const int32_t MIN_LOAD3D_REPEAT_MODE = 0;
const int32_t MIN_LOAD3D_REPEAT_TIMES = 1;
const int32_t MAX_LOAD3D_CSIZE = 1;
const int32_t MIN_LOAD3D_CSIZE = 0;

// load3dv2 param size
const int32_t MAX_LOAD3D_CHANNEL_SIZE = 65535;
const int32_t MIN_LOAD3D_CHANNEL_SIZE = 1;
const int32_t MAX_LOAD3D_EXTENSION = 65535;
const int32_t MIN_LOAD3D_EXTENSION = 1;
const int32_t MAX_LOAD3D_START_PT = 65535;
const int32_t MIN_LOAD3D_START_PT = 0;
const int32_t KEXTENSION_HALF = 16;
const int32_t MEXTENSION_HALF = 16;
const int32_t KSTARTPT_HALF = 16;
const int32_t MSTARTPT_HALF = 16;
const int32_t KEXTENSION_B8 = 32;
const int32_t MEXTENSION_B8 = 16;
const int32_t KSTARTPT_B8 = 32;
const int32_t MSTARTPT_B8 = 16;

// loadImageToLocal param size
constexpr int32_t MAX_LOADIMANG_L1_HORSIZE = 4095;
constexpr int32_t MIN_LOADIMANG_L1_HORSIZE = 1;
constexpr int32_t MAX_LOADIMANG_L1_VERSIZE = 4095;
constexpr int32_t MIN_LOADIMANG_L1_VERSIZE = 0;
constexpr int32_t MAX_LOADIMANG_L1_HWSTART = 4095;
constexpr int32_t MIN_LOADIMANG_L1_HWSTART = 0;
constexpr int32_t MAX_LOADIMANG_L1_SHORRES = 65535;
constexpr int32_t MIN_LOADIMANG_L1_SHORRES = 1;
constexpr int32_t MIN_LOADIMANG_L1_PADSIZE = 0;

// mmad param size
const int32_t MAX_M_K_N_SIZE = 4095;
const int32_t MIN_M_K_N_SIZE = 0;

// mrgsort4 param size
const int32_t MAX_SORT_ELE_LEN = 4095;
const int32_t MIN_SORT_ELE_LEN = 0;
const int32_t MIN_SORT_REPEAT_TIMES = 1;

template <typename T> std::string ScalarToString(T scalarValue);
template <> inline std::string ScalarToString(half scalarValue)
{
    return std::to_string(scalarValue.ToFloat());
}
#if __CCE_AICORE__ >= 220 && (!defined(__DAV_M310__))
template <> inline std::string ScalarToString(bfloat16_t scalarValue)
{
    return std::to_string(scalarValue.ToFloat());
}
#endif
template <typename T> uint64_t GetScalarBitcode(T scalarValue);
// deq tensor ptr could not be passed by cce instructions, so pass ptr to model by this function
void SetModelDeqTensor(void* deqTensor);
#if __CCE_AICORE__ == 300
void SetEleSrcPara(uint64_t baseAddr);
#endif
#if __CCE_AICORE__ == 200
void SetVbiSrc0Param(half* vbiSrc0Ptr, int32_t vbiSrc0Size);
void SetUnzipCompressedLen(uint32_t compressedLength);
#endif
void SetModelBiasTensor(void* biasTensor);
void SetIndexMatrix(void* indexMatrix);

// src0 of gatherb instr could not be accessed by cce instructions, so pass ptr to model by this function
void SetModelGatherbSrc0Tensor(uint64_t src0, const uint32_t length);

// dst0 of scatter instr could not be accessed by cce instructions, so pass ptr to model by this function
void SetModelScatterDst0Tensor(uint64_t dst0, const uint32_t length);

int32_t TensorWriteFile(const std::string& fileName, const void* buffer, size_t size);

#endif // ASCENDC_CPU_DEBUG
template <bool condition, class T1, class T2>
struct Conditional {
    using type = T1;
};

template <class T1, class T2>
struct Conditional<false, T1, T2> {
    using type = T2;
};

template <int bitNum, bool sign = true>
struct IntegerSubType {
    static int const kBits = bitNum;
    static bool const kSigned = sign;

    using T = typename Conditional<kSigned, int8_t, uint8_t>::type;
    using Storage = uint8_t;

    static Storage const mask = Storage(((static_cast<uint64_t>(1)) << static_cast<uint32_t>(kBits)) - 1);
    Storage storage;
    __aicore__ inline IntegerSubType() = default;

    __aicore__ inline IntegerSubType(uint32_t value)
        : storage(reinterpret_cast<Storage const &>(value) & mask) {}

    __aicore__ inline IntegerSubType(int32_t value)
        : storage(reinterpret_cast<Storage const &>(value) & mask) {}

    __aicore__ inline operator T() const
    {
        if (kSigned && ((storage & Storage(static_cast<uint64_t>(1) << static_cast<uint32_t>(kBits - 1))) != 0)) {
            // Sign extend
            return T(storage) | ~T(mask);
        }
        return T(storage);
    }

    __aicore__ inline bool operator == (IntegerSubType const &rhs) const
    {
        return storage == rhs.storage;
    }

    __aicore__ inline bool operator != (IntegerSubType const &rhs) const
    {
        return storage != rhs.storage;
    }

    __aicore__ inline bool operator > (IntegerSubType const &rhs) const
    {
        bool lhsIsNeg = (this->storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(this->kBits - 1)));
        bool rhsIsNeg = (rhs.storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(rhs.kBits - 1)));
        if (kSigned && (lhsIsNeg != rhsIsNeg)) {
            return (!lhsIsNeg) && rhsIsNeg;
        }
        return this->storage > rhs.storage;
    }

    __aicore__ inline bool operator >= (IntegerSubType const &rhs) const
    {
        bool lhsIsNeg = (this->storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(this->kBits - 1)));
        bool rhsIsNeg = (rhs.storage & (static_cast<uint64_t>(1) << static_cast<uint32_t>(rhs.kBits - 1)));
        if (kSigned && (lhsIsNeg != rhsIsNeg)) {
            return (!lhsIsNeg) && rhsIsNeg;
        }
        return storage >= rhs.storage;
    }

    __aicore__ inline bool operator < (IntegerSubType const &rhs) const
    {
        return !(*this >= rhs);
    }

    __aicore__ inline bool operator <= (IntegerSubType const &rhs) const
    {
        return !(*this > rhs);
    }
};

using int4b_t = IntegerSubType<INT4_BIT_NUM, true>;

template <typename T> struct SizeOfBits {};
template <>
struct SizeOfBits<int4b_t> {
    static int const value = INT4_BIT_NUM;
};

__aicore__ inline bool CheckCastOverlappingHigh(const uint64_t dstAddr, const uint64_t srcAddr,
    const uint32_t dstTypeSize, const uint32_t srcTypeSize, const uint32_t calCount)
{
    uint64_t addrLow = dstAddr > srcAddr ? srcAddr : dstAddr;
    uint64_t addrHigh = dstAddr > srcAddr ? dstAddr : srcAddr;
    uint64_t needSizeLow = dstAddr > srcAddr ? calCount * srcTypeSize : calCount * dstTypeSize;

    if ((srcTypeSize < dstTypeSize) && (srcAddr >= AlignUp(dstAddr + calCount * srcTypeSize, ONE_BLK_SIZE))) {
        return true;
    }
    if (dstTypeSize > srcTypeSize && srcAddr == dstAddr) {
        return false;
    }
    if ((needSizeLow > static_cast<uint64_t>(ONE_REPEAT_BYTE_SIZE)) && (srcAddr != dstAddr) &&
        ((addrLow + needSizeLow > addrHigh))) {
        return false;
    }
    return true;
}
} // namespace AscendC
#endif // ASCENDC_MODULE_UTILS_CONSTANTS_H