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
 * \file kernel_operator_dump_tensor_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_DUMP_TENSOR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_DUMP_TENSOR_IMPL_H

#include "kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_operator_common_impl.h"
#include "kernel_tpipe_impl.h"
#include "dav_m200/kernel_operator_data_copy_impl.h"
#include "kernel_pop_stack_buffer.h"

namespace AscendC {
/* **************************************************************************************************
 * DumpTensorImpl                                             *
 * ************************************************************************************************* */
__BLOCK_LOCAL__ __inline__ __gm__ uint8_t* g_dumpWorkspaceReserved;

template <typename T> __aicore__ inline uint32_t GetDataType(T data)
{
    uint32_t type;

    if (IsSameType<T, uint8_t>::value) {
        return DT_UINT8;
    } else if (IsSameType<T, int8_t>::value) {
        return DT_INT8;
    } else if (IsSameType<T, int16_t>::value) {
        return DT_INT16;
    } else if (IsSameType<T, uint16_t>::value) {
        return DT_UINT16;
    } else if (IsSameType<T, int32_t>::value) {
        return DT_INT32;
    } else if (IsSameType<T, uint32_t>::value) {
        return DT_UINT32;
    } else if (IsSameType<T, uint64_t>::value) {
        return DT_UINT64;
    } else if (IsSameType<T, int64_t>::value) {
        return DT_INT64;
    } else if (IsSameType<T, float>::value) {
        return DT_FLOAT;
    } else if (IsSameType<T, half>::value) {
        return DT_FLOAT16;
    } else {
        return DT_MAX;
    }
}

__aicore__ inline DataCopyParams GetDataCopyParamImpl(uint32_t offset)
{
    DataCopyParams repeatParams;
    repeatParams.blockCount = 1;
    repeatParams.blockLen = offset / ONE_BLK_SIZE;
    repeatParams.srcStride = 0;
    repeatParams.dstStride = 0;
    return repeatParams;
}

__aicore__ __gm__ inline BlockInfo *GetBlockInfo()
{
#ifdef __ENABLE_VECTOR_CORE__
#if defined(__DAV_M200_VEC__)
    uint32_t core = GetBlockIdxImpl() - get_data_main_base();
#else
    uint32_t core = GetBlockIdxImpl() - get_data_main_base() + AIV_CORE_NUM;
#endif
#else
    uint8_t core = GetBlockIdxImpl() + AIV_CORE_NUM;
#endif
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;
    __gm__ BlockInfo *blockInfo = (__gm__ BlockInfo *)(dumpWorkspaceStart +  DUMP_UINTSIZE * core);
    return blockInfo;
}

__aicore__ inline void UpdateBlockInfo( uint32_t tlvSize, uint32_t excSize)
{
    __gm__ BlockInfo *blockInfo = GetBlockInfo();
    __gm__ uint8_t *blockInfoStart = (__gm__ uint8_t *)blockInfo;
    dcci((__gm__ uint64_t *)blockInfoStart, cache_line_t::ENTIRE_DATA_CACHE);
    uint32_t remainSize = blockInfo->dumpOffset;
    uint64_t lastDumpAddr = blockInfo->dumpAddr;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_LEN_POS) = blockInfo->len;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_CORE_POS) = blockInfo->core;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_BLOCKNUM_POS) = blockInfo->blockNum;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_DUMPOFFSET_POS) = remainSize - tlvSize;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_MAGIC_POS) = 0x5aa5bccd;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_RSV_POS) = excSize;
    *((__gm__ uint64_t *)((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_DUMP_ADDR)) = lastDumpAddr + tlvSize;

    PipeBarrier<PIPE_ALL>();
    dcci((__gm__ uint64_t *)blockInfoStart, cache_line_t::ENTIRE_DATA_CACHE);
}

template <typename T>
__aicore__ inline void InitTmpTensor(LocalTensor<T> &tmpLocal, uint8_t quePos)
{
    TBuffAddr tbuf_tmpLocal;
    tbuf_tmpLocal.logicPos = quePos;
    tmpLocal.SetAddr(tbuf_tmpLocal);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    tmpLocal.address_.absAddr = reinterpret_cast<uint8_t *>(ConstDefiner::Instance().cpuUB);
#else
    tmpLocal.address_.bufferAddr = get_imm(0);
#endif
    tmpLocal.address_.dataLen = DUMP_UB_SIZE;
}

__aicore__ inline uint32_t GetdataCopyCount(uint32_t dataLen)
{
    if (dataLen % DUMP_UB_SIZE == 0) {
        return dataLen / DUMP_UB_SIZE;
    }
    return dataLen / DUMP_UB_SIZE + 1;
}

__aicore__ inline int64_t GetBlockNum();
__aicore__ inline void InitDumpImpl(bool mixFlag, uint32_t gmLen)
{
    if (g_dumpWorkspaceReserved == nullptr) {
        ASCENDC_ASSERT((false),
            { KERNEL_LOG(KERNEL_ERROR, "init dump get nullptr system workspace ptr"); });
        return;
    }
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;
    uint32_t totalBlockNum = GetBlockNum();
    uint32_t blockDumpSize = DUMP_UINTSIZE; // DUMP_UINTSIZE is 1M
#ifdef __ENABLE_VECTOR_CORE__
#if defined(__DAV_M200_VEC__)
    uint32_t blockDim = GetBlockIdxImpl() - get_data_main_base();
#else
    uint32_t blockDim = GetBlockIdxImpl() - get_data_main_base() + AIV_CORE_NUM;
#endif
#else
    uint32_t blockDim = GetBlockIdxImpl() + AIV_CORE_NUM;
#endif
    if (blockDim >= DUMP_CORE_COUNT) {
        return;
    }
    uint32_t blkInfoLen = sizeof(BlockInfo);
    uint64_t blockInfoStart = dumpWorkspaceStart + blockDim * DUMP_UINTSIZE;
    PipeBarrier<PIPE_ALL>();
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_LEN_POS) = blockDumpSize;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_CORE_POS) = blockDim;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_BLOCKNUM_POS) = totalBlockNum;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_DUMPOFFSET_POS) = blockDumpSize - blkInfoLen - sizeof(DumpMeta);
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_MAGIC_POS) = 0x5aa5bccd;
    *((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_RSV_POS) = 0;
    *((__gm__ uint64_t *)((__gm__ uint32_t *)blockInfoStart + BLOCK_INFO_DUMP_ADDR)) = blockInfoStart + blkInfoLen + sizeof(DumpMeta);
    PipeBarrier<PIPE_ALL>();
    dcci((__gm__ uint64_t *)blockInfoStart, cache_line_t::ENTIRE_DATA_CACHE);
    // write DUM_META message
    uint32_t coreType = 1;
#if defined(__DAV_M200_VEC__)
    coreType = 2;
#endif
    blockInfoStart = blockInfoStart + sizeof(BlockInfo);
    *(__gm__ uint32_t*)((__gm__ uint8_t*)blockInfoStart + DUMP_META_TYPE_POS) =
        static_cast<uint32_t>(DumpType::DUMP_META);
    *(__gm__ uint32_t*)((__gm__ uint8_t*)blockInfoStart + DUMP_META_LEN_POS) = 8;
    *(__gm__ uint16_t*)((__gm__ uint8_t*)blockInfoStart + DUMP_META_BLOCK_DIM_POS) =
        static_cast<uint16_t>(GetBlockNum());
    *(__gm__ uint8_t*)((__gm__ uint8_t*)blockInfoStart + DUMP_META_CORE_TYPE_POS) =
        static_cast<uint8_t>(coreType);
    *(__gm__ uint8_t*)((__gm__ uint8_t*)blockInfoStart + DUMP_META_TASK_RATION) =
        static_cast<uint8_t>(GetTaskRationImpl());
    *((__gm__ uint32_t*)blockInfoStart + DUMP_META_RSV_POS) = 0;
    PipeBarrier<PIPE_ALL>();
    dcci((__gm__ uint64_t *)blockInfoStart, cache_line_t::ENTIRE_DATA_CACHE);
}

template <typename T>
__aicore__ inline uint32_t CheckValidPosition(const LocalTensor<T> &tensor)
{
    uint32_t position = 0;
    if ((Hardware)GetPhyType((TPosition)tensor.GetPosition()) == Hardware::UB) {
        position = static_cast<uint32_t>(AscendC::Hardware::UB);
        return position;
    } else if ((Hardware)GetPhyType((TPosition)tensor.GetPosition()) == Hardware::L1) {
        position = static_cast<uint32_t>(AscendC::Hardware::L1);
        return position;
    } else if ((Hardware)GetPhyType((TPosition)tensor.GetPosition()) == Hardware::L0C) {
        position = static_cast<uint32_t>(AscendC::Hardware::L0C);
        return position;
    }
    return position;
}

__aicore__ inline __gm__ BlockInfo *GetCurCoreHeadPtr()
{
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;
#ifdef __ENABLE_VECTOR_CORE__
#if defined(__DAV_M200_VEC__)
    uint32_t core = GetBlockIdxImpl() - get_data_main_base();
#else
    uint32_t core = GetBlockIdxImpl() - get_data_main_base() + AIV_CORE_NUM;
#endif
#else
    uint8_t core = GetBlockIdxImpl() + AIV_CORE_NUM;
#endif
    __gm__ BlockInfo *coreHeadPtr = (__gm__ BlockInfo *)(dumpWorkspaceStart + DUMP_UINTSIZE * core);
    return coreHeadPtr;
}

__aicore__ inline void WriteHeadMsg(DumpMessageHead &dumpMsg)
{
    __gm__ BlockInfo *coreHeadPtr = GetCurCoreHeadPtr();
    *((__gm__ uint32_t *)coreHeadPtr->dumpAddr + DUMP_MESSAGE_HEAD_TYPE_POS) = dumpMsg.type; // static_cast<uint32_t>(DumpType::DUMP_TENSOR);
    *((__gm__ uint32_t *)coreHeadPtr->dumpAddr + DUMP_MESSAGE_HEAD_LEN_POS) = dumpMsg.lenth; // DUMP_MSG_HEAD_SIZE + dumpSize * sizeof(T);
    *((__gm__ uint32_t *)coreHeadPtr->dumpAddr + DUMP_MESSAGE_HEAD_ADDR_POS) = dumpMsg.addr; // static_cast<uint32_t>(tensorAddr);
    *((__gm__ uint32_t *)coreHeadPtr->dumpAddr + DUMP_MESSAGE_HEAD_DATA_TYPE_POS) = dumpMsg.dataType; //GetDataType(data);
    *((__gm__ uint32_t *)coreHeadPtr->dumpAddr + DUMP_MESSAGE_HEAD_DESC_POS) = dumpMsg.desc;
    *((__gm__ uint32_t *)coreHeadPtr->dumpAddr + DUMP_MESSAGE_HEAD_BUFFERID_POS) = dumpMsg.bufferId;
    *((__gm__ uint32_t *)coreHeadPtr->dumpAddr + DUMP_MESSAGE_HEAD_POSITION_POS) = dumpMsg.position;
    *((__gm__ uint32_t *)coreHeadPtr->dumpAddr + DUMP_MESSAGE_HEAD_RSV_POS) = dumpMsg.rsv;
    PipeBarrier<PIPE_ALL>();
    dcci((__gm__ uint64_t *)coreHeadPtr->dumpAddr, cache_line_t::ENTIRE_DATA_CACHE);
    UpdateBlockInfo(sizeof(DumpMessageHead), 0);
    return;
}

template <typename T>
__aicore__ inline void TensorDataLoopCopy(LocalTensor<uint8_t> &tmpLocal,
                                          const LocalTensor<T> &tensor,
                                          uint32_t copyLen)
{
    uint32_t loopCount = GetdataCopyCount(copyLen);
    DataCopyParams backupParams = GetDataCopyParamImpl(DUMP_UB_SIZE);
    __gm__ BlockInfo *coreHeadPtr = GetCurCoreHeadPtr();
    const Hardware srcHWPos = GetPhyType((TPosition)tensor.GetPosition());
    for (uint32_t i = 0; i < loopCount; i++) {
        if (srcHWPos == Hardware::L1) {
            PipeBarrier<PIPE_ALL>();
            DataCopyL12UBImpl((__ubuf__ uint8_t *)tmpLocal.GetPhyAddr(),
                              (__cbuf__ uint8_t *)tensor.GetPhyAddr() + i * DUMP_UB_SIZE,
                              backupParams); // L1 to UB
            PipeBarrier<PIPE_ALL>();
            DataCopyUB2GMImpl((__gm__ uint8_t *)(coreHeadPtr->dumpAddr + i * DUMP_UB_SIZE),
                              (__ubuf__ uint8_t *)tmpLocal.GetPhyAddr(),
                              backupParams); // UB to GM
            PipeBarrier<PIPE_ALL>();
        } else if (srcHWPos == Hardware::L0C) {
            DataCopyEnhancedParams enhancedParams;
            PipeBarrier<PIPE_ALL>();
            DataCopyL0C2UBImpl((__ubuf__ uint8_t *)tmpLocal.GetPhyAddr(),
                               (__cc__ int32_t *)((__cc__ uint8_t *)tensor.GetPhyAddr() + i * DUMP_UB_SIZE),
                               backupParams,
                               enhancedParams); // L0C to UB
            PipeBarrier<PIPE_ALL>();
            DataCopyUB2GMImpl((__gm__ uint8_t *)(coreHeadPtr->dumpAddr + i * DUMP_UB_SIZE),
                              (__ubuf__ uint8_t *)tmpLocal.GetPhyAddr() + i * DUMP_UB_SIZE,
                              backupParams); // UB to GM
            PipeBarrier<PIPE_ALL>();
        }
    }
    dcci((__gm__ uint64_t *)coreHeadPtr->dumpAddr, cache_line_t::ENTIRE_DATA_CACHE);
}
template <typename T>
__aicore__ inline void TensorDataCopy(const LocalTensor<T> &tensor, uint32_t copyLen)
{
    __gm__ BlockInfo *coreHeadPtr = GetCurCoreHeadPtr();
    DataCopyParams repeatParams = GetDataCopyParamImpl(copyLen);
    PipeBarrier<PIPE_ALL>();
    DataCopyUB2GMImpl((__gm__ T *)(coreHeadPtr->dumpAddr), (__ubuf__ T *)tensor.GetPhyAddr(), repeatParams); // UB to GM
    PipeBarrier<PIPE_ALL>();
    dcci((__gm__ uint64_t *)coreHeadPtr->dumpAddr, cache_line_t::ENTIRE_DATA_CACHE);
}

template <typename T>
__aicore__ inline void TensorDataLoopCopy(LocalTensor<uint8_t> &tmpLocal,
                                          const GlobalTensor<T> &tensor,
                                          uint32_t copyLen)

{
    dcci((__gm__ uint64_t *)tensor.GetPhyAddr(), cache_line_t::ENTIRE_DATA_CACHE);
    __gm__ BlockInfo *coreHeadPtr = GetCurCoreHeadPtr();
    DataCopyParams backupParams = GetDataCopyParamImpl(DUMP_UB_SIZE);
    uint32_t loopCount = GetdataCopyCount(copyLen);
    for (uint32_t i = 0; i < loopCount; i++) {
        PipeBarrier<PIPE_ALL>();
        DataCopyGM2UBImpl((__ubuf__ uint32_t *)tmpLocal.GetPhyAddr(),
                          (__gm__ uint32_t *)((__gm__ uint8_t *)tensor.GetPhyAddr() + i * DUMP_UB_SIZE),
                          backupParams); // GM to UB
        PipeBarrier<PIPE_ALL>();
        DataCopyUB2GMImpl((__gm__ uint8_t *)(coreHeadPtr->dumpAddr + i * DUMP_UB_SIZE),
                          (__ubuf__ uint8_t *)tmpLocal.GetPhyAddr(),
                          backupParams); // UB to GM
        PipeBarrier<PIPE_ALL>();
    }
    dcci((__gm__ uint64_t *)coreHeadPtr->dumpAddr, cache_line_t::ENTIRE_DATA_CACHE);
}

/***********************************每个core内存分配示意图*************************************************

|------------------------------------------------core_0-------- --------------------------------------|
|---已使用addr、 MAGIC---|---bMsg_1---|---data_1---|---bMsg_2---|---data_2---|///未使用///|---backup---|
           |________________________________________________________________^

**********************************************************************************************************/
template <typename T>
__aicore__ inline void DumpTensorLocal2GMImpl(const LocalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)
{
    /* offset: 实际data数据大小 copyLen:以DUMP_UB_SIZE对齐的data大小 */
    uint32_t offset = dumpSize * sizeof(T);
    uint32_t copyLen = GetdataCopyCount(offset) * DUMP_UB_SIZE;
    uint32_t DumpHeadSize = sizeof(DumpMessageHead);
    uint32_t workOffset = copyLen + DumpHeadSize + DUMP_UB_SIZE;
    if (offset % ONE_BLK_SIZE != 0) {
        KERNEL_LOG(KERNEL_ERROR, "dump size is %d, which must be 32B aligned", offset);
        return;
    }

    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;
#ifdef __ENABLE_VECTOR_CORE__
#if defined(__DAV_M200_VEC__)
    uint32_t core = GetBlockIdxImpl() - get_data_main_base();
#else
    uint32_t core = GetBlockIdxImpl() - get_data_main_base() + AIV_CORE_NUM;
#endif
#else
    uint8_t core = GetBlockIdxImpl() + AIV_CORE_NUM;
#endif
    if (core >= DUMP_CORE_COUNT) {
        return;
    }
    __gm__ BlockInfo *coreHeadPtr = (__gm__ BlockInfo *)(dumpWorkspaceStart + DUMP_UINTSIZE * core);
    dcci((__gm__ uint64_t *)coreHeadPtr, cache_line_t::ENTIRE_DATA_CACHE);
    if (coreHeadPtr->dumpOffset < DumpHeadSize + DUMP_UB_SIZE) {
            KERNEL_LOG(KERNEL_ERROR, "Remained space[%u] is not enough for check!", coreHeadPtr->dumpOffset);
            return;
    }

    __gm__ uint8_t *gmBackAddr = (__gm__ uint8_t *)(dumpWorkspaceStart + DUMP_UINTSIZE * (core + 1) - DUMP_UB_SIZE);
    PipeBarrier<PIPE_ALL>();
    dcci((__gm__ uint64_t *)gmBackAddr, cache_line_t::ENTIRE_DATA_CACHE);
    if ((coreHeadPtr->dumpOffset < workOffset)) {
        KERNEL_LOG(KERNEL_ERROR, "Remained space[%u] is less than workOffset[%u]", coreHeadPtr->dumpOffset, workOffset);
        UpdateBlockInfo(0, DUMP_EXC_FLAG);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    /*---------Write HeadMsg---------*/
    T tmpData;
    DumpMessageHead dumpMsg = DumpMessageHead(static_cast<uint32_t>(DumpType::DUMP_TENSOR),
                                              DUMP_MSG_HEAD_SIZE + dumpSize * sizeof(T),
                                              static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tensor.GetPhyAddr())),
                                              GetDataType(tmpData),
                                              desc,
                                              0,
                                              CheckValidPosition(tensor),
                                              0);
    WriteHeadMsg(dumpMsg);
    /*---------Copy Data---------*/
    const Hardware srcHWPos = GetPhyType((TPosition)tensor.GetPosition());
    if (srcHWPos == Hardware::UB) {
        TensorDataCopy(tensor, copyLen);
    } else {
            /*---------Backup UB start---------*/
        DataCopyParams backupParams = GetDataCopyParamImpl(DUMP_UB_SIZE);
        LocalTensor<uint8_t> tmpLocal;
        InitTmpTensor(tmpLocal, (uint8_t)TPosition::VECIN);
        PipeBarrier<PIPE_ALL>();
        DataCopyUB2GMImpl((__gm__ uint8_t *)gmBackAddr, (__ubuf__ uint8_t *)tmpLocal.GetPhyAddr(), backupParams);
        /*---------data copy---------*/
        TensorDataLoopCopy(tmpLocal, tensor, copyLen);
        /*---------Recovery UB---------*/
        PipeBarrier<PIPE_ALL>();
        DataCopyGM2UBImpl((__ubuf__ uint32_t *)tmpLocal.GetPhyAddr(), (__gm__ uint32_t *)gmBackAddr, backupParams);
        PipeBarrier<PIPE_ALL>();
    }
    UpdateBlockInfo(offset, 0);
}

/***********************************每个core内存分配示意图*************************************************

|------------------------------------------------core_0-----------------------------------------------|
|---已使用addr、 MAGIC---|---bMsg_1---|---data_1---|---bMsg_2---|---data_2---|///未使用///|---backup---|
           |________________________________________________________________^

**********************************************************************************************************/
template <typename T>
__aicore__ inline void DumpTensorGM2GMImpl(const GlobalTensor<T> &tensor, uint32_t desc, uint32_t dumpSize)
{
    /* offset: 实际data数据大小 copyLen:以DUMP_UB_SIZE对齐的data大小 */
    uint32_t offset = dumpSize * sizeof(T);
    uint32_t copyLen = GetdataCopyCount(offset) * DUMP_UB_SIZE;
    uint32_t DumpHeadSize = sizeof(DumpMessageHead);
    uint32_t workOffset = copyLen + DumpHeadSize + DUMP_UB_SIZE;
    if (offset % ONE_BLK_SIZE != 0) {
        KERNEL_LOG(KERNEL_ERROR, "dump size is %d, which must be 32B aligned", offset);
        return;
    }
    uint64_t dumpWorkspaceStart = reinterpret_cast<uint64_t>(g_dumpWorkspaceReserved) - DUMP_WORKSPACE_SIZE;
#ifdef __ENABLE_VECTOR_CORE__
#if defined(__DAV_M200_VEC__)
    uint32_t core = GetBlockIdxImpl() - get_data_main_base();
#else
    uint32_t core = GetBlockIdxImpl() - get_data_main_base() + AIV_CORE_NUM;
#endif
#else
    uint8_t core = GetBlockIdxImpl() + AIV_CORE_NUM;
#endif
    if (core >= DUMP_CORE_COUNT) {
        return;
    }
    __gm__ BlockInfo *coreHeadPtr = (__gm__ BlockInfo *)(dumpWorkspaceStart + DUMP_UINTSIZE * core);
    dcci((__gm__ uint64_t *)coreHeadPtr, cache_line_t::ENTIRE_DATA_CACHE);
    if (coreHeadPtr->dumpOffset < DumpHeadSize + DUMP_UB_SIZE) {
            KERNEL_LOG(KERNEL_ERROR, "Remained space[%u] is not enough for check!", coreHeadPtr->dumpOffset);
            return;
    }
    /*---------Backup UB start---------*/
    DataCopyParams backupParams = GetDataCopyParamImpl(DUMP_UB_SIZE);
    LocalTensor<uint8_t> tmpLocal;
    InitTmpTensor(tmpLocal, (uint8_t)TPosition::VECIN);
    __gm__ uint8_t *gmBackAddr = (__gm__ uint8_t *)(dumpWorkspaceStart + DUMP_UINTSIZE * (core + 1) - DUMP_UB_SIZE);
    PipeBarrier<PIPE_ALL>();
    DataCopyUB2GMImpl((__gm__ uint8_t *)gmBackAddr, (__ubuf__ uint8_t *)tmpLocal.GetPhyAddr(), backupParams);
    PipeBarrier<PIPE_ALL>();
    dcci((__gm__ uint64_t *)gmBackAddr, cache_line_t::ENTIRE_DATA_CACHE);
    if ((coreHeadPtr->dumpOffset < workOffset)) {
        KERNEL_LOG(KERNEL_ERROR, "Remained space[%u] is less than workOffset[%u]", coreHeadPtr->dumpOffset, workOffset);
        UpdateBlockInfo(0, DUMP_EXC_FLAG);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    /*---------Write HeadMsg---------*/
    T tmpData;
    DumpMessageHead dumpMsg = DumpMessageHead(static_cast<uint32_t>(DumpType::DUMP_TENSOR),
        DUMP_MSG_HEAD_SIZE + dumpSize * sizeof(T),
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tensor.GetPhyAddr())),  GetDataType(tmpData), desc, 0,
        static_cast<uint32_t>(AscendC::Hardware::GM), 0);
    WriteHeadMsg(dumpMsg);
    /*---------Copy Data---------*/
    TensorDataLoopCopy(tmpLocal, tensor, copyLen);
    UpdateBlockInfo(offset, 0);
    /*---------Recovery UB---------*/
    PipeBarrier<PIPE_ALL>();
    DataCopyGM2UBImpl((__ubuf__ uint32_t *)tmpLocal.GetPhyAddr(), (__gm__ uint32_t *)gmBackAddr, backupParams);
    PipeBarrier<PIPE_ALL>();
    return;
}

__aicore__ inline void DumpShapeImpl(const ShapeInfo& shapeInfo)
{
    /* offset: 实际data数据大小 copyLen:以DUMP_UB_SIZE对齐的data大小 */
    uint32_t offset = sizeof(DumpShapeMessageHead);
    if (offset % ONE_BLK_SIZE != 0) {
        offset = (offset + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }
#ifdef __ENABLE_VECTOR_CORE__
#if defined(__DAV_M200_VEC__)
    uint32_t blockIdx = GetBlockIdxImpl() - get_data_main_base();
#else
    uint32_t blockIdx = GetBlockIdxImpl() - get_data_main_base() + AIV_CORE_NUM;
#endif
#else
    uint8_t blockIdx = GetBlockIdxImpl() + AIV_CORE_NUM;
#endif
    if (blockIdx > DUMP_CORE_COUNT) {
        return;
    }
    __gm__ BlockInfo *blockInfo = GetBlockInfo();
    __gm__ uint8_t *blockInfoStart = (__gm__ uint8_t *)blockInfo;
    dcci((__gm__ uint64_t *)blockInfoStart, cache_line_t::ENTIRE_DATA_CACHE);
    uint32_t remainSize = blockInfo->dumpOffset - offset;
    uint64_t dumpAddr = blockInfo->dumpAddr;
    uint64_t gmBkUpAddr = dumpAddr + remainSize;
    uint32_t tlvSize = sizeof(DumpShapeMessageHead) + DUMP_SHAPE_MESSAGE_TL_LEN;
    LocalTensor<uint8_t> tmpLocal;
    InitTmpTensor(tmpLocal, (uint8_t)TPosition::VECIN);
    DataCopyParams backupParams = GetDataCopyParamImpl(offset);
    PipeBarrier<PIPE_ALL>();
    DataCopyUB2GMImpl((__gm__ uint8_t*)gmBkUpAddr, (__ubuf__ uint8_t *)tmpLocal.GetPhyAddr(), backupParams);
    PipeBarrier<PIPE_ALL>();
    if (tlvSize > remainSize) {
        UpdateBlockInfo(0, DUMP_EXC_FLAG);
        KERNEL_LOG(KERNEL_ERROR, "remain space is not enough for this print");
        PipeBarrier<PIPE_ALL>();
        return;
    }
    __gm__ uint8_t *tlvAddr = (__gm__ uint8_t*)dumpAddr;
    uint64_t ubAddr = (uint64_t)tmpLocal.GetPhyAddr();
    *((__ubuf__ uint32_t *)ubAddr + DUMP_SHAPE_MESSAGE_HEAD_TYPE_POS) = static_cast<uint32_t>(DumpType::DUMP_SHAPE);
    *((__ubuf__ uint32_t *)ubAddr + DUMP_SHAPE_MESSAGE_HEAD_LEN_POS) = sizeof(DumpShapeMessageHead);
    *((__ubuf__ uint32_t *)ubAddr + DUMP_SHAPE_MESSAGE_HEAD_DIM_POS) = shapeInfo.shapeDim;
    for (uint32_t idx = 0; idx < shapeInfo.shapeDim && idx < K_MAX_SHAPE_DIM; idx++) {
        *((__ubuf__ uint32_t*)ubAddr + DUMP_SHAPE_MESSAGE_HEAD_SHAPE_START_POS + idx) = shapeInfo.shape[idx];
    }
    *((__ubuf__ uint32_t*)ubAddr + DUMP_SHAPE_MESSAGE_HEAD_RSV_POS) = 0;
    PipeBarrier<PIPE_ALL>();
    DataCopyParams headParams = GetDataCopyParamImpl(offset);
    DataCopyUB2GMImpl((__gm__ uint64_t*)(tlvAddr), (__ubuf__ uint64_t*)ubAddr, headParams);
    PipeBarrier<PIPE_ALL>();
    dcci((__gm__ uint64_t*)tlvAddr, cache_line_t::ENTIRE_DATA_CACHE);
    UpdateBlockInfo(tlvSize, 0);
    PipeBarrier<PIPE_ALL>();
    DataCopyGM2UBImpl((__ubuf__ uint32_t*)tmpLocal.GetPhyAddr(), (__gm__ uint32_t*)gmBkUpAddr, backupParams);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline uint32_t GetArgsNum()
{
    return 0;
}

template <typename T, typename... Args>
__aicore__ inline uint32_t GetArgsNum(T scalar, Args... args)
{
    return 1 + GetArgsNum(args...);
}

__aicore__ inline uint32_t GetStringLength(__gm__ const char* s)
{
    uint32_t i = 0;
    while (*(s + i) != '\0') {
        i++;
    }
    return i + 1;
}

__aicore__ inline uint32_t GetArgsSize()
{
    return 0;
}

template <typename... Args>
__aicore__ inline uint32_t GetArgsSize(Args&&... args);

template <typename... Args>
__aicore__ inline uint32_t GetArgsSizeImpl(__gm__ const char* s, Args&&... args)
{
    uint32_t strLen = GetStringLength(s);
    uint32_t strParamSize = ONE_PARAM_SIZE + strLen;
    return strParamSize + GetArgsSize(args...);
}

template <typename T, typename... Args>
__aicore__ inline uint32_t GetArgsSizeImpl(T scalar, Args&&... args)
{
    return ONE_PARAM_SIZE + GetArgsSize(args...);
}

template <typename... Args>
__aicore__ inline uint32_t GetArgsSize(Args&&... args)
{
    return GetArgsSizeImpl(args...);
}

template <typename... Args>
__aicore__ inline uint32_t GetParamSize(__gm__ const char* fmt, Args&&... args)
{
    uint32_t fmtSize = GetStringLength(fmt);
    uint32_t argsSize = GetArgsSize(args...);
    return fmtSize + argsSize + ONE_PARAM_SIZE;
}

__aicore__ inline void WriteString(uint64_t tmpAddr, __gm__ uint8_t* paramAddr, uint32_t paramIdx,
                                   __gm__ const char* s, uint32_t& offset)
{
    __gm__ uint64_t *stringAddr = reinterpret_cast<__gm__ uint64_t *>(paramAddr) + paramIdx;
    __gm__ uint64_t *dstStrAddr = reinterpret_cast<__gm__ uint64_t *>(paramAddr + offset);

    // write string value offset
    *((__gm__ uint64_t *)stringAddr) = static_cast<uint64_t>(offset - ONE_PARAM_SIZE * paramIdx);
    dcci((__gm__ uint64_t*)stringAddr, cache_line_t::ENTIRE_DATA_CACHE);
    paramAddr += ONE_PARAM_SIZE;

    // write string content
    __ubuf__ char *d = (__ubuf__ char *)(tmpAddr);
    uint32_t strLen = GetStringLength(s);
    uint32_t alignSize = strLen % DUMP_UB_SIZE;
    for (uint32_t i = 0; i < strLen / DUMP_UB_SIZE; i++) {
        for (uint32_t j = 0; j < DUMP_UB_SIZE; j++) {
            *(d + j) = *(s + j + i * DUMP_UB_SIZE);
        }

        PipeBarrier<PIPE_ALL>();
        DataCopyParams repeatStrParams = GetDataCopyParamImpl(DUMP_UB_SIZE);
        copy_ubuf_to_gm((__gm__ void *)(dstStrAddr), (__ubuf__ void *)tmpAddr, 0, repeatStrParams.blockCount,
            repeatStrParams.blockLen, repeatStrParams.srcStride, repeatStrParams.dstStride);
        PipeBarrier<PIPE_ALL>();
        dstStrAddr += DUMP_UB_SIZE;
    }
    if (alignSize != 0) {
        for (uint8_t j = 0; j < alignSize; j++) {
            *(d + j) = *(s + strLen - alignSize + j);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyParams repeatStrParams = GetDataCopyParamImpl(DUMP_UB_SIZE);
        copy_ubuf_to_gm((__gm__ void *)(dstStrAddr), (__ubuf__ void *)tmpAddr, 0, repeatStrParams.blockCount,
            repeatStrParams.blockLen, repeatStrParams.srcStride, repeatStrParams.dstStride);
        PipeBarrier<PIPE_ALL>();
    }
    dcci((__gm__ uint64_t*)dstStrAddr, cache_line_t::ENTIRE_DATA_CACHE);
    offset += GetStringLength(s);
}
template <typename T>
__aicore__ inline void WriteScalar(uint64_t tmpAddr, __gm__ uint8_t* paramAddr, uint32_t paramIdx, T scalar)
{
    PipeBarrier<PIPE_ALL>();
    __gm__ uint64_t *scalarAddr = (__gm__ uint64_t *)paramAddr + paramIdx;

    *scalarAddr = 0;
    static_assert(!SupportType<T, double>(), "printf unsupport double type");

    if constexpr (SupportType<T, half, float>()) {
        *((__gm__ float *)scalarAddr) = static_cast<float>(scalar);
    } else if constexpr (std::is_signed<T>::value) {
        *((__gm__ int64_t *)scalarAddr) = static_cast<int64_t>(scalar);
    } else if constexpr(std::is_unsigned<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = static_cast<uint64_t>(scalar);
    } else if constexpr(std::is_pointer<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = (uintptr_t)scalar;
    } else if constexpr(std::is_enum<T>::value) {
        *((__gm__ uint64_t *)scalarAddr) = static_cast<uint64_t>(scalar);
    }

    dcci((__gm__ uint64_t*)scalarAddr, cache_line_t::ENTIRE_DATA_CACHE);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void SetParam(uint64_t tmpAddr, __gm__ uint8_t* paramAddr, uint32_t paramIdx, uint32_t& offset)
{
    return;
}

template <typename... Args>
__aicore__ inline void SetParam(uint64_t tmpAddr, __gm__ uint8_t* paramAddr, uint32_t paramIdx,
                                uint32_t& offset, Args&&... args);

template <typename... Args>
__aicore__ inline void SetParamImpl(uint64_t tmpAddr, __gm__ uint8_t *paramAddr, uint32_t paramIdx, uint32_t &offset,
                                    __gm__ const char *s, Args&&... args)
{
    WriteString(tmpAddr, paramAddr, paramIdx, s, offset);
    SetParam(tmpAddr, paramAddr, paramIdx + 1, offset, args...);
}

template <typename T, typename... Args>
__aicore__ inline void SetParamImpl(uint64_t tmpAddr, __gm__ uint8_t* paramAddr, uint32_t paramIdx,
                                    uint32_t& offset, T scalar, Args&&... args)
{
    WriteScalar(tmpAddr, paramAddr, paramIdx, scalar);
    SetParam(tmpAddr, paramAddr, paramIdx + 1, offset, args...);
}

template <typename... Args>
__aicore__ inline void SetParam(uint64_t tmpAddr, __gm__ uint8_t* paramAddr, uint32_t paramIdx,
                                uint32_t& offset, Args&&... args)
{
    SetParamImpl(tmpAddr, paramAddr, paramIdx, offset, args...);
}

__aicore__ inline void WriteTLHead(DumpType printType, uint64_t tmpAddr, __gm__ uint8_t *tlv, uint32_t valueSize)
{
    *((__ubuf__ uint32_t *)tmpAddr) = static_cast<uint32_t>(printType);
    *((__ubuf__ uint32_t *)tmpAddr + 1) = valueSize;

    PipeBarrier<PIPE_ALL>();
    DataCopyParams headParams = GetDataCopyParamImpl(ONE_BLK_SIZE);
    copy_ubuf_to_gm((__gm__ void *)(tlv), (__ubuf__ void *)tmpAddr, 0, headParams.blockCount,
        headParams.blockLen, headParams.srcStride, headParams.dstStride);
    PipeBarrier<PIPE_ALL>();
    dcci((__gm__ uint64_t*)tlv, cache_line_t::ENTIRE_DATA_CACHE);
}

template <class... Args>

__aicore__ inline void PrintfImpl(DumpType printType, __gm__ const char *fmt, Args&&... args)
{
#ifdef ASCENDC_DUMP
#ifdef __ENABLE_VECTOR_CORE__
#if defined(__DAV_M200_VEC__)
    uint32_t blockIdx = GetBlockIdxImpl() - get_data_main_base();
#else
    uint32_t blockIdx = GetBlockIdxImpl() - get_data_main_base() + AIV_CORE_NUM;
#endif
#else
    uint8_t blockIdx = GetBlockIdxImpl() + AIV_CORE_NUM;
#endif
    if (blockIdx >= DUMP_CORE_COUNT) {
        return;
    }
    __gm__ BlockInfo *blockInfo = GetBlockInfo();
    __gm__ uint8_t *blockInfoStart = (__gm__ uint8_t *)blockInfo;
    dcci((__gm__ uint64_t *)blockInfoStart, cache_line_t::ENTIRE_DATA_CACHE);
    uint32_t remainSize = blockInfo->dumpOffset - DUMP_UB_SIZE;
    uint64_t dumpAddr = blockInfo->dumpAddr;
    uint64_t gmBackUpAddr = dumpAddr + remainSize;
    uint32_t paramSize = GetParamSize(fmt, args...);
    uint32_t paramNum = GetArgsNum(args...) + 1;
    paramSize = (paramSize + ONE_PARAM_SIZE - 1) & (~(ONE_PARAM_SIZE - 1));
    uint32_t tlvSize = paramSize + ONE_PARAM_SIZE;
    LocalTensor<uint8_t> tmpLocal;
    InitTmpTensor(tmpLocal, (uint8_t)TPosition::VECIN);
    DataCopyParams backupParams = GetDataCopyParamImpl(DUMP_UB_SIZE);
    PipeBarrier<PIPE_ALL>();
    copy_ubuf_to_gm((__gm__ void *)gmBackUpAddr, (__ubuf__ void *)tmpLocal.GetPhyAddr(), 0,
        backupParams.blockCount, backupParams.blockLen, backupParams.srcStride, backupParams.dstStride);
    PipeBarrier<PIPE_ALL>();
    if (tlvSize > remainSize) {
        UpdateBlockInfo(0, DUMP_EXC_FLAG);
        KERNEL_LOG(KERNEL_ERROR, "remain space is not enough for this print");
        PipeBarrier<PIPE_ALL>();
        return;
    }

    __gm__ uint8_t *tlvAddr = (__gm__ uint8_t *)dumpAddr;
    __gm__ uint8_t *paramAddr = tlvAddr + ONE_PARAM_SIZE;
    WriteTLHead(printType, (uint64_t)tmpLocal.GetPhyAddr(), tlvAddr, paramSize); // HEAD 2 UB

    uint32_t offset = paramNum * ONE_PARAM_SIZE;
    uint32_t strLen = GetStringLength(fmt);

    WriteString((uint64_t)tmpLocal.GetPhyAddr(), paramAddr, 0, fmt, offset);

    uint32_t paramIdx = 1;
    SetParam((uint64_t)tmpLocal.GetPhyAddr(), paramAddr, paramIdx, offset, args...);

    UpdateBlockInfo(tlvSize, 0);
    PipeBarrier<PIPE_ALL>();

    copy_gm_to_ubuf((__ubuf__ void *)tmpLocal.GetPhyAddr(), (__gm__ void *)gmBackUpAddr, 0, backupParams.blockCount,
        backupParams.blockLen, backupParams.srcStride, backupParams.dstStride);
    PipeBarrier<PIPE_ALL>();
#endif
}

__aicore__ inline void DumpTimeStampImpl(uint32_t descId)
{
    return;
}

__aicore__ inline void AscendCTimeStamp(uint32_t descId, uint64_t pcPtr = 0)
{
    return;
}
__aicore__ inline void InitDump(bool mixFlag, uint32_t gmLen)
{
#if defined(ASCENDC_DUMP) || defined(ASCENDC_ACC_DUMP)
    g_dumpWorkspaceReserved = GetSysWorkSpacePtr();
    InitDumpImpl(mixFlag, gmLen);
#else
    return;
#endif
}
__aicore__ inline void InitDump(bool mixFlag, GM_ADDR dumpStartAddr, uint32_t gmLen)
{
#if defined(ASCENDC_DUMP) || defined(ASCENDC_ACC_DUMP)
    g_dumpWorkspaceReserved = dumpStartAddr + DUMP_WORKSPACE_SIZE;
    InitDumpImpl(mixFlag, gmLen);
#else
    return;
#endif
}
}
#endif
