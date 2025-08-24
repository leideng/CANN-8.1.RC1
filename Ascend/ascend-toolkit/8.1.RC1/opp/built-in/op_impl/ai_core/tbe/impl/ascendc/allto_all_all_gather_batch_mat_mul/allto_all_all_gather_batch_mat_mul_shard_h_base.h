/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

/* !
 * \file allto_all_all_gather_batch_mat_mul_shard_h_base.h
 * \brief
 */

#ifndef MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_SHARD_H_BASE_H
#define MC2_ALLTOALL_ALLGATHER_BATCHMATMUL_SHARD_H_BASE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../batch_mat_mul_v3/batch_mat_mul_v3.h"

using namespace AscendC;
using namespace matmul;

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
class AlltoAllAllGatherBatchMatMulShardHBase{
public:
    __aicore__ inline AlltoAllAllGatherBatchMatMulShardHBase(){};
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM, GM_ADDR y2GM, GM_ADDR y3GM,
                                GM_ADDR workspaceGM, TPipe *pipe, AlltoAllAllGatherBatchMatMulTilingData* tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void GetTilingData(AlltoAllAllGatherBatchMatMulTilingData *tilingData);
    __aicore__ inline void InitTpipe();

    __aicore__ inline void Transpose(GlobalTensor<DataType1> transOut);
    __aicore__ inline void Act(GlobalTensor<DataType1> bmmOut, GlobalTensor<DataType2> biasBaseGM);

    AlltoAllAllGatherBatchMatmulActType actType;

    // shardType 0/1: 输入 E,C,H
    uint64_t E;
    uint64_t C; //shardType 0: C   shardType 1: C/tp
    uint64_t H; //shardType 0: H/tp   shardType 1: H
    uint64_t M; // M/tp
	uint64_t MAlign; 
    uint64_t expertOverEp;

    uint32_t epRankSize;
    uint32_t tpRankSize;
    uint32_t epRankId;
    uint32_t tpRankId;

    uint64_t xDataLen;
    uint64_t biasDataLen;
    uint64_t shareTmpSize;
    uint64_t ubSize;
    uint32_t vecCoreNum;
    uint64_t ubCapacityForTrans;
    uint64_t ubCapacityForAct;

    uint32_t handleIdx;
    uint32_t handleOffset;
	bool isbias;

    GlobalTensor<DataType1> xGM;
    GlobalTensor<DataType1> weightGM;
    GlobalTensor<DataType2> biasGM;
    GlobalTensor<DataType1> y1GM;
    GlobalTensor<DataType1> y2GM;
    GlobalTensor<DataType1> y3GM;

    GlobalTensor<DataType1> allgatherLocalOutGM;
    GlobalTensor<DataType1> alltoallOutGM;
    GlobalTensor<DataType1> allgatherOutGM;
    GlobalTensor<DataType1> transposeOutGM; //每轮地址复用
    GlobalTensor<DataType1> bmmOutGM;
    Mc2MatmulTilingData *bmmTiling;

    Hccl<HCCL_SERVER_TYPE_AICPU> hcclAlltoall;
    Hccl<HCCL_SERVER_TYPE_AICPU> hcclAllgather;
    HcclHandle allgatherHandleList[MAX_HCCL_HANDLE];
    HcclDataType hcclDataType;

    TPipe *tpipe;
    TBuf<> tBuf;
    LocalTensor<DataType1> transposeTmp;
    LocalTensor<DataType1> xTmp; // 类型转换之前的数据
    LocalTensor<float> xCastTmp; // 类型转换之后的数据
	LocalTensor<DataType2> biasTmp;
    LocalTensor<half> actOutTmp; // 类型转换之后的输出
    LocalTensor<float> actCastOutTmp; // 类型转换之后的输出
    LocalTensor<uint8_t> sharedTmp;

    HcclHandle allgatherHandle;
};

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardHBase<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR biasGM, GM_ADDR y1GM, GM_ADDR y2GM, GM_ADDR y3GM,
                            GM_ADDR workspaceGM, TPipe *pipe, AlltoAllAllGatherBatchMatMulTilingData* tilingData)
{
    this->tpipe = pipe;
    GetTilingData(tilingData);
    InitTpipe();

    this->xGM.SetGlobalBuffer((__gm__ DataType1*)xGM);
    this->weightGM.SetGlobalBuffer((__gm__ DataType1*)weightGM);
    this->y1GM.SetGlobalBuffer((__gm__ DataType1*)y1GM);
    if constexpr(IsNeedBias) {
        this->biasGM.SetGlobalBuffer((__gm__ DataType2*)biasGM);
    }
    if constexpr(IsNeedY2) {
        this->y2GM.SetGlobalBuffer((__gm__ DataType1*)y2GM);
    }
    if constexpr(IsNeedY3) {
        this->y3GM.SetGlobalBuffer((__gm__ DataType1*)y3GM);
    }

    uint64_t alltoallOutSize = E * C * H * xDataLen;
    uint64_t allgatherOutSize = alltoallOutSize * tpRankSize;
    alltoallOutSize = Ceil(alltoallOutSize, 512) * 512 / xDataLen; // 512B对齐
    allgatherOutSize = Ceil(allgatherOutSize, 512) * 512 / xDataLen; // 512B对齐

    alltoallOutGM.SetGlobalBuffer((__gm__ DataType1*)workspaceGM);
    allgatherOutGM.SetGlobalBuffer((__gm__ DataType1*)alltoallOutGM.GetPhyAddr() + alltoallOutSize);
    transposeOutGM.SetGlobalBuffer((__gm__ DataType1*)allgatherOutGM.GetPhyAddr() + allgatherOutSize);
    bmmOutGM.SetGlobalBuffer((__gm__ DataType1*)transposeOutGM.GetPhyAddr() + allgatherOutSize);

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    auto contextGM1 = AscendC::GetHcclContext<1>();
    hcclAlltoall.Init(contextGM0);
    hcclAllgather.Init(contextGM1);

    if constexpr(AscendC::IsSameType<DataType1, bfloat16_t>::value) {
        hcclDataType = HCCL_DATA_TYPE_BFP16;
    } else {
        hcclDataType = HCCL_DATA_TYPE_FP16;
    }

    epRankId = hcclAlltoall.GetRankId();
    tpRankId = hcclAllgather.GetRankId();
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardHBase<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::InitTpipe()
{
    tpipe->InitBuffer(tBuf, ubSize);
    transposeTmp = tBuf.Get<DataType1>();
    xTmp = tBuf.Get<DataType1>(ubCapacityForAct);
    uint32_t offset = ubCapacityForAct * xDataLen;
    if (IsNeedBias || (actType != AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE)) {
        if (AscendC::IsSameType<DataType1, bfloat16_t>::value || (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU)) {
            xCastTmp = tBuf.Get<float>()[offset / SIZE_OF_FLOAT32];
            offset += (ubCapacityForAct * SIZE_OF_FLOAT32);
            actCastOutTmp = tBuf.Get<float>()[offset / SIZE_OF_FLOAT32];
            offset += (ubCapacityForAct * SIZE_OF_FLOAT32);
        } else {
            actOutTmp = tBuf.Get<half>()[offset / SIZE_OF_FLOAT16];
            offset += (ubCapacityForAct * SIZE_OF_FLOAT16);
        }

        if constexpr (IsNeedBias) {
            biasTmp = tBuf.Get<DataType2>()[offset / biasDataLen];
            offset += (ubCapacityForAct * biasDataLen);
        }

        if ((actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU) ||
            (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU)) {
            sharedTmp = tBuf.Get<uint8_t>()[offset];
        }
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardHBase<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::GetTilingData(AlltoAllAllGatherBatchMatMulTilingData *tilingData)
{
    E = tilingData->commonTiling.expert;
    C = tilingData->commonTiling.C;
    H = tilingData->commonTiling.HOverTp;
    M = tilingData->commonTiling.MOverTp;
    expertOverEp = tilingData->commonTiling.EOverEp;
    MAlign = (M % 16 == 0) ? M : ((M / 16 + 1) * 16); // 32B 对齐

    tpRankSize = tilingData->commonTiling.tpGroupSize;
    epRankSize = tilingData->commonTiling.epGroupSize;

    shareTmpSize = tilingData->commonTiling.fastGeluBuffer;
    ubCapacityForTrans = tilingData->commonTiling.ubCapacityForTrans;
    ubCapacityForAct = tilingData->commonTiling.ubCapacityForAddActivate;
    ubSize = tilingData->commonTiling.totalUbSize;
    ubSize = (ubSize / 32) * 32;
    vecCoreNum = tilingData->commonTiling.aivCoreNum;

    bmmTiling = &tilingData->localTiling;

    xDataLen = sizeof(DataType1);
    biasDataLen = sizeof(DataType2);

    actType = (enum AlltoAllAllGatherBatchMatmulActType)tilingData->commonTiling.activateType;
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardHBase<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::Process()
{
    GlobalTensor<DataType1> transOut;
    GlobalTensor<DataType1> bmmOut;
    if constexpr(IsNeedY2) {
        transOut = y2GM;
    } else {
        transOut = transposeOutGM;
    }
    if constexpr(IsNeedY3) {
        bmmOut = y3GM;
    } else if (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE){
        bmmOut = y1GM;
    } else {
        bmmOut = bmmOutGM;
    }

    uint64_t alltoallDataCounts[MAX_EP_RANK_SIZE] = {0U};
    uint64_t alltoallDispls[MAX_EP_RANK_SIZE] = {0U};
    for (uint32_t i = 0U; i < epRankSize; i++)
    {
        alltoallDataCounts[i] = expertOverEp * C * H;
        alltoallDispls[i] = i * expertOverEp * C * H;
    }
	if ASCEND_IS_AIV{
		HcclHandle alltoallHandle = hcclAlltoall.AlltoAllV<true>((__gm__ uint8_t*)xGM.GetPhyAddr(), alltoallDataCounts, alltoallDispls, hcclDataType,
                                                                 (__gm__ uint8_t*)alltoallOutGM.GetPhyAddr(), alltoallDataCounts, alltoallDispls,  hcclDataType);
        hcclAllgather.InterHcclGroupSync(HCCL_GROUP_ID_0, alltoallHandle);
        allgatherHandle = hcclAllgather.AllGather<true>((__gm__ uint8_t*)alltoallOutGM.GetPhyAddr(), (__gm__ uint8_t*)allgatherOutGM.GetPhyAddr(), E * C * H, hcclDataType, 0U);
        hcclAllgather.Wait(alltoallHandle);
	}
	SyncAll<false>();

    // tp, ep, E/ep, C, H/tp --> E/ep, ep, C, tp, H/tp
    Transpose(transOut);
	SyncAll<false>();

    tpipe->Reset();
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, false>;
    using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, IsTransposeWeight>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType1, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType2, false>;
    BatchMatMulCommonKernel<aType, bType, cType, biasType> bmmv3;
    GlobalTensor<DataType2> biasBuf;
    if constexpr(IsNeedBias) {
        biasBuf = biasGM;
    }
    bmmv3.Init((__gm__ uint8_t*)transOut.GetPhyAddr(), (__gm__ uint8_t*)weightGM.GetPhyAddr(), (__gm__ uint8_t*)bmmOut.GetPhyAddr(), nullptr, nullptr, nullptr, &bmmTiling->bmmTilingData, tpipe);
    bmmv3.Process();
    SyncAll<false>();

    tpipe->Reset();
    InitTpipe();
	if ASCEND_IS_AIV{
		Act(bmmOut, biasBuf);
	}

	if ASCEND_IS_AIV{
		hcclAllgather.Finalize();
        hcclAlltoall.Finalize();
	}
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardHBase<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::Transpose(GlobalTensor<DataType1> transOut)
{
    DataCopyExtParams dataCopyInParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyPadExtParams<DataType1> dataCopyInPadParams = {false, 0U, 0U, 0U};

    uint64_t ubCapacityForTrans = ubSize / 2;
    uint64_t totalM = tpRankSize * epRankSize * expertOverEp * C;
    uint64_t tileLen = totalM % vecCoreNum == 0 ? totalM / vecCoreNum : totalM / vecCoreNum + 1;
    uint32_t blockIdx = GetBlockIdx();
    uint64_t curRowOffset = tileLen * blockIdx;
    uint64_t curDataOffset = curRowOffset * H;
    uint64_t endDataOffset = tileLen * H * (blockIdx + 1U);
    endDataOffset = endDataOffset < totalM * H ? endDataOffset : totalM * H;
    uint64_t endRowOffset = endDataOffset / H;
    uint64_t curBlockMaxRow = ((curRowOffset / C) + 1) * C;
    curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
     uint64_t HAlign = (H % 256 == 0) ? H : ((H / 256 + 1) * 256); // 512字节对齐
    uint64_t perMaxRow = ubCapacityForTrans / HAlign;
    uint64_t dataCnt = 0UL;
    uint32_t t, e, k, c, blockInnerOffset;
    GlobalTensor<DataType1> dataCopyInGM, dataCopyOutGM;
    uint32_t i = 0U;

    while (curDataOffset < endDataOffset) {
        i += 1;
        t = curRowOffset / (epRankSize * expertOverEp * C);
        e = (curRowOffset % (epRankSize * expertOverEp * C)) / (expertOverEp * C);
        k = (curRowOffset % (expertOverEp * C)) / C;
        c = curRowOffset % C;
        blockInnerOffset = curDataOffset % H;
        if (perMaxRow == 0UL) { //H超大，分片切列，一片处理的最大数据量为ubCapacity
            if ((curRowOffset + 1) * H - curDataOffset > ubCapacityForTrans) {
                dataCnt = ubCapacityForTrans;
            } else {
                dataCnt = (curRowOffset + 1) * H - curDataOffset;
                curRowOffset += 1U;
            }
            dataCopyInParams.blockLen = dataCnt * xDataLen;
            dataCopyOutParams.blockLen = dataCnt * xDataLen;
        } else { //H较小，分片切行
            if (curBlockMaxRow * H - curDataOffset > perMaxRow * H) {
                dataCnt = perMaxRow * H;
            } else {
                dataCnt = curBlockMaxRow * H - curDataOffset;
                curBlockMaxRow += C;
                curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
            }
            dataCopyInParams.blockLen = H * xDataLen;
            dataCopyInParams.blockCount = dataCnt / H;
            dataCopyOutParams.blockLen = H * xDataLen;
            dataCopyOutParams.blockCount = dataCnt / H;
            dataCopyOutParams.dstStride = (H * tpRankSize - H) * xDataLen;
            curRowOffset += dataCnt / H;
        }
       // tp, ep, E/ep, C, H/tp --> E/ep, ep, C, tp, H/tp
        dataCopyInGM = allgatherOutGM[t * (epRankSize * expertOverEp * C * H) + e * (expertOverEp * C * H) + k * (C * H) + c * H + blockInnerOffset];
        dataCopyOutGM = transOut[k * (epRankSize * C * tpRankSize * H) + e * (C * tpRankSize * H) + c * (tpRankSize * H) + t * H + blockInnerOffset];
        if (i > 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        }
        DataCopyPad(transposeTmp, dataCopyInGM, dataCopyInParams, dataCopyInPadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        DataCopyPad(dataCopyOutGM, transposeTmp, dataCopyOutParams);

        curDataOffset += dataCnt;
    }
}

template <typename DataType1, typename DataType2, int64_t ShardType, bool IsTransposeWeight, bool IsNeedBias, bool IsNeedY2, bool IsNeedY3>
__aicore__ inline void AlltoAllAllGatherBatchMatMulShardHBase<DataType1, DataType2, ShardType, IsTransposeWeight, IsNeedBias, IsNeedY2, IsNeedY3>::Act(GlobalTensor<DataType1> bmmOut, GlobalTensor<DataType2> biasBaseGM)
{
    DataCopyExtParams dataCopyInParams = {1U, 0U, 0U, 0U, 0U};
	DataCopyExtParams dataCopyInBiasParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams = {1U, 0U, 0U, 0U, 0U};
    DataCopyPadExtParams<DataType1> dataCopyInPadParams = {false, 0U, 0U, 0U};
	DataCopyPadExtParams<DataType2> dataCopyInBiasPadParams = {false, 0U, 0U, 0U};

    // E/ep, ep, C, M/tp
    uint64_t totalM = expertOverEp * epRankSize * C;
    uint64_t tileLen = totalM % vecCoreNum == 0 ? (totalM / vecCoreNum) : (totalM / vecCoreNum + 1);
    uint32_t blockIdx = GetBlockIdx();
    uint64_t curRowOffset = tileLen * blockIdx;
    uint64_t curDataOffset = curRowOffset * M;
    uint64_t endDataOffset = tileLen * M * (blockIdx + 1U);
    endDataOffset = endDataOffset < totalM * M ? endDataOffset : totalM * M;
    uint64_t endRowOffset = endDataOffset / M;
    uint64_t curBlockMaxRow = ((curRowOffset / C) + 1) * C;
    curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
    uint64_t perMaxRow = ubCapacityForAct / MAlign;
    uint64_t dataCnt = 0UL;
	uint64_t realDataCnt = 0UL;
	uint64_t biasCnt = 0UL;
    uint32_t k, blockInnerOffset;
    GlobalTensor<DataType1> dataCopyInGM, dataCopyOutGM, y3OutGM;
    uint32_t i = 0U;

     while (curDataOffset < endDataOffset) {
        i += 1;
        k = curRowOffset / (epRankSize * C);
        blockInnerOffset = curDataOffset % M;
        if (perMaxRow == 0UL) { // M超大，分片切列，一片处理的最大数据量为ubCapacity
            if ((curRowOffset + 1) * M - curDataOffset > ubCapacityForAct) {
                dataCnt = ubCapacityForAct;
            } else {
                dataCnt = (curRowOffset + 1) * M - curDataOffset;
                curRowOffset += 1U;
            }
            biasCnt = 1UL;
            dataCopyInParams.blockLen = dataCnt * xDataLen;
            dataCopyOutParams.blockLen = dataCnt * xDataLen;
            dataCopyInBiasParams.blockLen = dataCnt * biasDataLen;
            realDataCnt = dataCnt;
        } else { // M较小，分片切行
            if (curBlockMaxRow - curRowOffset > perMaxRow) {
                dataCnt = perMaxRow * M;
            } else {
                dataCnt = curBlockMaxRow * M - curDataOffset;
                curBlockMaxRow += C;
                curBlockMaxRow = curBlockMaxRow < endRowOffset ? curBlockMaxRow : endRowOffset;
            }
            biasCnt = dataCnt / M;
            curRowOffset += biasCnt;
            dataCopyInParams.blockLen = M * xDataLen;
            dataCopyInParams.blockCount = biasCnt;
            dataCopyOutParams.blockLen = M * xDataLen;
            dataCopyOutParams.blockCount = biasCnt;
            dataCopyInBiasParams.blockLen = M * biasDataLen;
            realDataCnt = biasCnt * MAlign;
        }
        dataCopyInGM = bmmOut[curDataOffset];
        dataCopyOutGM = y1GM[curDataOffset];
        if (i > 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        }
        DataCopyPad(xTmp, dataCopyInGM, dataCopyInParams, dataCopyInPadParams);

        if ((!IsNeedBias) && (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE)) {
			AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
			DataCopyPad(dataCopyOutGM, xTmp, dataCopyOutParams);
			curDataOffset += dataCnt;
			continue;
		}

        if constexpr(IsNeedBias) {
			DataCopyPad(biasTmp, biasBaseGM[k * M + blockInnerOffset], dataCopyInBiasParams, dataCopyInBiasPadParams);
		}

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        if constexpr(AscendC::IsSameType<DataType1, bfloat16_t>::value) {
            Cast(xCastTmp, xTmp, RoundMode::CAST_NONE, realDataCnt);
            pipe_barrier(PIPE_V);
            if constexpr(IsNeedBias) {
                for (uint64_t index = 0UL; index < biasCnt; index++) {
                    Add(xCastTmp[index * MAlign], xCastTmp[index * MAlign], biasTmp, dataCopyInBiasParams.blockLen / biasDataLen);
                    pipe_barrier(PIPE_V);
                }
            }
            if constexpr(IsNeedY3) {
                Cast(xTmp, xCastTmp, RoundMode::CAST_ROUND, realDataCnt);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                y3OutGM = y3GM[curDataOffset];
                DataCopyPad(y3OutGM, xTmp, dataCopyOutParams);
            }

            if (actType != AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE) {
				switch (actType){
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU:
                        Gelu(actCastOutTmp, xCastTmp, sharedTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU:
                        Silu(actCastOutTmp, xCastTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_RELU:
                        Relu(actCastOutTmp, xCastTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU:
                        FasterGelu(actCastOutTmp, xCastTmp, sharedTmp, realDataCnt);
                        break;
                    default:
                        break;
                }  
                pipe_barrier(PIPE_V);
                if constexpr(IsNeedY3) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
					Cast(xTmp, actCastOutTmp, RoundMode::CAST_ROUND, realDataCnt);
                }			
			} else {
                Cast(xTmp, xCastTmp, RoundMode::CAST_ROUND, realDataCnt);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            DataCopyPad(dataCopyOutGM, xTmp, dataCopyOutParams);
        } else {
            if constexpr(IsNeedBias) {
				for (uint64_t index = 0UL; index < biasCnt; index++) {
					Add(xTmp[index * MAlign], xTmp[index * MAlign], biasTmp, dataCopyInBiasParams.blockLen / biasDataLen);
					pipe_barrier(PIPE_V);
				}
			}
			if constexpr(IsNeedY3) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
                y3OutGM = y3GM[curDataOffset];
				DataCopyPad(y3OutGM, xTmp, dataCopyOutParams);
            }

            if (actType != AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE) {
				switch (actType){
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_GELU:
                        Gelu(actOutTmp, xTmp, sharedTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU:
						Cast(xCastTmp, xTmp, RoundMode::CAST_NONE, realDataCnt);
                        pipe_barrier(PIPE_V);
                        Silu(actCastOutTmp, xCastTmp, realDataCnt);
                        pipe_barrier(PIPE_V);
                        if constexpr(IsNeedY3) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                        }
                        Cast(xTmp, actCastOutTmp, RoundMode::CAST_ROUND, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_RELU:
                        Relu(actOutTmp, xTmp, realDataCnt);
                        break;
                    case AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_FASTGELU:
                        FasterGelu(actOutTmp, xTmp, sharedTmp, realDataCnt);
                        break;
                    default:
                        break;
                }			
			}
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
			if ((actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_NONE) ||
                (actType == AlltoAllAllGatherBatchMatmulActType::ALLTOALL_ALLGATHER_BATCHMATMUL_ACT_TYPE_SILU)) {
                DataCopyPad(dataCopyOutGM, xTmp, dataCopyOutParams);
            } else {
                DataCopyPad(dataCopyOutGM, actOutTmp, dataCopyOutParams);
            }
        }
        curDataOffset += dataCnt;
    }
}
#endif