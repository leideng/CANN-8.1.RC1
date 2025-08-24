#ifndef __TIKCFW_KERNEL_TILING_H_
#define __TIKCFW_KERNEL_TILING_H_

#if defined(ASCENDC_CPU_DEBUG)
#include <cstdint>
#include <cstring>
#endif

#pragma pack(push, 8)
struct LogSoftMaxTiling {
    uint32_t srcM = 0;
    uint32_t srcK = 0;
    uint32_t srcSize = 0;
    uint32_t outMaxM = 0;
    uint32_t outMaxK = 0;
    uint32_t outMaxSize = 0;
    uint32_t splitM = 0;
    uint32_t splitK = 0;
    uint32_t splitSize = 0;
    uint32_t reduceM = 0;
    uint32_t reduceK = 0;
    uint32_t reduceSize = 0;
    uint32_t rangeM = 0;
    uint32_t tailM = 0;
    uint32_t tailSplitSize = 0;
    uint32_t tailReduceSize = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct SoftMaxTiling {
    uint32_t srcM = 0;
    uint32_t srcK = 0;
    uint32_t srcSize = 0;
    uint32_t outMaxM = 0;
    uint32_t outMaxK = 0;
    uint32_t outMaxSize = 0;
    uint32_t splitM = 0;
    uint32_t splitK = 0;
    uint32_t splitSize = 0;
    uint32_t reduceM = 0;
    uint32_t reduceK = 0;
    uint32_t reduceSize = 0;
    uint32_t rangeM = 0;
    uint32_t tailM = 0;
    uint32_t tailSplitSize = 0;
    uint32_t tailReduceSize = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct Mc2ServerCfg {
    uint32_t version = 0;
    uint8_t debugMode = 0;
    uint8_t sendArgIndex = 0;
    uint8_t recvArgIndex = 0;
    uint8_t commOutArgIndex = 0;
    uint8_t reserved[8] = {};
};
#pragma pack(pop)
#pragma pack(push, 8)
struct Mc2HcommCfg {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    char reserved[13] = {};
    char groupName[128] = {};
    char algConfig[128] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct Mc2InitTiling {
    uint8_t reserved[64] = {};
};
#pragma pack(pop)
#pragma pack(push, 8)
struct Mc2CcTiling {
    uint8_t reserved[280] = {};
};
#pragma pack(pop)
#pragma pack(push, 8)
struct TCubeTiling {
    int32_t usedCoreNum = 0;
    int32_t M = 0;
    int32_t N = 0;
    int32_t Ka = 0;
    int32_t Kb = 0;
    int32_t singleCoreM = 0;
    int32_t singleCoreN = 0;
    int32_t singleCoreK = 0;
    int32_t baseM = 0;
    int32_t baseN = 0;
    int32_t baseK = 0;
    int32_t depthA1 = 0;
    int32_t depthB1 = 0;
    int32_t stepM = 0;
    int32_t stepN = 0;
    int32_t isBias = 0;
    int32_t transLength = 0;
    int32_t iterateOrder = 0;
    int32_t shareMode = 0;
    int32_t shareL1Size = 0;
    int32_t shareL0CSize = 0;
    int32_t shareUbSize = 0;
    int32_t batchM = 0;
    int32_t batchN = 0;
    int32_t singleBatchM = 0;
    int32_t singleBatchN = 0;
    int32_t stepKa = 0;
    int32_t stepKb = 0;
    int32_t depthAL1CacheUB = 0;
    int32_t depthBL1CacheUB = 0;
    int32_t dbL0A = 0;
    int32_t dbL0B = 0;
    int32_t dbL0C = 0;
    int32_t ALayoutInfoB = 0;
    int32_t ALayoutInfoS = 0;
    int32_t ALayoutInfoN = 0;
    int32_t ALayoutInfoG = 0;
    int32_t ALayoutInfoD = 0;
    int32_t BLayoutInfoB = 0;
    int32_t BLayoutInfoS = 0;
    int32_t BLayoutInfoN = 0;
    int32_t BLayoutInfoG = 0;
    int32_t BLayoutInfoD = 0;
    int32_t CLayoutInfoB = 0;
    int32_t CLayoutInfoS1 = 0;
    int32_t CLayoutInfoN = 0;
    int32_t CLayoutInfoG = 0;
    int32_t CLayoutInfoS2 = 0;
    int32_t BatchNum = 0;
    int32_t reserved = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct BatchNormTiling {
    uint32_t originalBLength = 0;
    uint32_t meanVarSize = 0;
    uint32_t meanTmpTensorPos = 0;
    uint32_t varianceTmpTensorPos = 0;
    uint32_t tmpBufSize = 0;
    uint32_t oneTmpSize = 0;
    uint32_t firstTmpStartPos = 0;
    uint32_t secondTmpStartPos = 0;
    uint32_t thirdTmpStartPos = 0;
    uint32_t loopRound = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t meanVarTailSize = 0;
    uint32_t meanVarTailPos = 0;
    uint32_t bshCurLength = 0;
    uint32_t shCurLength = 0;
    float firstDimValueBack = 0;
    uint32_t castHalfRepStride = 0;
    uint32_t shCurLengthBlockNum = 0;
    uint32_t castHalfOutRepStride = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct DeepNormTiling {
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    uint32_t inputXSize = 0;
    uint32_t meanVarSize = 0;
    uint32_t numberOfTmpBuf = 0;
    uint32_t meanTmpTensorPos = 0;
    uint32_t meanTmpTensorSize = 0;
    uint32_t varianceTmpTensorPos = 0;
    uint32_t varianceTmpTensorSize = 0;
    uint32_t tmpBufSize = 0;
    uint32_t oneTmpSize = 0;
    uint32_t firstTmpStartPos = 0;
    uint32_t secondTmpStartPos = 0;
    uint32_t thirdTmpStartPos = 0;
    uint32_t loopRound = 0;
    uint32_t inputRoundSize = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t meanVarRoundSize = 0;
    uint32_t meanVarTailSize = 0;
    uint32_t meanVarTailPos = 0;
    uint32_t bshCurLength = 0;
    uint32_t bsCurLength = 0;
    float lastDimValueBack = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct GroupNormTiling {
    uint32_t n = 0;
    uint32_t c = 0;
    uint32_t hw = 0;
    uint32_t g = 0;
    uint32_t d = 0;
    uint32_t hwAlignSize = 0;
    uint32_t dhwAlignSize = 0;
    uint32_t inputXSize = 0;
    uint32_t meanVarSize = 0;
    uint32_t numberOfTmpBuf = 0;
    uint32_t meanTmpTensorPos = 0;
    uint32_t meanTmpTensorSize = 0;
    uint32_t varianceTmpTensorPos = 0;
    uint32_t varianceTmpTensorSize = 0;
    uint32_t tmpBufSize = 0;
    uint32_t oneTmpSize = 0;
    uint32_t firstTmpStartPos = 0;
    uint32_t secondTmpStartPos = 0;
    uint32_t thirdTmpStartPos = 0;
    uint32_t loopRound = 0;
    uint32_t inputRoundSize = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t meanVarRoundSize = 0;
    uint32_t meanVarTailSize = 0;
    uint32_t meanVarTailPos = 0;
    uint32_t bshCurLength = 0;
    uint32_t bsCurLength = 0;
    float factor = 0;
    bool smallShape = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct LayerNormGradBetaTiling {
    uint32_t stackBufferSize = 0;
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    uint32_t bshLength = 0;
    uint32_t bsLength = 0;
    uint32_t oneCalSize = 0;
    uint32_t numberOfTmpBuf = 0;
    uint32_t loopRound = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t bsTailSize = 0;
    uint32_t bshCurLength = 0;
    uint32_t bsCurLength = 0;
    uint32_t gammaTempTensorPos = 0;
    uint32_t betaTempTensorPos = 0;
    uint32_t inputDyTmpTensorPos = 0;
    uint32_t resForGammaTmpTensorPos = 0;
    uint32_t reserved = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct LayerNormGradTiling {
    uint32_t stackBufferSize = 0;
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    uint32_t oneCalSize = 0;
    uint32_t nohCalSize = 0;
    uint32_t loopNum = 0;
    uint32_t tailSize = 0;
    uint32_t nohTailSize = 0;
    uint32_t tmpTensorBSHPos = 0;
    uint32_t tmpTensorBSHSize = 0;
    uint32_t pdVarTensorPos = 0;
    uint32_t pdVarTensorSize = 0;
    uint32_t pdMeanTensorPos = 0;
    uint32_t pdMeanTensorSize = 0;
    uint32_t x1TensorPos = 0;
    uint32_t x1TensorSize = 0;
    uint32_t x2TensorPos = 0;
    uint32_t x2TensorSize = 0;
    uint32_t x3TensorPos = 0;
    uint32_t x3TensorSize = 0;
    uint32_t tmpTensorPos = 0;
    uint32_t tmpTensorSize = 0;
    uint32_t tmpTensor1Pos = 0;
    uint32_t tmpTensor1Size = 0;
    uint32_t tmpTensor2Pos = 0;
    uint32_t tmpTensor2Size = 0;
    uint32_t lastDimValueBack = 0;
    uint32_t lastDimValueBackMulTwo = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct LayerNormTiling {
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    uint32_t inputXSize = 0;
    uint32_t meanVarSize = 0;
    uint32_t numberOfTmpBuf = 0;
    uint32_t meanTmpTensorPos = 0;
    uint32_t meanTmpTensorSize = 0;
    uint32_t varianceTmpTensorPos = 0;
    uint32_t varianceTmpTensorSize = 0;
    uint32_t tmpBufSize = 0;
    uint32_t oneTmpSize = 0;
    uint32_t firstTmpStartPos = 0;
    uint32_t secondTmpStartPos = 0;
    uint32_t thirdTmpStartPos = 0;
    uint32_t loopRound = 0;
    uint32_t inputRoundSize = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t meanVarRoundSize = 0;
    uint32_t meanVarTailSize = 0;
    uint32_t meanVarTailPos = 0;
    uint32_t bshCurLength = 0;
    uint32_t bsCurLength = 0;
    float lastDimValueBack = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct LayerNormSeparateTiling {
    uint32_t aLength = 0;
    uint32_t rLength = 0;
    uint32_t halfAddRepeatTimes = 0;
    uint32_t rHeadLength = 0;
    float k2Rec = 0;
    float k2RRec = 0;
    uint32_t inputXSize = 0;
    uint32_t meanVarSize = 0;
    uint32_t numberOfTmpBuf = 0;
    uint32_t varianceTmpTensorPos = 0;
    uint32_t varianceTmpTensorSize = 0;
    uint32_t tmpBufSize = 0;
    uint32_t oneTmpSize = 0;
    uint32_t firstTmpStartPos = 0;
    uint32_t secondTmpStartPos = 0;
    uint32_t thirdTmpStartPos = 0;
    uint32_t loopRound = 0;
    uint32_t inputRoundSize = 0;
    uint32_t inputTailSize = 0;
    uint32_t inputTailPos = 0;
    uint32_t meanVarRoundSize = 0;
    uint32_t meanVarTailSize = 0;
    uint32_t meanVarTailPos = 0;
    uint32_t arCurLength = 0;
    uint32_t aCurLength = 0;
    float rValueBack = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct RmsNormTiling {
    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;
    float reciprocalOfHLength = 0;
    uint32_t mainBshLength = 0;
    uint32_t mainBsLength = 0;
    uint32_t mainBsLengthAlign = 0;
    uint32_t loopRound = 0;
    uint32_t inputTailPos = 0;
    uint32_t tailBshLength = 0;
    uint32_t tailBsLength = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct UnPadTiling {
    uint32_t srcHeight = 0;
    uint32_t srcWidth = 0;
    uint32_t tmpBuffer1BlockNum = 0;
    uint32_t tmpBuffer1RowNum = 0;
    uint32_t tmpBuffer2Offset = 0;
    uint32_t widthTiling = 0;
    uint32_t widthFractal = 0;
    uint32_t widthFractalTail = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct PadTiling {
    uint32_t srcHeight = 0;
    uint32_t srcWidth = 0;
    uint32_t srcOriWidth = 0;
    uint32_t widthWithoutLastBlock = 0;
    uint32_t blocksPerRow = 0;
    uint32_t heightTiling = 0;
    uint32_t heightFractal = 0;
    uint32_t heightFractalTail = 0;
    uint32_t mainLoopOffset = 0;
    uint32_t tailBlockOffset = 0;
    uint32_t tmpBuffer1BlockNum = 0;
    uint32_t tmpBuffer1RowNum = 0;
    uint32_t tmpBuffer2Offset = 0;
    uint32_t widthTiling = 0;
    uint32_t widthFractal = 0;
    uint32_t widthFractalTail = 0;
    uint32_t widthFractalTailAlingned = 0;
    uint32_t brcbTiling = 0;
    uint32_t brcbFractal = 0;
    uint32_t brcbFractalTail = 0;
    uint32_t maxRepeatTimes = 0;
    uint32_t brcbTilingRepeatTimes = 0;
    uint32_t brcbTilingRepeatTimesTail = 0;
    uint32_t brcbFractalTailRepeatTimes = 0;
    uint32_t brcbFractalTailRepeatTimesTail = 0;
    uint32_t reserved = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct TopkTiling {
    int32_t tmpLocalSize = 0;
    int32_t allDataSize = 0;
    int32_t innerDataSize = 0;
    uint32_t sortRepeat = 0;
    int32_t mrgSortRepeat = 0;
    int32_t kAlignFourBytes = 0;
    int32_t kAlignTwoBytes = 0;
    int32_t maskOffset = 0;
    int32_t maskVreducev2FourBytes = 0;
    int32_t maskVreducev2TwoBytes = 0;
    int32_t mrgSortSrc1offset = 0;
    int32_t mrgSortSrc2offset = 0;
    int32_t mrgSortSrc3offset = 0;
    int32_t mrgSortTwoQueueSrc1Offset = 0;
    int32_t mrgFourQueueTailPara1 = 0;
    int32_t mrgFourQueueTailPara2 = 0;
    int32_t srcIndexOffset = 0;
    uint32_t copyUbToUbBlockCount = 0;
    int32_t topkMrgSrc1MaskSizeOffset = 0;
    int32_t topkNSmallSrcIndexOffset = 0;
    uint32_t vreduceValMask0 = 0;
    uint32_t vreduceValMask1 = 0;
    uint32_t vreduceIdxMask0 = 0;
    uint32_t vreduceIdxMask1 = 0;
    uint16_t vreducehalfValMask0 = 0;
    uint16_t vreducehalfValMask1 = 0;
    uint16_t vreducehalfValMask2 = 0;
    uint16_t vreducehalfValMask3 = 0;
    uint16_t vreducehalfValMask4 = 0;
    uint16_t vreducehalfValMask5 = 0;
    uint16_t vreducehalfValMask6 = 0;
    uint16_t vreducehalfValMask7 = 0;
};
#pragma pack(pop)
#pragma pack(push, 8)
struct ConfusionTransposeTiling {
    uint32_t param0 = 0;
    uint32_t param1 = 0;
    uint32_t param2 = 0;
    uint32_t param3 = 0;
    uint32_t param4 = 0;
    uint32_t param5 = 0;
    uint32_t param6 = 0;
    uint32_t param7 = 0;
    uint32_t param8 = 0;
    uint32_t param9 = 0;
    uint32_t param10 = 0;
    uint32_t param11 = 0;
    uint32_t param12 = 0;
    uint32_t param13 = 0;
    uint32_t param14 = 0;
    uint32_t param15 = 0;
    uint32_t param16 = 0;
    uint32_t param17 = 0;
};
#pragma pack(pop)
#endif
