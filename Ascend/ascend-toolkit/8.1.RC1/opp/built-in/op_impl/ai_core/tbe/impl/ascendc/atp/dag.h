/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file dag.h
 * \brief
 */
#ifndef ATP_DAG_H_
#define ATP_DAG_H_
#include "aux1.h"
#include "buffer.h"

namespace AscendC {

constexpr static int MAX_BUFFER_NUMBER = 10;

/*
* 若非Bind类型，则原封返回
*/
template <class T, typename = void>
struct GetRealArgs {
  using Type = T;
};

/*
* 检测Bind入参中Cast节点是否时相同Dtype中间Cast（NoOp）。
* 若是，则递归获取Cast节点的第一个输入Bind作为当前Bind的入参。
*/
template <class T>
struct GetRealArgs<T, __aux::Void_t<typename T::RealArgs>> {
  using Type =
      __aux::Condition<(Vec::IsCastNoOp<typename T::Fun>::Value),
                       typename GetRealArgs<typename T::RealArgs::template At<0>>::Type,
                       typename T::RealBindType>;
};

/**
 * @tparam MemLevel
 * @tparam ReuseIn
 */
template <MemLevel MemLvl = MemLevel::LEVEL_0, bool ReuseIn = false>
struct MemOptCfg {
  static constexpr MemLevel memoryLevel = MemLvl;
  static constexpr bool reuseIn = ReuseIn;
};

/**
 * @tparam Func 需要执行的向量操作函数
 * @tparam Ts Func的入参
 */
template <typename Func, typename... Ts>
struct Bind {
public:
  using Fun = Func;
  using Args = Elems<Ts...>;
  using BindType = Bind<Fun, Ts...>;
  constexpr static bool IsBindFun = true;

  using RealArgs = typename Args::template ForEach<GetRealArgs>;

private:
  template <typename... Rs>
  struct CreateRealBind;

  template <typename... Rs>
  struct CreateRealBind<Elems<Rs...>> {
    using Type = __aux::Condition<(Vec::IsCastNoOp<Func>::Value),
                                   typename RealArgs::template At<0>,
                                   Bind<Func, Rs...>>;
  };

public:
  // 消除入参中CastNoOp之后的BindType
  using RealBindType = typename CreateRealBind<RealArgs>::Type;

  // 第 @offset 个出参Dtype
  template <int offset>
  using FunRetArgType = typename Fun::template FunRetArgType<offset>;
  // 第 @offset 个入参Dtype
  template <int offset>
  using FunInArgType = typename Fun::template FunInArgType<offset>;

public:
  // 当前Bind的输出Dtype
  using OutDataType = FunRetArgType<0>;

  // 入参中输出Placeholder的列表（输出GM)
  using OutHolders = typename Args::template Filter<__aux::TypeIsOutHolder>;
  // 入参列表（Args中过滤掉Placeholder::Out)
  using InArgs = typename Args::template Filter<__aux::TypeIsInput>;
  // 入参中输入Placeholder的列表（输入GM）
  using InHolders = typename Args::template Filter<__aux::TypeIsInHolder>;
  // 入参中函数列表（其他Bind类型）
  using InFuns = typename Args::template Filter<__aux::TypeIsInFun>;
  // 入参中Scalar变量列表
  using Vars = typename Args::template Filter<__aux::TypeIsVar>;
  // 入参中Const输入列表
  using ConstValues = typename Args::template Filter<__aux::TypeIsConst>;
  // 入参中输入是TensorScalar的列表
  using InScalarHolders = typename InHolders::template Filter<__aux::TypeIsInScalarHolder>;
  // 入参中函数是Scalar操作的列表
  using InScalarFuns = typename InFuns::template Filter<__aux::TypeIsScalarBind>;
  // 入参中非ScalarOp函数列表
  using InNonScalarFuns = typename InFuns::template Remove<InScalarFuns>;

  static_assert(InArgs::Size == InHolders::Size + InFuns::Size + Vars::Size + ConstValues::Size, "why?");

  // 标识当前Bind是否是Scalar操作（不需要使用任何UB空间）
  constexpr static bool IsScalarOp = InScalarHolders::Size == InHolders::Size && \
                                     InScalarFuns::Size == InFuns::Size && \
                                     !Vec::IsDuplicateOp<Fun>::Value;

  // 入参个数
  constexpr static uint32_t InputSize = InArgs::Size;
  // 当前Bind的依赖列表：输入依赖 + 自身
  using DependFuns = typename __aux::GetDependFunsAux<InFuns>::Type::template Append<RealBindType>;
  // 溯源当前Bind的输入PlaceHolder
  using SourceInHolders = typename __aux::GetSourceInHoldersAux<InFuns>::Type::template Union<InHolders>;
  // 判断当前Bind的依赖中是否存在Broadcast节点
  constexpr static bool DependOnBrc = __aux::IsDependOnBrc<DependFuns>();
};

/*
* 统计首个计算节点之前，从GM搬运数据的次数
* 模板参数：
*   1. FunList:   计算顺序列表
*   2. start：    递归调用时，当前节点索引
*   3. Acc:       存放已经统计到的搬运Bind
*/
template <typename FunList, int start = 0, typename Acc = Elems<>>
constexpr int GetCopyInCountBeforeFirstCalcNode() {
  /**
   * 获取首个计算节点前，搬入次数。不需要对Cast做特殊处理
   * 但需要跳过 ScalarOp 节点
   */
  if constexpr (start < FunList::Size) {
    using func = typename FunList::template At<start>;
    if constexpr (func::IsScalarOp) {
      return GetCopyInCountBeforeFirstCalcNode<FunList, start + 1, Acc>();
    } else if constexpr (Vec::IsCopyInOp<typename func::Fun>::Value) {
      using Next = typename Acc::template Append<func>;
      return GetCopyInCountBeforeFirstCalcNode<FunList, start + 1, Next>();
    } else if constexpr (__aux::IsSameTemplateType<typename func::Fun, Vec::CopyOut>::Value) {
      return GetCopyInCountBeforeFirstCalcNode<FunList, start + 1, Acc>();
    } else {
      return Acc::Size;
    }
  }
  return Acc::Size;
};

/*
* 获取Bind上的RealBindType
*/
template <class T, typename = void>
struct GetRealBindType {
  using Type = T;
};

template <class T>
struct GetRealBindType<T, __aux::Void_t<typename T::RealBindType>> {
  using Type = typename T::RealBindType;
};

/*
* 判断是否CopyInBrc节点
*/
template <class Target, class T>
struct CheckCopyInBrc {
  constexpr static bool Value = Vec::IsCopyInBrcOp<typename T::Fun>::Value;
};

/*
* 判断是否VecBrc节点
*/
template <class Target, class T>
struct CheckVecBrc {
  constexpr static bool Value = Vec::IsVecBrcOp<typename T::Fun>::Value;
};

template <class Target, class T>
struct CheckVecReduce {
  constexpr static bool Value = Vec::IsReduceOp<typename T::Fun>::Value;
};

template <typename Holders, int at=0, int max=0>
static constexpr int GetInHolderMaxIdx() {
  if constexpr (at < Holders::Size) {
    using holder = typename Holders::template At<at>;
    constexpr int pos = holder::Pos;
    constexpr int curMax = pos > max ? pos : max;
    return GetInHolderMaxIdx<Holders, at + 1, curMax>();
  }
  return max;
}

/*
* DAG 单向无环图处理
* 模板参数：
*   1. OutList_:      单向无环图的输出列表
*   2. ComputeOrder_: 单向无环图函数执行列表，默认空。用户可指定执行序
*   3. MemOptCfg_:    内存复用策略配置
*/
template <typename OutList_, typename ComputeOrder_ = void,
          typename MemOptCfg_ = MemOptCfg<> >
struct DAGSch {
public:
  using OutList = typename OutList_::template ForEach< GetRealBindType >;
  using MemOpt = MemOptCfg_;
  static_assert(MemOpt::memoryLevel == MemLevel::LEVEL_0 || \
                MemOpt::memoryLevel == MemLevel::LEVEL_1 || \
                MemOpt::memoryLevel == MemLevel::LEVEL_2,
                "Buffer level should be in [0, 1, 2].");

  constexpr static bool HasComputeOrder = !__aux::IsSameType<ComputeOrder_, void>::Value;

private:
  // 通过输出列表，反向推导调用顺序列表
  using FunsAux = typename OutList::template Export<__aux::FunListAux>::Type;
  using FunListOriginal = __aux::Condition<__aux::IsSameType<ComputeOrder_, void>::Value, FunsAux, ComputeOrder_>;

public:
  // Filter CastNoOp
  using FunList =
      typename FunListOriginal::template ForEach<GetRealBindType>::Unique;

  // 统计 输入/输出 PlaceHolder 列表. 类型是Elems<PlaceHolder<In0>, PlaceHolder<In1>, ...>
  using InHolders = typename __aux::GetInHolder<FunList>::Type;
  using OutHolders = typename __aux::GetOutHolder<FunList>::Type;

  // 统计 CopyInBrc 及 VecBrc 列表
  using CopyBrcNodes = typename FunList::template Filter<CheckCopyInBrc>;
  using VecBrcNodes = typename FunList::template Filter<CheckVecBrc>;
  using VecReduceNodes = typename FunList::template Filter<CheckVecReduce>;
  // 统计 输入是 TensorScalar的列表
  using InScalarHolders = typename InHolders::template Filter<__aux::TypeIsInScalarHolder>::Unique;
  // FunList中ScalarOp列表
  using ScalarOpNodes = typename FunList::template Filter<__aux::TypeIsScalarBind>;

  // 统计 输入/输出 GM 数量； CopyInBrc 及 VecBrc 数量
  constexpr static uint32_t InputSize = InHolders::Size;
  constexpr static uint32_t OutputSize = OutHolders::Size;
  constexpr static uint32_t CopyBrcSize = CopyBrcNodes::Size;
  constexpr static uint32_t VecBrcSize = VecBrcNodes::Size;
  constexpr static int32_t ReduceOpPos = __aux::GetReducePosition<FunList>();
  using VecPreReduceNodes = typename __aux::FilterDagAux<FunList, ReduceOpPos, 0, __aux::CheckPre>::Type;
  using VecPostReduceNodes = typename __aux::FilterDagAux<FunList, ReduceOpPos, 0, __aux::CheckPost>::Type;
  constexpr static uint32_t TensorScalarSize = InScalarHolders::Size;
  // 刨去 TensorScalar 后的 输入GM 数量
  constexpr static uint32_t InputSizeWoScalar = InputSize - TensorScalarSize;

  // Scalar
  using Vars = typename __aux::GetVars<FunList>::Type;
  using VarType = typename Vars::template Export<Placeholder::VarTypeAux>::Type;
  constexpr static uint32_t VarSize = Vars::Size;

  // 检查某个输入节点是否可以释放（删除）
  template <int posInFuns, typename InArg>
  constexpr static bool ChkInputCanFree = __aux::InputIsAbleToFreeAux<FunList, posInFuns, InArg>();

  // 检查某个节点是否直连搬出节点.
  template <int posInFuns, typename InArg>
  constexpr static bool IsConnectOutput = __aux::IsConnectOutput<FunList, posInFuns, InArg>();

  // 某个CopyIn节点@InFun，若被后续某个VecBrc节点依赖，则返回VecBrc在VecBrcNodes中的索引位置，否则返回-1
  template <typename InFun>
  constexpr static int VecBrcIdxDepend = __aux::GetDependByVecBrcIdx<VecBrcNodes, InFun>();

  constexpr static auto MaxAliveNodeInfo = __aux::MaxAliveNode<FunList, OutList>(__aux::DagMaxAliveInfo());
  // 最大存活节点数量
  constexpr static uint32_t MaxAliveNode = MaxAliveNodeInfo.aliveNode;
  // 刨去 输入/输出 占用的Buffer，中间计算占用的最大临时节点数量
  constexpr static uint32_t TempCalcNode = MaxAliveNodeInfo.tempCalcNode;
  // 计算途中最大/最小字节数
  constexpr static uint32_t MaxDtypeBytes = MaxAliveNodeInfo.maxDtypeBytes;
  constexpr static uint32_t MinDtypeBytes = MaxAliveNodeInfo.minDtypeBytes;
  // NDDMA场景下最大存活节点数量 及 中间计算占用的最大临时节点数量（刨去了CopyInBrc上的TempNode）
  constexpr static uint32_t MaxAliveNodeForNddma = MaxAliveNodeInfo.aliveNodeNoCopyBrcTmpBuf;
  constexpr static uint32_t TempCalcNodeForNddma = MaxAliveNodeInfo.tempCalcNodeNoCopyBrcTmpBuf;

  // 首个计算节点前，搬运GM的节点数量
  constexpr static uint32_t GMCountBeforeFirstCalcNode = GetCopyInCountBeforeFirstCalcNode<FunList>();

private:
  constexpr static __aux::DagMaxAliveInfo GetAliveNodeInfoForCacheBrc() {
    if constexpr (CopyBrcSize == 0 && VecBrcSize == 0) {
      return MaxAliveNodeInfo;
    } else {
      return __aux::MaxAliveNode<FunList, OutList,
                                 typename CopyBrcNodes::template Union<VecBrcNodes>
                                >(__aux::DagMaxAliveInfo());
    }
  }
  constexpr static auto MaxAliveNodeInfoForCacheBrc = GetAliveNodeInfoForCacheBrc();

  constexpr static __aux::DagMaxAliveInfo GetAliveNodeInfoForPreReduce() {
    if constexpr (ReduceOpPos <= 0) {
      return MaxAliveNodeInfo;
    } else {
      return __aux::MaxAliveNode<VecPreReduceNodes, OutList>(__aux::DagMaxAliveInfo());
    }
  }
  constexpr static auto MaxAliveNodeInfoForPreReduce = GetAliveNodeInfoForPreReduce();

  constexpr static __aux::DagMaxAliveInfo GetAliveNodeInfoForPostReduce() {
    if constexpr (ReduceOpPos <= 0) {
      return MaxAliveNodeInfo;
    } else {
      return __aux::MaxAliveNode<VecPostReduceNodes, OutList>(__aux::DagMaxAliveInfo());
    }
  }
  constexpr static auto MaxAliveNodeInfoForPostReduce = GetAliveNodeInfoForPostReduce();

public:
  // CacheBrc场景下 存活节点/中间计算占用的临时节点数量统计
  constexpr static uint32_t MaxAliveNodeForCacheBrc = MaxAliveNodeInfoForCacheBrc.aliveNode;
  constexpr static uint32_t TempCalcNodeForCacheBrc = MaxAliveNodeInfoForCacheBrc.tempCalcNode;
  constexpr static uint32_t MaxAliveNodeForNddmaCacheBrc = MaxAliveNodeInfoForCacheBrc.aliveNodeNoCopyBrcTmpBuf;
  constexpr static uint32_t TempCalcNodeForNddmaCacheBrc = MaxAliveNodeInfoForCacheBrc.tempCalcNodeNoCopyBrcTmpBuf;
  constexpr static uint32_t PreReduceAliveNode = MaxAliveNodeInfoForPreReduce.aliveNode;
  constexpr static uint32_t PreReduceTempCalcNode = MaxAliveNodeInfoForPreReduce.tempCalcNode;
  constexpr static uint32_t PostReduceAliveNode = MaxAliveNodeInfoForPostReduce.aliveNode;
  constexpr static uint32_t PostReduceTempCalcNode = MaxAliveNodeInfoForPostReduce.tempCalcNode;

#ifdef __ATP_UT__
public:
#else
private:
#endif
  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetMaxAliveNodeSize() {
    if constexpr (use_nddma && cache_brc) {
      return MaxAliveNodeForNddmaCacheBrc;
    } else if constexpr (use_nddma && !cache_brc) {
      return MaxAliveNodeForNddma;
    } else if constexpr (!use_nddma && cache_brc) {
      return MaxAliveNodeForCacheBrc;
    } else { // !use_nddma && !cache_brc
      return MaxAliveNode;
    }
  }

  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetTempCalcNodeSize() {
    if constexpr (use_nddma && cache_brc) {
      return TempCalcNodeForNddmaCacheBrc;
    } else if constexpr (use_nddma && !cache_brc) {
      return TempCalcNodeForNddma;
    } else if constexpr (!use_nddma && cache_brc) {
      return TempCalcNodeForCacheBrc;
    } else { // !use_nddma && !cache_brc
      return TempCalcNode;
    }
  }

  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetFirstCopyOutNodeGMCount() {
    constexpr uint32_t maxAliveNodeSize = GetMaxAliveNodeSize<use_nddma, cache_brc>();
    return maxAliveNodeSize > GMCountBeforeFirstCalcNode ? 1 : 0;
  }

  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetLvl12Mte3Count() {
    // all CopyIn connects to CopyOut, so no MTE3 is needed.
    // NOTE: Scenario [some CopyIn connects to CopyOut, some not] is not considered.
    return FunList::Size == (InputSizeWoScalar + ScalarOpNodes::Size + OutputSize) ? 0 : OutputSize;
  }

  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetLvl1TmpSize() {
    constexpr uint32_t maxAliveNodeSize = GetMaxAliveNodeSize<use_nddma, cache_brc>();
    constexpr uint32_t tempCalcNodeSize = GetTempCalcNodeSize<use_nddma, cache_brc>();
    return tempCalcNodeSize > 0 ? (
            maxAliveNodeSize > InputSizeWoScalar ? maxAliveNodeSize - InputSizeWoScalar : 0
          ) : 0;
  }

  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetLvl0TmpSize() {
    constexpr uint32_t maxAliveNodeSize = GetMaxAliveNodeSize<use_nddma, cache_brc>();
    constexpr uint32_t firstCopyOutNodeGMCount = GetFirstCopyOutNodeGMCount<use_nddma, cache_brc>();
    return maxAliveNodeSize - (GMCountBeforeFirstCalcNode + firstCopyOutNodeGMCount);
  }

  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetBufferNumLevel0() {
    constexpr uint32_t maxAliveNodeSize = GetMaxAliveNodeSize<use_nddma, cache_brc>();
    constexpr uint32_t firstCopyOutNodeGMCount = GetFirstCopyOutNodeGMCount<use_nddma, cache_brc>();
    return maxAliveNodeSize + GMCountBeforeFirstCalcNode + firstCopyOutNodeGMCount;
  }

  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetBufferNumLevel1() {
    return GetLvl1TmpSize<use_nddma, cache_brc>() + \
            InputSizeWoScalar * BUF_PING_PONG + \
            GetLvl12Mte3Count<use_nddma, cache_brc>() * BUF_PING_PONG;
  }

  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetBufferNumLevel2() {
    constexpr uint32_t tempCalcNodeSize = GetTempCalcNodeSize<use_nddma, cache_brc>();
    constexpr uint32_t lvl12Mte3Count = GetLvl12Mte3Count<use_nddma, cache_brc>();
    return tempCalcNodeSize + InputSizeWoScalar * BUF_PING_PONG + lvl12Mte3Count * BUF_PING_PONG;
  }

#ifdef __ATP_UT__
public:
#endif
  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static MemLevel ChooseBufferLevel() {
    if constexpr (MemOpt::memoryLevel == MemLevel::LEVEL_0) {
      if constexpr (GetBufferNumLevel2<use_nddma, cache_brc>() <= MAX_BUFFER_NUMBER) {
        return MemLevel::LEVEL_2;
      } else if constexpr (GetBufferNumLevel1<use_nddma, cache_brc>() <= MAX_BUFFER_NUMBER) {
        return MemLevel::LEVEL_1;
      } else {
        return MemLevel::LEVEL_0;
      }
    } else {
      return MemOpt::memoryLevel;
    }
  }

public:
  constexpr static MemLevel BufLevel = ChooseBufferLevel<true, false>();

#ifdef __ATP_UT__
public:
#else
private:
#endif
  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetMte2Num() {
    if constexpr (ChooseBufferLevel<use_nddma, cache_brc>() == MemLevel::LEVEL_0) {
      return GMCountBeforeFirstCalcNode;
    } else {
      return InputSizeWoScalar;
    }
  }

  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetMte3Num() {
    if constexpr (ChooseBufferLevel<use_nddma, cache_brc>() == MemLevel::LEVEL_0) {
      return GetFirstCopyOutNodeGMCount<use_nddma, cache_brc>();
    } else {
      return GetLvl12Mte3Count<use_nddma, cache_brc>();
    }
  }

  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetTempBufNum() {
    constexpr MemLevel bufferLvl = ChooseBufferLevel<use_nddma, cache_brc>();
    if constexpr (bufferLvl == MemLevel::LEVEL_0) {
      return GetLvl0TmpSize<use_nddma, cache_brc>();
    } else if constexpr (bufferLvl == MemLevel::LEVEL_1) {
      return GetLvl1TmpSize<use_nddma, cache_brc>();
    } else {
      return GetTempCalcNodeSize<use_nddma, cache_brc>();
    }
  }

public:
  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static uint32_t GetBufferNum() {
    constexpr MemLevel bufferLvl = ChooseBufferLevel<use_nddma, cache_brc>();
    if constexpr (bufferLvl == MemLevel::LEVEL_0) {
      return GetBufferNumLevel0<use_nddma, cache_brc>();
    } else if constexpr (bufferLvl == MemLevel::LEVEL_1) {
      return GetBufferNumLevel1<use_nddma, cache_brc>();
    } else {// bufferLvl == 2
      return GetBufferNumLevel2<use_nddma, cache_brc>();
    }
  }

public:
  constexpr static uint32_t BufferNum = GetBufferNum<true, false>();
  constexpr static uint32_t Mte2Num = GetMte2Num<true, false>();
  constexpr static uint32_t Mte3Num = GetMte3Num<true, false>();

public:
  template <bool use_nddma = true, bool cache_brc = false>
  constexpr static const int* const *GetBufferIds() {
    // |mte2|mte3|tmp|mte2|mte3|
    constexpr MemLevel bufferLvl = ChooseBufferLevel<use_nddma, cache_brc>();
    constexpr uint32_t mte2Count = GetMte2Num<use_nddma, cache_brc>();
    constexpr uint32_t mte3Count = GetMte3Num<use_nddma, cache_brc>();
    constexpr uint32_t tempBufCount = GetTempBufNum<use_nddma, cache_brc>();
    static_assert(((mte2Count + mte3Count) * BUF_PING_PONG + tempBufCount) <= BUF_MAX_COUNT,
                  "Buffer count exceeded 32. Please try to switch MemLevel to LEVEL_1 or LEVEL_2.");
    constexpr uint32_t PongOffset = mte2Count + mte3Count + tempBufCount;
    using Mte2Es = typename GenerateBufferWrappers<mte2Count, BUF_TYPE_MTE2>::Type;
    using Mte3Es = typename GenerateBufferWrappers<mte3Count, BUF_TYPE_MTE3, mte2Count>::Type;
    using TmpEs = typename GenerateBufferWrappers<tempBufCount, BUF_TYPE_TEMP,
                                                  mte2Count + mte3Count >::Type;
    using PongMte3Es = __aux::Condition<bufferLvl == MemLevel::LEVEL_0,
                                        typename GenerateBufferWrappers<
                                            mte3Count, BUF_TYPE_MTE3,
                                            mte2Count * BUF_PING_PONG + mte3Count + tempBufCount,
                                            BUF_PONG>::Type,
                                        Elems<> >;
    return GenerateBufferIdOrder<FunList,
                                 Elems<Mte2Es, Mte3Es, TmpEs, PongMte3Es>,
                                 PongOffset, bufferLvl,
                                 use_nddma, cache_brc>();
  }
public:
  constexpr static auto BufferIds = GetBufferIds<true, false>();
};

}  // namespace AscendC
#endif  // ATP_DAG_H_
