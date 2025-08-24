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
 * \file buffer.h
 * \brief
 */
#ifndef ATP_BUFFER_H_
#define ATP_BUFFER_H_

namespace AscendC {
enum class MemLevel : uint8_t {
  LEVEL_0 = 0,
  LEVEL_1 = 1,
  LEVEL_2 = 2,
};

constexpr static uint32_t BUF_TYPE_MTE2 = 0b00'001;
constexpr static uint32_t BUF_TYPE_MTE3 = 0b00'010;
constexpr static uint32_t BUF_TYPE_TEMP = 0b00'100;
constexpr static uint32_t BUF_TYPE_PLACEHOLDER = 0b01'000;
constexpr static uint32_t BUF_TYPE_SCALAR = 0b10'000;

constexpr static uint32_t BUF_PING = 0b000'01;
constexpr static uint32_t BUF_PONG = 0b000'10;

constexpr static uint8_t BUF_COMBINE_SHIFT = 5;
constexpr static uint32_t BUF_COMBINE_MASK = 0x1F;
constexpr static uint8_t BUF_COMBINED_MAX = 5;

constexpr static uint8_t BUF_PING_PONG = 2;
constexpr static uint8_t BUF_MAX_COUNT = 32;

enum class BufPosInList : uint8_t {
  MTE2 = 0,
  MTE3 = 1,
  TEMP = 2,
  PONG_MTE3 = 3,
};

constexpr static int BUF_ALLOCATED_IDX = 4;
constexpr static int BUF_TO_RELEASE_IDX = 4;

/*
* Buffer wrapper Tempalte
* Template Args:
* ID: buffer id
* T: buffer type
* Where: 0-ping, 1-pong
*/
template<int ID, uint32_t T, uint32_t Where = BUF_PING>
struct BufferWrapper {
  // < 0 means no buffer will be allocated, only a placeholder to make schedule happy.
  constexpr static int BufferId = ID;
  constexpr static uint32_t BufferType = T;
  constexpr static uint32_t PingPong = Where;
};

/*
* Template used to generate buffer wrappers.
* Template Args:
* N: max count
* T: buffer type
* Offset: buffer id offset
* Where: 0-ping, 1-pong
* Es: existed buffer wrappers
*/
template<int N, uint32_t T, uint32_t Offset = 0, uint32_t Where = BUF_PING, typename Es = Elems<>>
struct GenerateBufferWrappers {
  using Type = typename GenerateBufferWrappers<N - 1, T, Offset, Where, Es>::Type::template \
                        Append< BufferWrapper<N - 1 + Offset, T, Where> >;
};

template<uint32_t T, uint32_t Offset, uint32_t Where, typename Es>
struct GenerateBufferWrappers<0, T, Offset, Where, Es> {
  using Type = Elems<>;
};

template<uint32_t T, uint32_t Offset, uint32_t Where, typename Es>
struct GenerateBufferWrappers<-1, T, Offset, Where, Es> {
  using Type = Elems<>;
};

// 辅助模板类，记录Bind与该Bind释放时需要释放的Buffer对应关系
template <typename B, typename Es>
struct Mapping {
  using Bind = B;
  using Buffers = Es;
};

// 辅助模板，递归组合BufferWrapper
template <typename... Ts>
struct CombineBufferWrapper{};

template <typename T>
struct CombineBufferWrapper<T> {
  constexpr static uint32_t BufferId = static_cast<uint32_t>(T::BufferId);
  constexpr static uint32_t BufferType = T::BufferType;
  constexpr static uint32_t PingPong = T::PingPong;
};

template <typename T, typename... Ts>
struct CombineBufferWrapper<T, Ts...> {
  constexpr static uint32_t BufferId = static_cast<uint32_t>(T::BufferId) << (BUF_COMBINE_SHIFT * sizeof...(Ts)) | \
                                       CombineBufferWrapper<Ts...>::BufferId;
  constexpr static uint32_t BufferType = T::BufferType << (BUF_COMBINE_SHIFT * sizeof...(Ts)) | \
                                         CombineBufferWrapper<Ts...>::BufferType;
  constexpr static uint32_t PingPong = T::PingPong << (BUF_COMBINE_SHIFT * sizeof...(Ts)) | \
                                       CombineBufferWrapper<Ts...>::PingPong;;
};

// Combined buffer wrapper template
template <typename... Ts>
struct CombinedBufferWrappers: BufferWrapper<static_cast<int>(
                                              static_cast<uint32_t>(sizeof...(Ts)) << \
                                              (BUF_COMBINE_SHIFT * BUF_COMBINED_MAX) | \
                                              CombineBufferWrapper<Ts...>::BufferId),
                                             CombineBufferWrapper<Ts...>::BufferType,
                                             CombineBufferWrapper<Ts...>::PingPong> {
};

// 根据BufferType进行ID偏移计算Pong的BufferID
static constexpr int CalcPongBufferId(int bufferId, uint32_t bufferType,
                                      uint32_t pingPong, int pongOffset) {
  if (pingPong > BUF_PONG) {
    // combined buffer wrapper. type/id/pingPong will always non-negtive.
    if (pingPong == 0) {
      return 0;
    } else {
      const uint32_t id = static_cast<uint32_t>(bufferId) & BUF_COMBINE_MASK;
      const uint32_t type = bufferType & BUF_COMBINE_MASK;
      const uint32_t pp = pingPong & BUF_COMBINE_MASK;
      const uint32_t currentId = pp == BUF_PING ? (type == BUF_TYPE_TEMP ? id : (id + static_cast<uint32_t>(pongOffset))) \
                                            : (id - static_cast<uint32_t>(pongOffset));
      const int idNext = static_cast<int>(static_cast<uint32_t>(bufferId) >> BUF_COMBINE_SHIFT);
      const uint32_t typeNext = bufferType >> BUF_COMBINE_SHIFT;
      const uint32_t ppNext = pingPong >> BUF_COMBINE_SHIFT;
      return static_cast<int>(
                static_cast<uint32_t>(
                  CalcPongBufferId(idNext, typeNext, ppNext, pongOffset)
                ) << BUF_COMBINE_SHIFT | currentId
              );
    }
  } else {
    // pure buffer wrapper
    return pingPong == BUF_PING ? ((bufferType == BUF_TYPE_TEMP || bufferType == BUF_TYPE_SCALAR) ? \
                               bufferId : \
                               (bufferType == BUF_TYPE_PLACEHOLDER ? -1 : (bufferId + pongOffset))) \
                            : (bufferId - pongOffset);
  }
}

// 递归提取BufferID辅助类
template <typename Es, int PongOffset>
struct ExtractBufferId{};

template <typename... Ts, int PongOffset>
struct ExtractBufferId<Elems<Ts...>, PongOffset> {
  static constexpr size_t size = sizeof...(Ts);
  constexpr static int arr[2][size] = {
    {Ts::BufferId...},
    {CalcPongBufferId(Ts::BufferId, Ts::BufferType, Ts::PingPong, PongOffset)...}
  };
  constexpr static const int* Value[2] = {arr[0], arr[1]};
};

// BufferID解码
template<int... Ints>
struct IntegerSequence {};

template<int N, int... Ints>
struct MakeIntegerSequenceAux : MakeIntegerSequenceAux<N-1, N-1, Ints...> {};

template<int... Ints>
struct MakeIntegerSequenceAux<0, Ints...>{
  using Type = IntegerSequence<Ints...>;
};

template<int N>
using MakeIntegerSequence = typename MakeIntegerSequenceAux<N>::Type;

template <int BufferId, uint32_t N, int pos>
struct DecodeBufferIdWithPos {
  constexpr static int Value = static_cast<int>(
                              static_cast<uint32_t>(BufferId) >> (BUF_COMBINE_SHIFT * (N - 1 - pos)) & \
                              BUF_COMBINE_MASK
                              );
};

template<int BufferId>
struct CombinedBufferCount {
  const static uint32_t tmp = static_cast<uint32_t>(BufferId) >> \
                                    (BUF_COMBINE_SHIFT * BUF_COMBINED_MAX) & \
                                    BUF_COMBINE_MASK;
  constexpr static uint32_t Value = (tmp == 0 || tmp > BUF_COMBINED_MAX) ? 1 : tmp;
};

template<int BufferId, typename intSeq>
struct DecodeBufferIdAux {};

template<int BufferId, int... Ints>
struct DecodeBufferIdAux<BufferId, IntegerSequence<Ints...> > {
  constexpr static int Value[sizeof...(Ints)] = {
    DecodeBufferIdWithPos<BufferId, sizeof...(Ints), Ints>::Value...
  };
};

template <int BufferId, uint32_t N = CombinedBufferCount<BufferId>::Value>
struct DecodeBufferId {
  using IntSequence = MakeIntegerSequence<static_cast<int>(N)>;
  constexpr static const int* const Value = {DecodeBufferIdAux<BufferId, IntSequence>::Value};
};

// 内存释放辅助类
template <typename inFuns, typename FunList, typename ToReleaseEs,
          int currentNodePos, uint32_t BufferType, bool cache_brc = false, uint32_t Where = BUF_PING>
struct ReleaseBufferByTypeAux{};

template <typename FunList, typename ToReleaseEs,
          int currentNodePos, uint32_t BufferType, bool cache_brc, uint32_t Where >
struct ReleaseBufferByTypeAux<Elems<>, FunList, ToReleaseEs,
                              currentNodePos, BufferType, cache_brc, Where> {
  using Type = Elems<>;
};

template <typename F, typename... Fs, typename FunList, typename ToReleaseEs,
          int currentNodePos, uint32_t BufferType, bool cache_brc, uint32_t Where >
struct ReleaseBufferByTypeAux<Elems<F, Fs...>, FunList, ToReleaseEs,
                              currentNodePos, BufferType, cache_brc, Where> {
protected:
  // 在ToReleaseEs中找到指定节点F
  template <class Target, class T>
  struct BindEqual {
    constexpr static bool Value = __aux::IsSameType<F, typename T::Bind>::Value;
  };

  // 在指定节点F待释放内存列表中找到指定Type的内存
  template <class Target, class T>
  struct BufferTypeEqual {
    constexpr static bool Value = BufferType == T::BufferType && T::PingPong == Where;
  };

  constexpr static bool isCacheBrcNode = cache_brc && (Vec::IsCopyInBrcOp<typename F::Fun>::Value ||
                                                       Vec::IsVecBrcOp<typename F::Fun>::Value);
  constexpr static bool ableToRelease = !isCacheBrcNode && \
                                         __aux::InputIsAbleToFreeAux<FunList, currentNodePos + 1, F>();
  // 找到指定的mapping列表
  using mappings = typename ToReleaseEs::template Filter< BindEqual >;
  static_assert(mappings::Size == 1, "mapping::Size == 1");
  // 在mapping映射的内存列表中找到指定类型的内存
  using mapping = typename mappings::template At<0>;
  using buffers = typename mapping::Buffers::template Filter< BufferTypeEqual >;
  using left = typename ReleaseBufferByTypeAux<Elems<Fs...>, FunList, ToReleaseEs,
                                               currentNodePos, BufferType, cache_brc, Where>::Type;

public:
  using Type = typename __aux::Condition<ableToRelease, buffers, Elems<>>::template Union<left>;
};

// 释放指定类型的Buffer
template <typename FunList, typename ToReleaseEs,
          int currentNodePos, uint32_t BufferType, bool cache_brc = false, uint32_t Where = BUF_PING>
struct ReleaseUnusedBufferByType {
  using fun = typename FunList::template At<currentNodePos>;
  using inFuns = typename fun::InNonScalarFuns;
  using Type = typename ReleaseBufferByTypeAux<inFuns, FunList, ToReleaseEs,
                                               currentNodePos, BufferType, cache_brc, Where>::Type;
};

// 从ToReleaseEs中删除已释放节点 辅助类
template <typename inFuns, typename FunList, typename ToReleaseEs,
          int currentNodePos, bool cache_brc>
struct ReleaseUnusedInputAux{};

template <typename FunList, typename ToReleaseEs,
          int currentNodePos, bool cache_brc>
struct ReleaseUnusedInputAux<Elems<>, FunList, ToReleaseEs, currentNodePos, cache_brc> {
  using Type = Elems<>;
};

template <typename F, typename... Fs, typename FunList, typename ToReleaseEs,
          int currentNodePos, bool cache_brc>
struct ReleaseUnusedInputAux<Elems<F, Fs...>, FunList, ToReleaseEs, currentNodePos, cache_brc> {
protected:
  // 在ToReleaseEs中找到指定节点F
  template <class Target, class T>
  struct BindEqual {
    constexpr static bool Value = __aux::IsSameType<F, typename T::Bind>::Value;
  };

  constexpr static bool isCacheBrcNode = cache_brc && (Vec::IsCopyInBrcOp<typename F::Fun>::Value ||
                                                       Vec::IsVecBrcOp<typename F::Fun>::Value);
  constexpr static bool ableToRelease = !isCacheBrcNode && \
                                        __aux::InputIsAbleToFreeAux<FunList, currentNodePos + 1, F>();
  // 找到指定的mapping列表
  using mappings = typename ToReleaseEs::template Filter< BindEqual >;
  // same input Bind to one VEC-API scenario will be 0.
  static_assert(mappings::Size <= 1, "mapping::Size <= 1");
  using left = typename ReleaseUnusedInputAux<Elems<Fs...>, FunList, ToReleaseEs,
                                              currentNodePos, cache_brc>::Type;

public:
  using Type = typename __aux::Condition<ableToRelease, mappings, Elems<>>::template Union<left>;
};

// 从ToReleaseEs中删除已释放节点
template <typename FunList, typename ToReleaseEs, int currentNodePos, bool cache_brc>
struct ReleaseUnusedInput {
  using fun = typename FunList::template At<currentNodePos>;
  using inFuns = typename fun::InNonScalarFuns;
  using needReleaseEs = typename ReleaseUnusedInputAux<inFuns, FunList, ToReleaseEs,
                                                       currentNodePos, cache_brc>::Type;
  using Type = typename ToReleaseEs::template Remove<needReleaseEs>::Type;
};

// 辅助模板类，递归模板来检查Elems列表是否至少有一个大于0
template <typename... Eses>
struct CheckEsListSize{};

template <>
struct CheckEsListSize<> {
  constexpr static bool Value = false;
};

template <typename First, typename... Rest>
struct CheckEsListSize<First, Rest...> {
  constexpr static bool rest = CheckEsListSize<Rest...>::Value;
  constexpr static bool Value = (First::Size > 0) || rest;
};

// 辅助模板类，按照优先级从Elems列表Pop出1个元素
template <typename Es>
struct PriorityGetFrontAux{};

template <typename T>
struct PriorityGetFrontAux<Elems<T>> {
  using Type = typename T::template At<0>;
};

template <typename T, typename... Ts>
struct PriorityGetFrontAux<Elems<T, Ts...>> {
  using left = typename PriorityGetFrontAux<Elems<Ts...>>::Type;
  using Type = __aux::Condition<(T::Size > 0),
                                typename T::template At<0>,
                                left>;
};

// 辅助模板类，从多个Elems中Pop出1个元素
template <typename... Eses>
struct PriorityGetFront {
  static_assert(CheckEsListSize<Eses...>::Value,
                "At least one memory Es should not be empty.");
  using Type = typename PriorityGetFrontAux<Elems<Eses...>>::Type;
};

// 分配MTE2内存
template<typename FunList, typename BufListList, MemLevel MemLvl>
struct AllocMte2 {
private:
  using Mte2Es = typename BufListList::template At<static_cast<int>(BufPosInList::MTE2)>;
  using Mte3Es = typename BufListList::template At<static_cast<int>(BufPosInList::MTE3)>;
  using TmpEs = typename BufListList::template At<static_cast<int>(BufPosInList::TEMP)>;
  using PongMte3Es = typename BufListList::template At<static_cast<int>(BufPosInList::PONG_MTE3)>;
  // pop front
  using usedTmpEs = __aux::Condition<MemLvl == MemLevel::LEVEL_2, Elems<>, TmpEs>;
  using usedPongMte3Es = __aux::Condition<MemLvl != MemLevel::LEVEL_0, Elems<>, PongMte3Es>;
  using mte2 = typename PriorityGetFront<Mte2Es, usedTmpEs, usedPongMte3Es>::Type;
  // update Es
  using Mte2EsNext = typename Mte2Es::template PopFront<mte2>::Type;
  using TmpEsNext = typename TmpEs::template PopFront<mte2>::Type;
  using PongMte3EsNext = typename PongMte3Es::template PopFront<mte2>::Type;
public:
  using Type = Elems<Mte2EsNext, Mte3Es, TmpEsNext, PongMte3EsNext, mte2>;
};

// 分配Temp内存
template<typename FunList, typename BufListList, MemLevel MemLvl>
struct AllocTempBuffer {
private:
  using Mte2Es = typename BufListList::template At<static_cast<int>(BufPosInList::MTE2)>;
  using Mte3Es = typename BufListList::template At<static_cast<int>(BufPosInList::MTE3)>;
  using TmpEs = typename BufListList::template At<static_cast<int>(BufPosInList::TEMP)>;
  using PongMte3Es = typename BufListList::template At<static_cast<int>(BufPosInList::PONG_MTE3)>;
  // pop front
  using usedPongMte3Es = __aux::Condition<MemLvl != MemLevel::LEVEL_0, Elems<>, PongMte3Es>;
  using usedMte2Es = __aux::Condition<MemLvl == MemLevel::LEVEL_2, Elems<>, Mte2Es>;
  using tmp = typename PriorityGetFront<TmpEs, usedPongMte3Es, usedMte2Es>::Type;
  // update Es
  using TmpEsNext = typename TmpEs::template PopFront<tmp>::Type;
  using PongMte3EsNext = typename PongMte3Es::template PopFront<tmp>::Type;
  using Mte2EsNext = typename Mte2Es::template PopFront<tmp>::Type;
public:
  using Type = Elems<Mte2EsNext, Mte3Es, TmpEsNext, PongMte3EsNext, tmp>;
};

// 分配Mte3内存
template<typename FunList, typename BufListList, MemLevel MemLvl>
struct AllocMte3 {
private:
  using Mte2Es = typename BufListList::template At<static_cast<int>(BufPosInList::MTE2)>;
  using Mte3Es = typename BufListList::template At<static_cast<int>(BufPosInList::MTE3)>;
  using TmpEs = typename BufListList::template At<static_cast<int>(BufPosInList::TEMP)>;
  using PongMte3Es = typename BufListList::template At<static_cast<int>(BufPosInList::PONG_MTE3)>;
  // pop front
  using usedTmpEs = __aux::Condition<MemLvl != MemLevel::LEVEL_0, Elems<>, TmpEs>;
  using usedPongMte3Es = __aux::Condition<MemLvl != MemLevel::LEVEL_0, Elems<>, PongMte3Es>;
  using usedMte2Es = __aux::Condition<MemLvl != MemLevel::LEVEL_0, Elems<>, Mte2Es>;
  using mte3 = typename PriorityGetFront<Mte3Es, usedTmpEs, usedPongMte3Es, usedMte2Es>::Type;
  // update Es
  using Mte3EsNext = typename Mte3Es::template PopFront<mte3>::Type;
  using TmpEsNext = typename TmpEs::template PopFront<mte3>::Type;
  using PongMte3EsNext = typename PongMte3Es::template PopFront<mte3>::Type;
  using Mte2EsNext = typename Mte2Es::template PopFront<mte3>::Type;
public:
  using Type = Elems<Mte2EsNext, Mte3EsNext, TmpEsNext, PongMte3EsNext, mte3>;
};

// 释放并更新内存队列
template<typename FunList, typename BufListList, typename ToReleaseEs,
         bool cache_brc, int funPos>
struct ReleaseAndUpdateEs {
private:
  // release & update
  using Mte2EsNext = typename BufListList::template At<static_cast<int>(BufPosInList::MTE2)>::template \
                                Concat<typename ReleaseUnusedBufferByType<FunList, ToReleaseEs,
                                       funPos, BUF_TYPE_MTE2, cache_brc>::Type>;
  using Mte3EsNext = typename BufListList::template At<static_cast<int>(BufPosInList::MTE3)>::template \
                                Concat<typename ReleaseUnusedBufferByType<FunList, ToReleaseEs,
                                       funPos, BUF_TYPE_MTE3, cache_brc>::Type>;
  using TmpEsNext = typename BufListList::template At<static_cast<int>(BufPosInList::TEMP)>::template \
                                Concat<typename ReleaseUnusedBufferByType<FunList, ToReleaseEs,
                                       funPos, BUF_TYPE_TEMP, cache_brc>::Type>;
  using PongMte3EsNext = typename BufListList::template At<static_cast<int>(BufPosInList::PONG_MTE3)>::template \
                               Concat<typename ReleaseUnusedBufferByType<FunList, ToReleaseEs,
                                      funPos, BUF_TYPE_MTE3, cache_brc, BUF_PONG>::Type>;
  // update ToReleaseEs to speedup next release
  using ToReleaseEsNext = typename ReleaseUnusedInput<FunList, ToReleaseEs, funPos, cache_brc>::Type;
public:
  using Type = Elems<Mte2EsNext, Mte3EsNext, TmpEsNext, PongMte3EsNext, ToReleaseEsNext>;
};

/*
* 策略1/2/3遍历图生成BufferId列表
* 模板参数：
*   1. FunList: 完整的计算图执行节点列表
*   2. BufListList：空闲的 Mte2/Mte3/TmpBuffer/PongMte3 列表
*   3. PongOffset: Pong Buffer ID偏移量
*   4. MemLvl: 内存策略
*   5. use_nddma: CopyInBrc是否使用NDDMA
*   6. cache_brc: 是否Cache CopyInBrc & VecBrc节点
*   7. AllocEs: 当前已分配的BufferWrapper列表
*   8. ToReleaseEs: 待释放的Mapping<Bind, Elems<BufferWrapper...>>列表
*   9. scalarIdx: 下一个scalar索引
*  10. funPos: 当前执行节点位置
* 返回值：
*   1. std::array<int, N2>: 完整的计算图PingPong BufferId列表
*/
template<typename FunList, typename BufListList,
         int PongOffset, MemLevel MemLvl = MemLevel::LEVEL_2,
         bool use_nddma = true, bool cache_brc = false,
         typename AllocEs = Elems<>, typename ToReleaseEs = Elems<>,
#ifdef __ATP_UT__
         int scalarIdx = 50,
#else
         int scalarIdx = 0,
#endif
         int funPos = 0>
static constexpr const int* const *GenerateBufferIdOrder() {
  if constexpr (funPos < FunList::Size) {
    using fun = typename FunList::template At<funPos>;
    if constexpr (fun::IsScalarOp) {
      using AllocEsNext = typename AllocEs::template Append<BufferWrapper<scalarIdx, BUF_TYPE_SCALAR>>;
      // next
      return GenerateBufferIdOrder<FunList, BufListList, PongOffset, MemLvl,
                                   use_nddma, cache_brc, AllocEsNext, ToReleaseEs,
                                   scalarIdx + 1, funPos + 1>();
    } else if constexpr (Vec::IsCopyInBrcOp<typename fun::Fun>::Value && !use_nddma) {
      // use copyIn + ubBrc to implement nddma.
      using NextEs0 = typename AllocMte2<FunList, BufListList, MemLvl>::Type;
      using mte2 = typename NextEs0::template At<BUF_ALLOCATED_IDX>;
      if constexpr (__aux::IsConnectOutput<FunList, funPos + 1, fun>()) {
        // if CopyInBrc -> CopyOut, should use MTE3
        using NextEs = typename AllocMte3<FunList, NextEs0, MemLvl>::Type;
        using mte3 = typename NextEs::template At<BUF_ALLOCATED_IDX>;
        // add to alloc list
        using AllocEsNext = typename AllocEs::template Append<CombinedBufferWrappers<mte3, mte2>>;
        using ToReleaseEsNext = typename ToReleaseEs::template Append<Mapping<fun, Elems<mte3, mte2>>>;
        // next
        return GenerateBufferIdOrder<FunList, NextEs, PongOffset, MemLvl,
                                     use_nddma, cache_brc, AllocEsNext, ToReleaseEsNext,
                                     scalarIdx, funPos + 1>();
      } else {
        using NextEs = typename AllocTempBuffer<FunList, NextEs0, MemLvl>::Type;
        using tmp = typename NextEs::template At<BUF_ALLOCATED_IDX>;
        // add to alloc list
        using AllocEsNext = typename AllocEs::template Append<CombinedBufferWrappers<tmp, mte2>>;
        using ToReleaseEsNext = typename ToReleaseEs::template Append<Mapping<fun, Elems<tmp, mte2>>>;
        // next
        return GenerateBufferIdOrder<FunList, NextEs, PongOffset, MemLvl,
                                     use_nddma, cache_brc, AllocEsNext, ToReleaseEsNext,
                                     scalarIdx, funPos + 1>();
      }
    } else if constexpr (Vec::IsCopyInOp<typename fun::Fun>::Value) {
      using NextEs = typename AllocMte2<FunList, BufListList, MemLvl>::Type;
      using mte2 = typename NextEs::template At<BUF_ALLOCATED_IDX>;
      // add to alloc list
      using AllocEsNext = typename AllocEs::template Append<mte2>;
      using ToReleaseEsNext = typename ToReleaseEs::template Append<Mapping<fun, Elems<mte2>>>;
      // next
      return GenerateBufferIdOrder<FunList, NextEs, PongOffset, MemLvl,
                                   use_nddma, cache_brc, AllocEsNext, ToReleaseEsNext,
                                   scalarIdx, funPos + 1>();
    } else if constexpr (Vec::IsCopyOutOp<typename fun::Fun>::Value) {
      using NextEs = typename ReleaseAndUpdateEs<FunList, BufListList, ToReleaseEs, cache_brc, funPos>::Type;
      // release and update
      using ToReleaseEsNext = typename NextEs::template At<BUF_TO_RELEASE_IDX>;
      // fill -1 when no buffer is needed.
      using AllocEsNext = typename AllocEs::template Append<BufferWrapper<-1, BUF_TYPE_PLACEHOLDER>>;
      // next
      return GenerateBufferIdOrder<FunList, NextEs, PongOffset, MemLvl,
                                   use_nddma, cache_brc, AllocEsNext, ToReleaseEsNext,
                                   scalarIdx, funPos + 1>();
    } else {
      if constexpr (__aux::IsConnectOutput<FunList, funPos + 1, fun>()) {
        using NextEs0 = typename AllocMte3<FunList, BufListList, MemLvl>::Type;
        using mte3 = typename NextEs0::template At<BUF_ALLOCATED_IDX>;
        // add to alloc list
        using AllocEsNext = typename AllocEs::template Append<mte3>;
        // release and update
        using NextEs = typename ReleaseAndUpdateEs<FunList, NextEs0, ToReleaseEs, cache_brc, funPos>::Type;
        using ToReleaseEsNext = typename NextEs::template At<BUF_TO_RELEASE_IDX>::template \
                                          Append<Mapping<fun, Elems<mte3>>>;
        // next
        return GenerateBufferIdOrder<FunList, NextEs, PongOffset, MemLvl,
                                     use_nddma, cache_brc, AllocEsNext, ToReleaseEsNext,
                                     scalarIdx, funPos + 1>();
      } else {
        using NextEs0 = typename AllocTempBuffer<FunList, BufListList, MemLvl>::Type;
        using tmp = typename NextEs0::template At<BUF_ALLOCATED_IDX>;
        // add to alloc list
        using AllocEsNext = typename AllocEs::template Append<tmp>;
        // release and update
        using NextEs = typename ReleaseAndUpdateEs<FunList, NextEs0, ToReleaseEs, cache_brc, funPos>::Type;
        using ToReleaseEsNext = typename NextEs::template At<BUF_TO_RELEASE_IDX>::template \
                                          Append<Mapping<fun, Elems<tmp>>>;
        // next
        return GenerateBufferIdOrder<FunList, NextEs, PongOffset, MemLvl,
                                     use_nddma, cache_brc, AllocEsNext, ToReleaseEsNext,
                                     scalarIdx, funPos + 1>();
      }
    }
  } else {
    // generate buffer id list for ping-pong as per AllocEs.
    static_assert(FunList::Size == AllocEs::Size, "N == AllocEs::Size");
    return ExtractBufferId<AllocEs, PongOffset>::Value;
  }
}
} // namespace AscendC
#endif  // ATP_BUFFER_H_