//===----------------------------------------------------------------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2019-2023. All rights reserved.
//
// Interface design document refer to :
// wiki-davinci-llvm-project-CCEC-intrinsics-interface-design-doc
//===----------------------------------------------------------------------===//

#ifndef __CLANG_CCE_VECTOR_INTRINSICS_H
#define __CLANG_CCE_VECTOR_INTRINSICS_H

#include "__clang_cce_vector_types.h"
#include <stdint.h>

#define CCE_INTRINSIC                                                          \
  [aicore] static __attribute__((overloadable, cce_builtin_api, always_inline))

#define ULL unsigned long long

#include <type_traits>
enum class Mode { UNKNOWN_VALUE, MERGING_VALUE, ZEROING_VALUE };
typedef std::integral_constant<Mode, Mode::UNKNOWN_VALUE> Mode_Unknown_Type;
typedef std::integral_constant<Mode, Mode::MERGING_VALUE> Mode_Merging_Type;
typedef std::integral_constant<Mode, Mode::ZEROING_VALUE> Mode_Zeroing_Type;
#define MODE_UNKNOWN Mode_Unknown_Type()
#define MODE_MERGING Mode_Merging_Type()
#define MODE_ZEROING Mode_Zeroing_Type()

enum class Pos { LOWEST, HIGHEST };
typedef std::integral_constant<Pos, Pos::LOWEST> Lowest_Type;
typedef std::integral_constant<Pos, Pos::HIGHEST> Highest_Type;
#define POS_LOWEST Lowest_Type()
#define POS_HIGHEST Highest_Type()

enum class StoreMode { NOSTORED, STORED };
typedef std::integral_constant<StoreMode, StoreMode::NOSTORED> NoStoredType;
typedef std::integral_constant<StoreMode, StoreMode::STORED> StoredType;
#define MODE_NO_STORED NoStoredType()
#define MODE_STORED StoredType()

enum class HiloPart { Lower, Higher };
typedef std::integral_constant<HiloPart, HiloPart::Lower> Lower_Type;
typedef std::integral_constant<HiloPart, HiloPart::Higher> Higher_Type;
#define LOWER Lower_Type()
#define HIGHER Higher_Type()

enum class Post { NO_POST_UPDATE_VALUE, POST_UPDATE_VALUE };
typedef std::integral_constant<Post, Post::NO_POST_UPDATE_VALUE>
    NoPostUpdateType;
typedef std::integral_constant<Post, Post::POST_UPDATE_VALUE> PostUpdateType;
#define NO_POST_UPDATE NoPostUpdateType()
#define POST_UPDATE PostUpdateType()

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define DEPRECATED_AFTER_V210 __attribute__((deprecated))
#else
#define DEPRECATED_AFTER_V210
#endif

#define DEPRECATED __attribute__((deprecated))
#define NOT_DEPRECATED

[aicore] constexpr bool isV300Target() {
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
  return true;
#else
  return false;
#endif
}

[aicore] constexpr bool isV310Target() {
#if defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
  return true;
#else
  return false;
#endif
}

[aicore] constexpr bool isV210Target() {
#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
  return true;
#else
  return false;
#endif
}

[aicore] constexpr bool isSoftwareMergeMode() {
  return isV300Target() || isV310Target();
}

//----------------------------------------------------------------------------//
//  vintlv
//----------------------------------------------------------------------------//
#define VINTLV(TYPE, NUM)                                                      \
  CCE_INTRINSIC void vintlv(vector_##TYPE &dst0, vector_##TYPE &dst1,          \
                            vector_##TYPE src0, vector_##TYPE src1) {          \
    vector_##TYPE##x2_t __ret;                                                 \
    __builtin_cce_vintlv_##NUM##TYPE(&__ret, src0, src1);                      \
    dst0 = __ret.val[0];                                                       \
    dst1 = __ret.val[1];                                                       \
    return;                                                                    \
  }

VINTLV(s32, v64)
VINTLV(u32, v64)
VINTLV(s16, v128)
VINTLV(u16, v128)
VINTLV(s8, v256)
VINTLV(u8, v256)
VINTLV(f32, v64)
VINTLV(f16, v128)
#undef VINTLV

//----------------------------------------------------------------------------//
//  vdintlv
//----------------------------------------------------------------------------//
#define VDINTLV(TYPE, NUM)                                                     \
  CCE_INTRINSIC void vdintlv(vector_##TYPE &dst0, vector_##TYPE &dst1,         \
                             vector_##TYPE src0, vector_##TYPE src1) {         \
    vector_##TYPE##x2_t __ret;                                                 \
    __builtin_cce_vdintlv_##NUM##TYPE(&__ret, src0, src1);                     \
    dst0 = __ret.val[0];                                                       \
    dst1 = __ret.val[1];                                                       \
    return;                                                                    \
  }

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) ||                      \
    defined(__DAV_T210__) || defined(__DAV_M300__) || defined(__DAV_L300__) || \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
VDINTLV(s64, v32)
VDINTLV(u64, v32)
#endif
VDINTLV(s32, v64)
VDINTLV(u32, v64)
VDINTLV(s16, v128)
VDINTLV(u16, v128)
VDINTLV(s8, v256)
VDINTLV(u8, v256)
VDINTLV(f32, v64)
VDINTLV(f16, v128)
#undef VDINTLV

//----------------------------------------------------------------------------//
//  pge
//----------------------------------------------------------------------------//
enum class Pat {
  ALL,
  VL1,
  VL2,
  VL3,
  VL4,
  VL8,
  VL16,
  VL32,
  VL64,
  VL128,
  M3,
  M4,
  H,
  Q,
  ALLF = 15
};

typedef std::integral_constant<Pat, Pat::ALL> PatALLType;
typedef std::integral_constant<Pat, Pat::VL1> PatVL1Type;
typedef std::integral_constant<Pat, Pat::VL2> PatVL2Type;
typedef std::integral_constant<Pat, Pat::VL3> PatVL3Type;
typedef std::integral_constant<Pat, Pat::VL4> PatVL4Type;
typedef std::integral_constant<Pat, Pat::VL8> PatVL8Type;
typedef std::integral_constant<Pat, Pat::VL16> PatVL16Type;
typedef std::integral_constant<Pat, Pat::VL32> PatVL32Type;
typedef std::integral_constant<Pat, Pat::VL64> PatVL64Type;
typedef std::integral_constant<Pat, Pat::VL128> PatVL128Type;
typedef std::integral_constant<Pat, Pat::M3> PatM3Type;
typedef std::integral_constant<Pat, Pat::M4> PatM4Type;
typedef std::integral_constant<Pat, Pat::H> PatHType;
typedef std::integral_constant<Pat, Pat::Q> PatQType;
typedef std::integral_constant<Pat, Pat::ALLF> PatALLFType;

#define PAT_ALL PatALLType()
#define PAT_VL1 PatVL1Type()
#define PAT_VL2 PatVL2Type()
#define PAT_VL3 PatVL3Type()
#define PAT_VL4 PatVL4Type()
#define PAT_VL8 PatVL8Type()
#define PAT_VL16 PatVL16Type()
#define PAT_VL32 PatVL32Type()
#define PAT_VL64 PatVL64Type()
#define PAT_VL128 PatVL128Type()
#define PAT_M3 PatM3Type()
#define PAT_M4 PatM4Type()
#define PAT_H PatHType()
#define PAT_Q PatQType()
#define PAT_ALLF PatALLFType()

#define INVALID_VALUE_PGE "The input argument of PGE is not valid"
#define INVALID_VALUE_PSET "The input argument of PSET is not valid"
#define INVALID_VALUE_PREDICATE_MODE                                           \
  "The last argument can only be 'MODE_ZEROING', 'MODE_UNKNOWN', "             \
  "'MODE_MERGING' or empty."
#define INVALID_VALUE_V210_MODE                                                \
  "V210 only support MODE_MERGING and MODE_UNKNOWN"
#define INVALID_VALUE_PART "The value of PART is not valid"

#define PAT_STATIC_ASSERT                                                      \
  (std::is_same<T, PatALLType>::value || std::is_same<T, PatVL1Type>::value || \
   std::is_same<T, PatVL2Type>::value || std::is_same<T, PatVL3Type>::value || \
   std::is_same<T, PatVL4Type>::value || std::is_same<T, PatVL8Type>::value || \
   std::is_same<T, PatVL16Type>::value ||                                      \
   std::is_same<T, PatVL32Type>::value ||                                      \
   std::is_same<T, PatVL64Type>::value ||                                      \
   std::is_same<T, PatVL128Type>::value ||                                     \
   std::is_same<T, PatM3Type>::value || std::is_same<T, PatM4Type>::value ||   \
   std::is_same<T, PatHType>::value || std::is_same<T, PatQType>::value ||     \
   std::is_same<T, PatALLFType>::value)

template <class T>
DEPRECATED_AFTER_V210 CCE_INTRINSIC vector_bool pge_b8(T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PGE);
  static_assert(PAT_STATIC_ASSERT, INVALID_VALUE_PGE);
  if (isV210Target())
    return pge_b8_v210((ULL)dist.value, 0);
  return pge_b8((ULL)dist.value, 0);
}

template <class T>
DEPRECATED_AFTER_V210 CCE_INTRINSIC vector_bool pge_b16(T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PGE);
  static_assert(PAT_STATIC_ASSERT, INVALID_VALUE_PGE);
  if (isV210Target())
    return pge_b16_v210((ULL)dist.value, 0);
  return pge_b16((ULL)dist.value, 0);
}

template <class T>
DEPRECATED_AFTER_V210 CCE_INTRINSIC vector_bool pge_b32(T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PGE);
  static_assert(PAT_STATIC_ASSERT, INVALID_VALUE_PGE);
  if (isV210Target())
    return pge_b32_v210((ULL)dist.value, 0);
  return pge_b32((ULL)dist.value, 0);
}

//----------------------------------------------------------------------------//
//  pset
//----------------------------------------------------------------------------//
template <class T> CCE_INTRINSIC vector_bool pset_b8(T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PSET);
  static_assert(PAT_STATIC_ASSERT, INVALID_VALUE_PSET);
  return __builtin_cce_pset_b8((ULL)dist.value);
}

template <class T> CCE_INTRINSIC vector_bool pset_b16(T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PSET);
  static_assert(PAT_STATIC_ASSERT, INVALID_VALUE_PSET);
  return __builtin_cce_pset_b16((ULL)dist.value);
}

template <class T> CCE_INTRINSIC vector_bool pset_b32(T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PSET);
  static_assert(PAT_STATIC_ASSERT, INVALID_VALUE_PSET);
  return __builtin_cce_pset_b32((ULL)dist.value);
}

#if defined(__DAV_C310__)
template <class T> CCE_INTRINSIC vector_bool pset_2xvl_b64(T dist) {
  return pset_b32(dist);
}
#endif

// Dist for VLD/PLD/PST and their variants
enum class Dist {
  DIST_NORM, // vld, pld, pst
  DIST_BRC_B8,
  DIST_BRC_B16,
  DIST_BRC_B32,
  // rpt2_b8 = 4, deprecated
  // rpt2_b16 = 5, deprecated
  DIST_US_B8 = 6,
  DIST_US_B16,
  DIST_DS_B8,
  DIST_DS_B16,
  DIST_BDINTLV,
  DIST_DINTLV_B8,
  DIST_DINTLV_B16,
  DIST_UNPK_B8,
  DIST_UNPK_B16,
  DIST_BLK,
  DIST_E2B_B16,
  DIST_E2B_B32,
  DIST_UNPK_B32,
  DIST_DINTLV_B32,
  DIST_UNPK4_B8,
  DIST_SPLT4CHN_B8,
  DIST_SPLT2CHN_B8,
  DIST_SPLT2CHN_B16,
  DIST_US = 1, // pld
  DIST_DS = 2, // pld
  DIST_PK = 1  // pst
};

//----------------------------------------------------------------------------//
//  pld
//----------------------------------------------------------------------------//
typedef std::integral_constant<Dist, Dist::DIST_NORM> NormType;
typedef std::integral_constant<Dist, Dist::DIST_US> USType;
typedef std::integral_constant<Dist, Dist::DIST_DS> DSType;
#define NORM NormType()
#define US USType()
#define DS DSType()

#define INVALID_VALUE_PLD "The 4th argument of pld is not valid"
#define ERROR_VALUE_PLD "The 4th argument of this pld can only be: NORM, US, DS"

template <class T>
CCE_INTRINSIC void pld(vector_bool &dst, __ubuf__ uint32_t *base,
                       vector_address offset, T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PLD);
  static_assert(std::is_same<T, NormType>::value ||
                    std::is_same<T, USType>::value ||
                    std::is_same<T, DSType>::value,
                ERROR_VALUE_PLD);
  dst = __builtin_cce_pld_b8(base, offset, (unsigned)dist.value, 0 /* #loop */);
  return;
}

// PLDI, Pre update
template <class T>
CCE_INTRINSIC void pldi(vector_bool &dst, __ubuf__ uint32_t *base,
                        int32_t offset, T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PLD);
  static_assert(std::is_same<T, NormType>::value ||
                    std::is_same<T, USType>::value ||
                    std::is_same<T, DSType>::value,
                ERROR_VALUE_PLD);
  dst = __builtin_cce_pldi_b8(base, offset, (unsigned)dist.value,
                              0 /* post update mode */);
  return;
}

// PLDI Post update
template <class T, class T2>
CCE_INTRINSIC void pldi(vector_bool &dst, __ubuf__ uint32_t *&base,
                        int32_t offset, T dist, T2 post) {
  static_assert(std::is_same<T2, PostUpdateType>::value,
                "The last argument can only be 'POST_UPDATE'.");
  static_assert(std::is_class<T>::value, INVALID_VALUE_PLD);
  static_assert(std::is_same<T, NormType>::value ||
                    std::is_same<T, USType>::value ||
                    std::is_same<T, DSType>::value,
                ERROR_VALUE_PLD);
  struct {
    vector_bool dstData;
    __ubuf__ uint32_t *baseData;
  } ret;
  __builtin_cce_pldi_post_b8(&ret, base, offset, (unsigned)dist.value,
                             1 /* post update mode */);
  dst = ret.dstData;
  base = ret.baseData;
  return;
}

// PLDS, Pre update
template <class T>
CCE_INTRINSIC void plds(vector_bool &dst, __ubuf__ uint32_t *base,
                        int32_t offset, T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PLD);
  static_assert(std::is_same<T, NormType>::value ||
                    std::is_same<T, USType>::value ||
                    std::is_same<T, DSType>::value,
                ERROR_VALUE_PLD);
  dst = __builtin_cce_plds_b8(base, offset, (unsigned)dist.value,
                              0 /* post update mode */);
  return;
}

// PLDS Post update
template <class T, class T2>
CCE_INTRINSIC void plds(vector_bool &dst, __ubuf__ uint32_t *&base,
                        int32_t offset, T dist, T2 post) {
  static_assert(std::is_same<T2, PostUpdateType>::value,
                "The last argument can only be 'POST_UPDATE'.");
  static_assert(std::is_class<T>::value, INVALID_VALUE_PLD);
  static_assert(std::is_same<T, NormType>::value ||
                    std::is_same<T, USType>::value ||
                    std::is_same<T, DSType>::value,
                ERROR_VALUE_PLD);
  struct {
    vector_bool dstData;
    __ubuf__ uint32_t *baseData;
  } ret;
  __builtin_cce_plds_post_b8(&ret, base, offset, (unsigned)dist.value,
                             1 /* post update mode */);
  dst = ret.dstData;
  base = ret.baseData;
  return;
}

//----------------------------------------------------------------------------//
//  pst
//----------------------------------------------------------------------------//
typedef std::integral_constant<Dist, Dist::DIST_PK> PKType;
#define PK PKType()

#define INVALID_VALUE_PST "The 4th argument of pst is not valid"

template <class T>
CCE_INTRINSIC void pst(vector_bool src, __ubuf__ uint32_t *base,
                       vector_address offset, T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PST);
  static_assert(dist.value == NORM.value || dist.value == PK.value,
                INVALID_VALUE_PST);
  return __builtin_cce_pst_b8(src, base, offset, (unsigned)dist.value, 0 /* #loop */);
}

//----------------------------------------------------------------------------//
//  psti
//----------------------------------------------------------------------------//
template <class T>
CCE_INTRINSIC void psti(vector_bool src, __ubuf__ uint32_t *base,
                        int32_t offset, T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PST);
  static_assert(dist.value == NORM.value || dist.value == PK.value,
                INVALID_VALUE_PST);
  return __builtin_cce_psti_b8(src, base, offset, (ULL)dist.value, 0 /* post */);
}

template <class T, class T2>
CCE_INTRINSIC void psti(vector_bool src, __ubuf__ uint32_t *&base,
                        int32_t offset, T dist, T2 post) {
  static_assert(std::is_same<T2, PostUpdateType>::value,
                "The last argument can only be 'POST_UPDATE'.");
  static_assert(std::is_class<T>::value, INVALID_VALUE_PST);
  static_assert(dist.value == NORM.value || dist.value == PK.value,
                INVALID_VALUE_PST);
  base = (__ubuf__ uint32_t *)__builtin_cce_psti_post_b8(
      src, base, offset, (unsigned)dist.value, 1 /* post */);
  return;
}

//----------------------------------------------------------------------------//
//  psts
//----------------------------------------------------------------------------//
template <class T>
CCE_INTRINSIC void psts(vector_bool src, __ubuf__ uint32_t *base,
                        int32_t offset, T dist) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_PST);
  static_assert(dist.value == NORM.value || dist.value == PK.value,
                INVALID_VALUE_PST);
  return __builtin_cce_psts_b8(src, base, offset, (unsigned)dist.value, 0 /* post */);
}

template <class T, class T2>
CCE_INTRINSIC void psts(vector_bool src, __ubuf__ uint32_t *&base,
                        int32_t offset, T dist, T2 post) {
  static_assert(std::is_same<T2, PostUpdateType>::value,
                "The last argument can only be 'POST_UPDATE'.");
  static_assert(std::is_class<T>::value, INVALID_VALUE_PST);
  static_assert(dist.value == NORM.value || dist.value == PK.value,
                INVALID_VALUE_PST);
  base = (__ubuf__ uint32_t *)__builtin_cce_psts_post_b8(
      src, base, offset, (unsigned)dist.value, 1 /* post */);
  return;
}

//----------------------------------------------------------------------------//
//  vld
//----------------------------------------------------------------------------//
typedef std::integral_constant<Dist, Dist::DIST_BRC_B8> BRC_B8_Type;
typedef std::integral_constant<Dist, Dist::DIST_BRC_B16> BRC_B16_Type;
typedef std::integral_constant<Dist, Dist::DIST_BRC_B32> BRC_B32_Type;
typedef std::integral_constant<Dist, Dist::DIST_US_B8> US_B8_Type;
typedef std::integral_constant<Dist, Dist::DIST_US_B16> US_B16_Type;
typedef std::integral_constant<Dist, Dist::DIST_DS_B8> DS_B8_Type;
typedef std::integral_constant<Dist, Dist::DIST_DS_B16> DS_B16_Type;
typedef std::integral_constant<Dist, Dist::DIST_BDINTLV> BDINTLV_Type;
typedef std::integral_constant<Dist, Dist::DIST_DINTLV_B8> DINTLV_B8_Type;
typedef std::integral_constant<Dist, Dist::DIST_DINTLV_B16> DINTLV_B16_Type;
typedef std::integral_constant<Dist, Dist::DIST_UNPK_B8> UNPK_B8_Type;
typedef std::integral_constant<Dist, Dist::DIST_UNPK_B16> UNPK_B16_Type;
typedef std::integral_constant<Dist, Dist::DIST_BLK> BLK_Type;
typedef std::integral_constant<Dist, Dist::DIST_E2B_B16> E2B_B16_Type;
typedef std::integral_constant<Dist, Dist::DIST_E2B_B32> E2B_B32_Type;
typedef std::integral_constant<Dist, Dist::DIST_UNPK_B32> UNPK_B32_Type;
typedef std::integral_constant<Dist, Dist::DIST_DINTLV_B32> DINTLV_B32_Type;
typedef std::integral_constant<Dist, Dist::DIST_UNPK4_B8> UNPK4_B8_Type;
typedef std::integral_constant<Dist, Dist::DIST_SPLT4CHN_B8> SPLT4CHN_B8_Type;
typedef std::integral_constant<Dist, Dist::DIST_SPLT2CHN_B8> SPLT2CHN_B8_Type;
typedef std::integral_constant<Dist, Dist::DIST_SPLT2CHN_B16> SPLT2CHN_B16_Type;
#define BRC_B8 BRC_B8_Type()
#define BRC_B16 BRC_B16_Type()
#define BRC_B32 BRC_B32_Type()
#define US_B8 US_B8_Type()
#define US_B16 US_B16_Type()
#define DS_B8 DS_B8_Type()
#define DS_B16 DS_B16_Type()
#define BDINTLV BDINTLV_Type()
#define DINTLV_B8 DINTLV_B8_Type()
#define DINTLV_B16 DINTLV_B16_Type()
#define UNPK_B8 UNPK_B8_Type()
#define UNPK_B16 UNPK_B16_Type()
#define BLK BLK_Type()
#define E2B_B16 E2B_B16_Type()
#define E2B_B32 E2B_B32_Type()
#define UNPK_B32 UNPK_B32_Type()
#define DINTLV_B32 DINTLV_B32_Type()
#define UNPK4_B8 UNPK4_B8_Type()
#define SPLT4CHN_B8 SPLT4CHN_B8_Type()
#define SPLT2CHN_B8 SPLT2CHN_B8_Type()
#define SPLT2CHN_B16 SPLT2CHN_B16_Type()

#define LDX1_STATIC_ASSERT_V210                                                \
  (std::is_same<T, NormType>::value || std::is_same<T, BRC_B8_Type>::value ||  \
   std::is_same<T, BRC_B16_Type>::value ||                                     \
   std::is_same<T, BRC_B32_Type>::value ||                                     \
   std::is_same<T, US_B8_Type>::value ||                                       \
   std::is_same<T, US_B16_Type>::value ||                                      \
   std::is_same<T, DS_B8_Type>::value ||                                       \
   std::is_same<T, DS_B16_Type>::value ||                                      \
   std::is_same<T, UNPK_B8_Type>::value ||                                     \
   std::is_same<T, UNPK_B16_Type>::value ||                                    \
   std::is_same<T, BLK_Type>::value || std::is_same<T, E2B_B16_Type>::value)

#define LDX1_STATIC_ASSERT_V300                                                \
  (std::is_same<T, E2B_B32_Type>::value ||                                     \
   std::is_same<T, UNPK_B32_Type>::value)

#define LDX1_STATIC_ASSERT_V310                                                \
  (std::is_same<T, UNPK_B32_Type>::value ||                                    \
   std::is_same<T, UNPK4_B8_Type>::value ||                                    \
   std::is_same<T, SPLT4CHN_B8_Type>::value ||                                 \
   std::is_same<T, SPLT2CHN_B8_Type>::value ||                                 \
   std::is_same<T, SPLT2CHN_B16_Type>::value)

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
#define LDX1_STATIC_ASSERT (LDX1_STATIC_ASSERT_V210 || LDX1_STATIC_ASSERT_V300)
#elif defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define LDX1_STATIC_ASSERT                                                     \
  (LDX1_STATIC_ASSERT_V210 || LDX1_STATIC_ASSERT_V300 ||                       \
   LDX1_STATIC_ASSERT_V310)
#else
#define LDX1_STATIC_ASSERT LDX1_STATIC_ASSERT_V210
#endif

#define INVALID_VALUE_VLD "The last argument of vld is not valid"

#if defined(__DAV_L210__)
#define VLD(LT, ST, NUM)                                                       \
  template <class T>                                                           \
  CCE_INTRINSIC void vld(vector_##ST &dst, __ubuf__ LT *base,                  \
                         vector_address offset, T dist) {                      \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VLD);                 \
    static_assert(LDX1_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    dst = __builtin_cce_vldx1_##NUM##ST(base, offset, (ULL)dist.value,         \
                                        0 /* #loop */);                        \
    return;                                                                    \
  }                                                                            \
                                                                               \
  template <class T>                                                           \
  CCE_INTRINSIC void vld(vector_##ST &dst, __ubuf__ LT *base, T dist) {        \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VLD);                 \
    static_assert(LDX1_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    dst = __builtin_cce_vldoncex1_##NUM##ST(base, (ULL)dist.value,             \
                                            4 /* #loop */);                    \
    return;                                                                    \
  }

VLD(int64_t, s64, v32)
VLD(uint64_t, u64, v32)
VLD(int8_t, s8, v256)
VLD(uint8_t, u8, v256)
VLD(int16_t, s16, v128)
VLD(uint16_t, u16, v128)
VLD(int32_t, s32, v64)
VLD(uint32_t, u32, v64)
VLD(half, f16, v128)
VLD(float, f32, v64)
#undef VLD
#elif defined(__DAV_M210_VEC__)
#define VLD(LT, ST, NUM)                                                       \
  template <class T>                                                           \
  CCE_INTRINSIC void vld(vector_##ST &dst, __ubuf__ LT *base,                  \
                         vector_address offset, T dist) {                      \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VLD);                 \
    static_assert(std::is_same<T, NormType>::value ||                          \
                      std::is_same<T, BRC_B8_Type>::value ||                   \
                      std::is_same<T, BRC_B16_Type>::value ||                  \
                      std::is_same<T, BRC_B32_Type>::value ||                  \
                      std::is_same<T, US_B8_Type>::value ||                    \
                      std::is_same<T, US_B16_Type>::value ||                   \
                      std::is_same<T, DS_B8_Type>::value ||                    \
                      std::is_same<T, DS_B16_Type>::value ||                   \
                      std::is_same<T, UNPK_B8_Type>::value ||                  \
                      std::is_same<T, UNPK_B16_Type>::value,                   \
                  INVALID_VALUE_VLD);                                          \
    dst = __builtin_cce_vldx1_##NUM##ST(base, offset, (ULL)dist.value,         \
                                        0 /* #loop */);                        \
    return;                                                                    \
  }                                                                            \
                                                                               \
  template <class T>                                                           \
  CCE_INTRINSIC void vld(vector_##ST &dst, __ubuf__ LT *base, T dist) {        \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VLD);                 \
    static_assert(std::is_same<T, NormType>::value ||                          \
                      std::is_same<T, BRC_B8_Type>::value ||                   \
                      std::is_same<T, BRC_B16_Type>::value ||                  \
                      std::is_same<T, BRC_B32_Type>::value ||                  \
                      std::is_same<T, US_B8_Type>::value ||                    \
                      std::is_same<T, US_B16_Type>::value ||                   \
                      std::is_same<T, DS_B8_Type>::value ||                    \
                      std::is_same<T, DS_B16_Type>::value ||                   \
                      std::is_same<T, UNPK_B8_Type>::value ||                  \
                      std::is_same<T, UNPK_B16_Type>::value,                   \
                  INVALID_VALUE_VLD);                                          \
    dst = __builtin_cce_vldoncex1_##NUM##ST(base, (ULL)dist.value,             \
                                            4 /* #loop */);                    \
    return;                                                                    \
  }

VLD(int64_t, s64, v32)
VLD(uint64_t, u64, v32)
VLD(int8_t, s8, v256)
VLD(uint8_t, u8, v256)
VLD(int16_t, s16, v128)
VLD(uint16_t, u16, v128)
VLD(int32_t, s32, v64)
VLD(uint32_t, u32, v64)
VLD(half, f16, v128)
VLD(float, f32, v64)
#undef VLD
#elif defined(__DAV_M300__) || defined(__DAV_L300__) ||                        \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VLD(LT, ST, NUM)                                                       \
  template <class T>                                                           \
  CCE_INTRINSIC void vld(vector_##ST &dst, __ubuf__ LT *base,                  \
                         vector_address offset, T dist) {                      \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VLD);                 \
    static_assert(LDX1_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    dst = __builtin_cce_vldx1_##NUM##ST(base, offset, (ULL)dist.value,         \
                                        0 /* #loop */);                        \
    return;                                                                    \
  }                                                                            \
  template <class T>                                                           \
  CCE_INTRINSIC void vld(vector_##ST &dst, __ubuf__ LT *base, T dist) {        \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VLD);                 \
    static_assert(LDX1_STATIC_ASSERT_V210, INVALID_VALUE_VLD);                 \
    vlds(dst, base, 0, dist);                                                  \
    return;                                                                    \
  }

// the vld without offset is use for compatible v300 with v210
VLD(int8_t, s8, v256)
VLD(uint8_t, u8, v256)
VLD(int16_t, s16, v128)
VLD(uint16_t, u16, v128)
VLD(int32_t, s32, v64)
VLD(uint64_t, u64, v32)
VLD(uint32_t, u32, v64)
VLD(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VLD(bfloat16_t, bf16, v128)
#endif
VLD(half, f16, v128)
VLD(float, f32, v64)
#undef VLD
#endif

#define LDX2_STATIC_ASSERT_V210                                                \
  (std::is_same<T, BDINTLV_Type>::value ||                                     \
   std::is_same<T, DINTLV_B8_Type>::value ||                                   \
   std::is_same<T, DINTLV_B16_Type>::value)

#define LDX2_STATIC_ASSERT_V310 (std::is_same<T, DINTLV_B32_Type>::value)

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) ||                      \
    defined(__DAV_T210__) || defined(__DAV_M300__) || defined(__DAV_L300__) || \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
#define LDX2_STATIC_ASSERT LDX2_STATIC_ASSERT_V210
#elif defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define LDX2_STATIC_ASSERT (LDX2_STATIC_ASSERT_V210 || LDX2_STATIC_ASSERT_V310)
#endif

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VLDX2(LT, ST, NUM)                                                     \
  template <class T>                                                           \
  CCE_INTRINSIC void vld(vector_##ST &dst0, vector_##ST &dst1,                 \
                         __ubuf__ LT *base, vector_address offset, T dist) {   \
    static_assert(LDX2_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    vector_##ST##x2_t __ret;                                                   \
    __builtin_cce_vldx2_##NUM##ST(&__ret, base, offset, (ULL)dist.value,       \
                                  0 /* #loop */);                              \
    dst0 = __ret.val[0];                                                       \
    dst1 = __ret.val[1];                                                       \
    return;                                                                    \
  }                                                                            \
  template <class T>                                                           \
  CCE_INTRINSIC void vld(vector_##ST &dst0, vector_##ST &dst1,                 \
                         __ubuf__ LT *base, T dist) {                          \
    static_assert(LDX2_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    vlds(dst0, dst1, base, 0, dist);                                           \
    return;                                                                    \
  }

// the vld without offset is use for compatible v300 with v210
VLDX2(int8_t, s8, v256)
VLDX2(uint8_t, u8, v256)
VLDX2(int16_t, s16, v128)
VLDX2(uint16_t, u16, v128)
VLDX2(int32_t, s32, v64)
VLDX2(uint64_t, u64, v32)
VLDX2(uint32_t, u32, v64)
VLDX2(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VLDX2(bfloat16_t, bf16, v128)
#endif
VLDX2(half, f16, v128)
VLDX2(float, f32, v64)
#undef VLDX2
#endif

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
#define VLDX2(LT, ST, NUM)                                                     \
  template <class T>                                                           \
  CCE_INTRINSIC void vld(vector_##ST &dst0, vector_##ST &dst1,                 \
                         __ubuf__ LT *base, vector_address offset, T dist) {   \
    static_assert(LDX2_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    vector_##ST##x2_t __ret;                                                   \
    __builtin_cce_vldx2_##NUM##ST(&__ret, base, offset, (ULL)dist.value,       \
                                  0 /* #loop */);                              \
    dst0 = __ret.val[0];                                                       \
    dst1 = __ret.val[1];                                                       \
    return;                                                                    \
  }                                                                            \
                                                                               \
  template <class T>                                                           \
  CCE_INTRINSIC void vld(vector_##ST &dst0, vector_##ST &dst1,                 \
                         __ubuf__ LT *base, T dist) {                          \
    static_assert(LDX2_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    vector_##ST##x2_t __ret;                                                   \
    __builtin_cce_vldoncex2_##NUM##ST(&__ret, base, (ULL)dist.value,           \
                                      4 /* #loop */);                          \
    dst0 = __ret.val[0];                                                       \
    dst1 = __ret.val[1];                                                       \
    return;                                                                    \
  }

VLDX2(int8_t, s8, v256)
VLDX2(uint8_t, u8, v256)
VLDX2(int16_t, s16, v128)
VLDX2(uint16_t, u16, v128)
VLDX2(int32_t, s32, v64)
VLDX2(uint32_t, u32, v64)
VLDX2(half, f16, v128)
VLDX2(float, f32, v64)
#undef VLDX2
#endif

//----------------------------------------------------------------------------//
//  vst
//----------------------------------------------------------------------------//
enum class DistVST {
  DIST_NORM_B8,
  DIST_NORM_B16,
  DIST_NORM_B32,
  DIST_ONEPT_B8,
  DIST_ONEPT_B16,
  DIST_ONEPT_B32,
  DIST_PK_B16,
  DIST_PK_B32,
  DIST_INTLV_B8,
  DIST_INTLV_B16,
  DIST_PK_B64,
  DIST_INTLV_B32,
  DIST_PK4_B32,
  DIST_MRG4CHN_B8,
  DIST_MRG2CHN_B8,
  DIST_MRG2CHN_B16,
};
typedef std::integral_constant<DistVST, DistVST::DIST_NORM_B8> NORM_B8_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_NORM_B16> NORM_B16_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_NORM_B32> NORM_B32_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_ONEPT_B8> ONEPT_B8_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_ONEPT_B16> ONEPT_B16_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_ONEPT_B32> ONEPT_B32_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_PK_B16> PK_B16_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_PK_B32> PK_B32_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_INTLV_B8> INTLV_B8_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_INTLV_B16> INTLV_B16_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_PK_B64> PK_B64_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_INTLV_B32> INTLV_B32_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_PK4_B32> PK4_B32_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_MRG4CHN_B8>
    MRG4CHN_B8_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_MRG2CHN_B8>
    MRG2CHN_B8_Type;
typedef std::integral_constant<DistVST, DistVST::DIST_MRG2CHN_B16>
    MRG2CHN_B16_Type;
#define NORM_B8 NORM_B8_Type()
#define NORM_B16 NORM_B16_Type()
#define NORM_B32 NORM_B32_Type()
#define ONEPT_B8 ONEPT_B8_Type()
#define ONEPT_B16 ONEPT_B16_Type()
#define ONEPT_B32 ONEPT_B32_Type()
#define PK_B16 PK_B16_Type()
#define PK_B32 PK_B32_Type()
#define INTLV_B8 INTLV_B8_Type()
#define INTLV_B16 INTLV_B16_Type()
#define PK_B64 PK_B64_Type()
#define INTLV_B32 INTLV_B32_Type()
#define PK4_B32 PK4_B32_Type()
#define MRG4CHN_B8 MRG4CHN_B8_Type()
#define MRG2CHN_B8 MRG2CHN_B8_Type()
#define MRG2CHN_B16 MRG2CHN_B16_Type()

#define STX1_STATIC_ASSERT_V210                                                \
  (std::is_same<T, NORM_B8_Type>::value ||                                     \
   std::is_same<T, NORM_B16_Type>::value ||                                    \
   std::is_same<T, NORM_B32_Type>::value ||                                    \
   std::is_same<T, ONEPT_B8_Type>::value ||                                    \
   std::is_same<T, ONEPT_B16_Type>::value ||                                   \
   std::is_same<T, ONEPT_B32_Type>::value ||                                   \
   std::is_same<T, PK_B16_Type>::value ||                                      \
   std::is_same<T, PK_B32_Type>::value || std::is_same<T, PK_B64_Type>::value)

#define STX1_STATIC_ASSERT_V310                                                \
  (std::is_same<T, PK4_B32_Type>::value ||                                     \
   std::is_same<T, MRG4CHN_B8_Type>::value ||                                  \
   std::is_same<T, MRG2CHN_B8_Type>::value ||                                  \
   std::is_same<T, MRG2CHN_B16_Type>::value)

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) ||                      \
    defined(__DAV_T210__) || defined(__DAV_M300__) || defined(__DAV_L300__) || \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
#define STX1_STATIC_ASSERT STX1_STATIC_ASSERT_V210
#elif defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define STX1_STATIC_ASSERT (STX1_STATIC_ASSERT_V210 || STX1_STATIC_ASSERT_V310)
#endif

#define INVALID_VALUE_VST "The 4th argument of vst is not valid"

#define VST(LT, ST, NUM)                                                       \
  template <class T>                                                           \
  CCE_INTRINSIC void vst(vector_##ST data, __ubuf__ LT *base,                  \
                         vector_address offset, T dist, vector_bool mask) {    \
    static_assert(STX1_STATIC_ASSERT, INVALID_VALUE_VST);                      \
    return __builtin_cce_vstx1_##NUM##ST(data, base, offset, (ULL)dist.value,  \
                                         0 /* #loop */, mask);                 \
  }

VST(int8_t, s8, v256)
VST(uint8_t, u8, v256)
VST(int16_t, s16, v128)
VST(uint16_t, u16, v128)
VST(int32_t, s32, v64)
VST(uint32_t, u32, v64)
VST(half, f16, v128)
VST(float, f32, v64)
VST(uint64_t, u64, v32)
VST(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VST(bfloat16_t, bf16, v128)
#endif
#undef VST

//----------------------------------------------------------------------------//
//  vstx2
//  data type can only be b8/b16, predicate register is neglected here.
//----------------------------------------------------------------------------//
#define STX2_STATIC_ASSERT_V210                                                \
  (std::is_same<T, INTLV_B8_Type>::value ||                                    \
   std::is_same<T, INTLV_B16_Type>::value)

#define STX2_STATIC_ASSERT_V310 (std::is_same<T, INTLV_B32_Type>::value)

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) ||                      \
    defined(__DAV_T210__) || defined(__DAV_M300__) || defined(__DAV_L300__) || \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
#define STX2_STATIC_ASSERT STX2_STATIC_ASSERT_V210
#elif defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define STX2_STATIC_ASSERT (STX2_STATIC_ASSERT_V210 || STX2_STATIC_ASSERT_V310)
#endif

#define INVALID_VALUE_VSTX2 "The 5th argument of vst is not valid"

#define VSTX2(LT, ST, NUM)                                                     \
  template <class T>                                                           \
  CCE_INTRINSIC void vst(vector_##ST src0, vector_##ST src1,                   \
                         __ubuf__ LT *base, vector_address offset, T dist,     \
                         vector_bool mask) {                                   \
    static_assert(STX2_STATIC_ASSERT, INVALID_VALUE_VSTX2);                    \
    __builtin_cce_vstx2_##NUM##ST(src0, src1, base, offset, (ULL)dist.value,   \
                                  0 /* #loop */, mask);                        \
    return;                                                                    \
  }

VSTX2(int8_t, s8, v256)
VSTX2(uint8_t, u8, v256)
VSTX2(int16_t, s16, v128)
VSTX2(uint16_t, u16, v128)
#if defined(__DAV_C310__)
VSTX2(int32_t, s32, v64)
VSTX2(uint32_t, u32, v64)
#endif
VSTX2(half, f16, v128)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTX2(bfloat16_t, bf16, v128)
#endif
#undef VSTX2

//----------------------------------------------------------------------------//
//  vlds
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VLDS(LT, ST, NUM)                                                      \
  template <class T>                                                           \
  CCE_INTRINSIC void vlds(vector_##ST &dst, __ubuf__ LT *base,                 \
                          int32_t offset /* in unit of element */, T dist) {   \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VLD);                 \
    static_assert(LDX1_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    dst = __builtin_cce_vldsx1_##NUM##ST(                                      \
        base, offset * sizeof(LT), (ULL)dist.value, 0 /* post update mode */); \
    return;                                                                    \
  }

VLDS(int8_t, s8, v256)
VLDS(uint8_t, u8, v256)
VLDS(int16_t, s16, v128)
VLDS(uint16_t, u16, v128)
VLDS(uint64_t, u64, v32)
VLDS(int32_t, s32, v64)
VLDS(uint32_t, u32, v64)
VLDS(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VLDS(bfloat16_t, bf16, v128)
#endif
VLDS(half, f16, v128)
VLDS(float, f32, v64)
#undef VLDS

#define VLDSX2(LT, ST, NUM)                                                    \
  template <class T>                                                           \
  CCE_INTRINSIC void vlds(vector_##ST &dst0, vector_##ST &dst1,                \
                          __ubuf__ LT *base,                                   \
                          int32_t offset /* in unit of element */, T dist) {   \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VLD);                 \
    static_assert(LDX2_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    vector_##ST##x2_t __ret;                                                   \
    __builtin_cce_vldsx2_##NUM##ST(&__ret, base, offset * sizeof(LT),          \
                                   (ULL)dist.value, 0 /* post update mode */); \
    dst0 = __ret.val[0];                                                       \
    dst1 = __ret.val[1];                                                       \
    return;                                                                    \
  }

VLDSX2(int8_t, s8, v256)
VLDSX2(uint8_t, u8, v256)
VLDSX2(int16_t, s16, v128)
VLDSX2(uint16_t, u16, v128)
VLDSX2(uint64_t, u64, v32)
VLDSX2(int64_t, s64, v32)
VLDSX2(int32_t, s32, v64)
VLDSX2(uint32_t, u32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VLDSX2(bfloat16_t, bf16, v128)
#endif
VLDSX2(half, f16, v128)
VLDSX2(float, f32, v64)
#undef VLDSX2

#define VLDS_POST(LT, ST, NUM)                                                 \
  template <class T, class T2>                                                 \
  CCE_INTRINSIC void vlds(vector_##ST &dst, __ubuf__ LT *&base,                \
                          int32_t offset /* in unit of element */, T dist,     \
                          T2 post) {                                           \
    static_assert(std::is_same<T2, PostUpdateType>::value,                     \
                  "The last argument can only be 'POST_UPDATE'.");             \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VLD);                 \
    static_assert(LDX1_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    struct {                                                                   \
      vector_##ST vecData;                                                     \
      __ubuf__ LT *baseData;                                                   \
    } ret;                                                                     \
    __builtin_cce_vldsx1_post_##NUM##ST(&ret, base, offset * sizeof(LT),       \
                                        (ULL)dist.value,                       \
                                        1 /* post update mode */);             \
    dst = ret.vecData;                                                         \
    base = ret.baseData;                                                       \
    return;                                                                    \
  }

VLDS_POST(int8_t, s8, v256)
VLDS_POST(uint8_t, u8, v256)
VLDS_POST(int16_t, s16, v128)
VLDS_POST(uint16_t, u16, v128)
VLDS_POST(uint64_t, u64, v32)
VLDS_POST(int64_t, s64, v32)
VLDS_POST(int32_t, s32, v64)
VLDS_POST(uint32_t, u32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VLDS_POST(bfloat16_t, bf16, v128)
#endif
VLDS_POST(half, f16, v128)
VLDS_POST(float, f32, v64)
#undef VLDS_POST

#define VLDSX2_POST(LT, ST, NUM)                                               \
  template <class T, class T2>                                                 \
  CCE_INTRINSIC void vlds(                                                     \
      vector_##ST &dst0, vector_##ST &dst1, __ubuf__ LT *&base,                \
      int32_t offset /* in unit of element */, T dist, T2 post) {              \
    static_assert(std::is_same<T2, PostUpdateType>::value,                     \
                  "The last argument can only be 'POST_UPDATE'.");             \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VLD);                 \
    static_assert(LDX2_STATIC_ASSERT, INVALID_VALUE_VLD);                      \
    struct {                                                                   \
      vector_##ST##x2_t vecData;                                               \
      __ubuf__ LT *baseData;                                                   \
    } __ret;                                                                   \
    __builtin_cce_vldsx2_post_##NUM##ST(&__ret, base, offset * sizeof(LT),     \
                                        (ULL)dist.value,                       \
                                        1 /* post update mode */);             \
    dst0 = __ret.vecData.val[0];                                               \
    dst1 = __ret.vecData.val[1];                                               \
    base = __ret.baseData;                                                     \
    return;                                                                    \
  }

VLDSX2_POST(int8_t, s8, v256)
VLDSX2_POST(uint8_t, u8, v256)
VLDSX2_POST(int16_t, s16, v128)
VLDSX2_POST(uint16_t, u16, v128)
VLDSX2_POST(uint64_t, u64, v32)
VLDSX2_POST(int32_t, s32, v64)
VLDSX2_POST(uint32_t, u32, v64)
VLDSX2_POST(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VLDSX2_POST(bfloat16_t, bf16, v128)
#endif
VLDSX2_POST(half, f16, v128)
VLDSX2_POST(float, f32, v64)
#undef VLDSX2_POST
#endif // defined(__DAV_M300__) || defined(__DAV_L300__) ||
       // defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||
       // defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)

//----------------------------------------------------------------------------//
//  vsldb
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VSLDB(LT, ST, NUM)                                                     \
  CCE_INTRINSIC void vsldb(vector_##ST &dst, __ubuf__ LT *base,                \
                           int32_t offset, vector_bool mask) {                 \
    dst = __builtin_cce_vsldb_##NUM##ST(base, offset,                          \
                                        0 /* post update mode */, mask);       \
    return;                                                                    \
  }

VSLDB(int8_t, s8, v256)
VSLDB(uint8_t, u8, v256)
VSLDB(int16_t, s16, v128)
VSLDB(uint16_t, u16, v128)
VSLDB(int32_t, s32, v64)
VSLDB(uint32_t, u32, v64)
VSLDB(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSLDB(bfloat16_t, bf16, v128)
#endif
VSLDB(half, f16, v128)
VSLDB(float, f32, v64)
#undef VSLDB

#define VSLDB_POST(LT, ST, NUM)                                                \
  template <class T2>                                                          \
  CCE_INTRINSIC void vsldb(vector_##ST &dst, __ubuf__ LT *&base,               \
                           int32_t offset, vector_bool mask, T2 post) {        \
    static_assert(std::is_same<T2, PostUpdateType>::value,                     \
                  "The last argument can only be 'POST_UPDATE'.");             \
    struct {                                                                   \
      vector_##ST vecData;                                                     \
      __ubuf__ LT *baseData;                                                   \
    } ret;                                                                     \
    __builtin_cce_vsldb_post_##NUM##ST(&ret, base, offset,                     \
                                       1 /* post update mode */, mask);        \
    dst = ret.vecData;                                                         \
    base = ret.baseData;                                                       \
    return;                                                                    \
  }

VSLDB_POST(int8_t, s8, v256)
VSLDB_POST(uint8_t, u8, v256)
VSLDB_POST(int16_t, s16, v128)
VSLDB_POST(uint16_t, u16, v128)
VSLDB_POST(int32_t, s32, v64)
VSLDB_POST(uint32_t, u32, v64)
VSLDB_POST(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSLDB_POST(bfloat16_t, bf16, v128)
#endif
VSLDB_POST(half, f16, v128)
VSLDB_POST(float, f32, v64)
#undef VSLDB_POST
#endif

//----------------------------------------------------------------------------//
//  vldi
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_M310__) || defined(__DAV_L310__)
#define INVALID_VALUE_VLDI "The 4th argument of vldi is not valid"

#define VLDI(LT, ST, NUM)                                                      \
  template <class T>                                                           \
  DEPRECATED CCE_INTRINSIC void vldi(vector_##ST &dst, __ubuf__ LT *base,      \
                                     int32_t offset /* in unit of element */,  \
                                     T dist) {                                 \
    static_assert(LDX1_STATIC_ASSERT, INVALID_VALUE_VLDI);                     \
    dst = __builtin_cce_vldix1_##NUM##ST(                                      \
        base, offset * sizeof(LT), (ULL)dist.value, 0 /* post update mode */); \
    return;                                                                    \
  }
VLDI(int8_t, s8, v256)
VLDI(uint8_t, u8, v256)
VLDI(int16_t, s16, v128)
VLDI(uint16_t, u16, v128)
VLDI(int32_t, s32, v64)
VLDI(uint32_t, u32, v64)
VLDI(uint64_t, u64, v32)
VLDI(half, f16, v128)
VLDI(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
VLDI(bfloat16_t, bf16, v128)
#endif
VLDI(float, f32, v64)
#undef VLDI

#define INVALID_VALUE_VLDIX2 "The 5th argument of vldi is not valid"

#define VLDIX2(LT, ST, NUM)                                                    \
  template <class T>                                                           \
  DEPRECATED CCE_INTRINSIC void vldi(                                          \
      vector_##ST &dst0, vector_##ST &dst1, __ubuf__ LT *base,                 \
      int32_t offset /* in unit of element */, T dist) {                       \
    static_assert(LDX2_STATIC_ASSERT, INVALID_VALUE_VLDIX2);                   \
    vector_##ST##x2_t __ret;                                                   \
    __builtin_cce_vldix2_##NUM##ST(&__ret, base, offset * sizeof(LT),          \
                                   (ULL)dist.value, 0 /* post update mode */); \
    dst0 = __ret.val[0];                                                       \
    dst1 = __ret.val[1];                                                       \
    return;                                                                    \
  }

VLDIX2(int8_t, s8, v256)
VLDIX2(uint8_t, u8, v256)
VLDIX2(int16_t, s16, v128)
VLDIX2(uint16_t, u16, v128)
VLDIX2(int32_t, s32, v64)
VLDIX2(uint32_t, u32, v64)
VLDIX2(uint64_t, u64, v32)
VLDIX2(half, f16, v128)
VLDIX2(float, f32, v64)
VLDIX2(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
VLDIX2(bfloat16_t, bf16, v128)
#endif
#undef VLDIX2

#define INVALID_VALUE_VLDI_POST "The 5th argument of vldi is not valid"

#define VLDI_POST(LT, ST, NUM)                                                 \
  template <class T, class T2>                                                 \
  DEPRECATED CCE_INTRINSIC void vldi(vector_##ST &dst, __ubuf__ LT *&base,     \
                                     int32_t offset /* in unit of element */,  \
                                     T dist, T2 post) {                        \
    static_assert(std::is_same<T2, PostUpdateType>::value,                     \
                  "The last argument can only be 'POST_UPDATE'.");             \
    static_assert(LDX1_STATIC_ASSERT, INVALID_VALUE_VLDI_POST);                \
    struct {                                                                   \
      vector_##ST vecData;                                                     \
      __ubuf__ LT *baseData;                                                   \
    } ret;                                                                     \
    __builtin_cce_vldix1_post_##NUM##ST(&ret, base, offset * sizeof(LT),       \
                                        (ULL)dist.value,                       \
                                        1 /* post update mode */);             \
    dst = ret.vecData;                                                         \
    base = ret.baseData;                                                       \
    return;                                                                    \
  }

VLDI_POST(int8_t, s8, v256)
VLDI_POST(uint8_t, u8, v256)
VLDI_POST(int16_t, s16, v128)
VLDI_POST(uint16_t, u16, v128)
VLDI_POST(int32_t, s32, v64)
VLDI_POST(uint32_t, u32, v64)
VLDI_POST(half, f16, v128)
VLDI_POST(float, f32, v64)
VLDI_POST(uint64_t, u64, v32)
VLDI_POST(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
VLDI_POST(bfloat16_t, bf16, v128)
#endif
#undef VLDI_POST

#define INVALID_VALUE_VLDIX2_POST "The 5th argument of vsti is not valid"

#define VLDIX2_POST(LT, ST, NUM)                                               \
  template <class T, class T2>                                                 \
  DEPRECATED CCE_INTRINSIC void vldi(                                          \
      vector_##ST &dst0, vector_##ST &dst1, __ubuf__ LT *&base,                \
      int32_t offset /* in unit of element */, T dist, T2 post) {              \
    static_assert(std::is_same<T2, PostUpdateType>::value,                     \
                  "The last argument can only be 'POST_UPDATE'.");             \
    static_assert(LDX2_STATIC_ASSERT, INVALID_VALUE_VLDIX2_POST);              \
    struct {                                                                   \
      vector_##ST##x2_t vecData;                                               \
      __ubuf__ LT *baseData;                                                   \
    } ret;                                                                     \
    __builtin_cce_vldix2_post_##NUM##ST(&ret, base, offset * sizeof(LT),       \
                                        (ULL)dist.value,                       \
                                        1 /* post update mode */);             \
    dst0 = ret.vecData.val[0];                                                 \
    dst1 = ret.vecData.val[1];                                                 \
    base = ret.baseData;                                                       \
    return;                                                                    \
  }

VLDIX2_POST(int8_t, s8, v256)
VLDIX2_POST(uint8_t, u8, v256)
VLDIX2_POST(int16_t, s16, v128)
VLDIX2_POST(uint16_t, u16, v128)
VLDIX2_POST(int32_t, s32, v64)
VLDIX2_POST(uint32_t, u32, v64)
VLDIX2_POST(uint64_t, u64, v32)
VLDIX2_POST(half, f16, v128)
VLDIX2_POST(float, f32, v64)
VLDIX2_POST(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
VLDIX2_POST(bfloat16_t, bf16, v128)
#endif
#undef VLDIX2_POST // define VLDIX2_POST(LT, ST, NUM)
#endif             // defined(__DAV_M300__) || defined(__DAV_L300__) ||
                   // defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||
                   // defined(__DAV_L310__) || defined(__DAV_M310__)

//----------------------------------------------------------------------------//
//  vsts/vsti
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VSTS(OP, LT, ST, NUM, DEPRECATED_ATTR)                                 \
  template <class T>                                                           \
  DEPRECATED_ATTR CCE_INTRINSIC void OP(                                       \
      vector_##ST data, __ubuf__ LT *base,                                     \
      int32_t offset /* in unit of element */, T dist, vector_bool mask) {     \
    static_assert(STX1_STATIC_ASSERT, INVALID_VALUE_VST);                      \
    return __builtin_cce_##OP##x1_##NUM##ST(                                   \
        data, base, offset * sizeof(LT), (ULL)dist.value, 0 /* post */, mask); \
  }                                                                            \
  template <class T, class T2>                                                 \
  DEPRECATED_ATTR CCE_INTRINSIC void OP(                                       \
      vector_##ST data, __ubuf__ LT *&base,                                    \
      int32_t offset /* in unit of element */, T dist, vector_bool mask,       \
      T2 post) {                                                               \
    static_assert(std::is_same<T2, PostUpdateType>::value,                     \
                  "The last argument can only be 'POST_UPDATE'.");             \
    static_assert(STX1_STATIC_ASSERT, INVALID_VALUE_VST);                      \
    base = (__ubuf__ LT *)__builtin_cce_##OP##x1_post_##NUM##ST(               \
        data, base, offset * sizeof(LT), (ULL)dist.value, 1 /* post */, mask); \
    return;                                                                    \
  }

VSTS(vsts, int8_t, s8, v256, NOT_DEPRECATED)
VSTS(vsts, uint8_t, u8, v256, NOT_DEPRECATED)
VSTS(vsts, int16_t, s16, v128, NOT_DEPRECATED)
VSTS(vsts, uint16_t, u16, v128, NOT_DEPRECATED)
VSTS(vsts, int32_t, s32, v64, NOT_DEPRECATED)
VSTS(vsts, uint32_t, u32, v64, NOT_DEPRECATED)
VSTS(vsts, half, f16, v128, NOT_DEPRECATED)
VSTS(vsts, float, f32, v64, NOT_DEPRECATED)
VSTS(vsts, int64_t, s64, v32, NOT_DEPRECATED)
VSTS(vsts, uint64_t, u64, v32, NOT_DEPRECATED)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTS(vsts, bfloat16_t, bf16, v128, NOT_DEPRECATED)
#endif

#ifndef __DAV_C310__
// vsti use same macro with vsts
VSTS(vsti, int8_t, s8, v256, DEPRECATED)
VSTS(vsti, uint8_t, u8, v256, DEPRECATED)
VSTS(vsti, int16_t, s16, v128, DEPRECATED)
VSTS(vsti, uint16_t, u16, v128, DEPRECATED)
VSTS(vsti, int32_t, s32, v64, DEPRECATED)
VSTS(vsti, uint32_t, u32, v64, DEPRECATED)
VSTS(vsti, half, f16, v128, DEPRECATED)
VSTS(vsti, float, f32, v64, DEPRECATED)
VSTS(vsti, int64_t, s64, v32, DEPRECATED)
VSTS(vsti, uint64_t, u64, v32, DEPRECATED)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__)
VSTS(vsti, bfloat16_t, bf16, v128, DEPRECATED)
#endif
#endif // not def __DAV_C310__
#undef VSTS
#endif // defined(__DAV_M300__) || defined(__DAV_L300__) ||
       // defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||
       // defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VSTSX2(OP, LT, ST, NUM)                                                \
  template <class T>                                                           \
  CCE_INTRINSIC void OP(vector_##ST src0, vector_##ST src1, __ubuf__ LT *base, \
                        int32_t offset /* in unit of element */, T dist,       \
                        vector_bool mask) {                                    \
    static_assert(STX2_STATIC_ASSERT, INVALID_VALUE_VST);                      \
    return __builtin_cce_##OP##x2_##NUM(src0, src1, base, offset * sizeof(LT), \
                                        (ULL)dist.value, 0 /* post */, mask);  \
  }

// only addded no post update interface for vsts
// if the offset is a Immediate will lower to VSTI in NEXT MR
// post update interfaces will be added if user submits a request to us
VSTSX2(vsts, int8_t, s8, v256b8)
VSTSX2(vsts, uint8_t, u8, v256b8)
VSTSX2(vsts, int16_t, s16, v128b16)
VSTSX2(vsts, uint16_t, u16, v128b16)
#if defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
VSTSX2(vsts, int32_t, s32, v64b32)
VSTSX2(vsts, uint32_t, u32, v64b32)
#endif
VSTSX2(vsts, half, f16, v128b16)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTSX2(vsts, bfloat16_t, bf16, v128b16)
#endif
#undef VSTSX2
#endif // defined(__DAV_M300__) || defined(__DAV_L300__) ||
       // defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||
       // defined(__DAV_C310__)

//----------------------------------------------------------------------------//
//  vsstb
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VSSTB(LT, ST, NUM)                                                     \
  CCE_INTRINSIC void vsstb(vector_##ST data, __ubuf__ LT *base,                \
                           int32_t offset, vector_bool mask) {                 \
    return __builtin_cce_vsstb_##NUM##ST(data, base, offset, 0 /* post */,     \
                                         mask);                                \
  }

VSSTB(int8_t, s8, v256)
VSSTB(uint8_t, u8, v256)
VSSTB(int16_t, s16, v128)
VSSTB(uint16_t, u16, v128)
VSSTB(int32_t, s32, v64)
VSSTB(uint32_t, u32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSSTB(bfloat16_t, bf16, v128)
#endif
VSSTB(half, f16, v128)
VSSTB(float, f32, v64)
#undef VSSTB

#define VSSTB_POST(LT, ST, NUM)                                                \
  template <class T2>                                                          \
  CCE_INTRINSIC void vsstb(vector_##ST data, __ubuf__ LT *&base,               \
                           int32_t offset, vector_bool mask, T2 post) {        \
    static_assert(std::is_same<T2, PostUpdateType>::value,                     \
                  "The last argument can only be 'POST_UPDATE'.");             \
    base = (__ubuf__ LT *)__builtin_cce_vsstb_post_##NUM##ST(                  \
        data, base, offset, 1 /* post */, mask);                               \
    return;                                                                    \
  }

VSSTB_POST(int8_t, s8, v256)
VSSTB_POST(uint8_t, u8, v256)
VSSTB_POST(int16_t, s16, v128)
VSSTB_POST(uint16_t, u16, v128)
VSSTB_POST(int32_t, s32, v64)
VSSTB_POST(uint32_t, u32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSSTB_POST(bfloat16_t, bf16, v128)
#endif
VSSTB_POST(half, f16, v128)
VSSTB_POST(float, f32, v64)
#undef VSSTB_POST

#endif

//----------------------------------------------------------------------------//
//  vstu
//----------------------------------------------------------------------------//
#define INVALID_VALUE_VSTU "The 5th argument of vstu is not valid"

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)

#define VSTU(LT, ST)                                                           \
  template <class T>                                                           \
  CCE_INTRINSIC void vstu(vector_align &alignData, vector_address &offset,     \
                          vector_##ST src, __ubuf__ LT *base, T post) {        \
    static_assert(std::is_same<T, NoPostUpdateType>::value ||                  \
                      std::is_same<T, PostUpdateType>::value,                  \
                  INVALID_VALUE_VSTU);                                         \
    struct {                                                                   \
      vector_align_data alignData;                                             \
      vector_address offset;                                                   \
    } ret;                                                                     \
    __builtin_cce_vstu_##ST(&ret, src, base, offset, alignData, 0 /* #loop */, \
                            (ULL)post.value);                                  \
    alignData = ret.alignData;                                                 \
    offset = ret.offset;                                                       \
    return;                                                                    \
  }
VSTU(int8_t, s8)
VSTU(uint8_t, u8)
VSTU(int16_t, s16)
VSTU(uint16_t, u16)
VSTU(int32_t, s32)
VSTU(uint32_t, u32)
VSTU(half, f16)
VSTU(float, f32)
#undef VSTU

#elif defined(__DAV_M300__) || defined(__DAV_L300__) ||                        \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VSTU_POST(LT, ST)                                                      \
  template <class T>                                                           \
  CCE_INTRINSIC void vstu(vector_align &alignData, vector_address &offset,     \
                          vector_##ST src, __ubuf__ LT *base, T post) {        \
    static_assert(std::is_same<T, PostUpdateType>::value, INVALID_VALUE_VSTU); \
    struct {                                                                   \
      vector_align_data alignData;                                             \
      vector_address offset;                                                   \
    } ret;                                                                     \
    __builtin_cce_vstu_##ST(&ret, src, base, offset, alignData,                \
                            1 /*post update mode*/, 0 /*loop*/);               \
    alignData = ret.alignData;                                                 \
    offset = ret.offset;                                                       \
    return;                                                                    \
  }
VSTU_POST(int8_t, s8)
VSTU_POST(uint8_t, u8)
VSTU_POST(int16_t, s16)
VSTU_POST(uint16_t, u16)
VSTU_POST(int32_t, s32)
VSTU_POST(uint32_t, u32)
VSTU_POST(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTU_POST(bfloat16_t, bf16)
#endif
VSTU_POST(half, f16)
VSTU_POST(float, f32)
#undef VSTU_POST

#endif

//----------------------------------------------------------------------------//
//  vstus
//  - the no post-update interface is just implemented on software level. it
//  will be transformed into post-update mode.
//----------------------------------------------------------------------------//
#define VSTUS(LT, ST)                                                          \
  CCE_INTRINSIC void vstus(vector_align &alignData,                            \
                           uint32_t vl /* in unit of element */,           \
                           vector_##ST src, __ubuf__ LT *base) {               \
    vector_align ret;                                                          \
    __builtin_cce_vstus_##ST(&ret, src, base, vl * sizeof(LT), alignData); \
    alignData = ret;                                                           \
    return;                                                                    \
  }                                                                            \
  template <class T>                                                           \
  CCE_INTRINSIC void vstus(vector_align &alignData,                            \
                           uint32_t offset /* in unit of element */,           \
                           vector_##ST src, __ubuf__ LT *&base, T post) {      \
    static_assert(std::is_same<T, PostUpdateType>::value,                      \
                  "The last argument can only be 'POST_UPDATE' .");            \
    struct {                                                                   \
      vector_align_data alignData;                                             \
      __ubuf__ LT *base;                                                       \
    } ret;                                                                     \
    __builtin_cce_vstus_post_##ST(&ret, src, base, offset * sizeof(LT),        \
                                  alignData);                                  \
    alignData = ret.alignData;                                                 \
    base = ret.base;                                                           \
    return;                                                                    \
  }
VSTUS(int8_t, s8)
VSTUS(uint8_t, u8)
VSTUS(int16_t, s16)
VSTUS(uint16_t, u16)
VSTUS(int32_t, s32)
VSTUS(uint32_t, u32)
VSTUS(half, f16)
VSTUS(float, f32)
VSTUS(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTUS(bfloat16_t, bf16)
#endif
#undef VSTUS

//  vstui
//  - the no post-update interface is just implemented on software level. it
//  will be transformed into post-update mode.
//----------------------------------------------------------------------------//
#define VSTUI(LT, ST)                                                          \
  DEPRECATED CCE_INTRINSIC void vstui(                                         \
      vector_align &alignData, uint32_t vl /* in unit of element */,       \
      vector_##ST src, __ubuf__ LT *base) {                                    \
    vector_align ret;                                                          \
    __builtin_cce_vstui_##ST(&ret, src, base, vl * sizeof(LT), alignData); \
    alignData = ret;                                                           \
    return;                                                                    \
  }                                                                            \
  template <class T>                                                           \
  DEPRECATED CCE_INTRINSIC void vstui(                                         \
      vector_align &alignData, uint32_t offset /* in unit of element */,       \
      vector_##ST src, __ubuf__ LT *&base, T post) {                           \
    static_assert(std::is_same<T, PostUpdateType>::value,                      \
                  "The last argument can only be 'POST_UPDATE'.");             \
    struct {                                                                   \
      vector_align_data alignData;                                             \
      __ubuf__ LT *base;                                                       \
    } ret;                                                                     \
    __builtin_cce_vstui_post_##ST(&ret, src, base, offset * sizeof(LT),        \
                                  alignData);                                  \
    alignData = ret.alignData;                                                 \
    base = ret.base;                                                           \
    return;                                                                    \
  }
VSTUI(int8_t, s8)
VSTUI(uint8_t, u8)
VSTUI(int16_t, s16)
VSTUI(uint16_t, u16)
VSTUI(int32_t, s32)
VSTUI(uint32_t, u32)
VSTUI(half, f16)
VSTUI(float, f32)
VSTUI(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTUI(bfloat16_t, bf16)
#endif
#undef VSTUI
//----------------------------------------------------------------------------//
#define VSTA(LT, ST)                                                           \
  CCE_INTRINSIC void vsta(vector_align data, __ubuf__ LT *base,                \
                          vector_address offset) {                             \
    __builtin_cce_vsta_##ST(data, base, offset, 0 /* #loop */);                \
  }

VSTA(int8_t, s8)
VSTA(uint8_t, u8)
VSTA(int16_t, s16)
VSTA(uint16_t, u16)
VSTA(int32_t, s32)
VSTA(uint32_t, u32)
VSTA(half, f16)
VSTA(float, f32)
VSTA(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTA(bfloat16_t, bf16)
#endif
#undef VSTA

//----------------------------------------------------------------------------//
//  vstai
//----------------------------------------------------------------------------//
#define VSTAI(LT, ST)                                                          \
  DEPRECATED CCE_INTRINSIC void vstai(                                         \
      vector_align data, __ubuf__ LT *base,                                    \
      int32_t offset /* in unit of element */) {                               \
    __builtin_cce_vstai_##ST(data, base, offset * sizeof(LT),                  \
                             0 /* pre mode*/);                                 \
  }

VSTAI(int8_t, s8)
VSTAI(uint8_t, u8)
VSTAI(int16_t, s16)
VSTAI(uint16_t, u16)
VSTAI(int32_t, s32)
VSTAI(uint32_t, u32)
VSTAI(half, f16)
VSTAI(float, f32)
VSTAI(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTAI(bfloat16_t, bf16)
#endif
#undef VSTAI

#define VSTAI_POST(LT, ST)                                                     \
  template <class T>                                                           \
  DEPRECATED CCE_INTRINSIC void vstai(vector_align data, __ubuf__ LT *&base,   \
                                      int32_t offset /* in unit of element */, \
                                      T post) {                                \
    static_assert(std::is_same<T, PostUpdateType>::value,                      \
                  "The last argument can only be 'POST_UPDATE'.");             \
    base = (__ubuf__ LT *)__builtin_cce_vstai_post_##ST(                       \
        data, base, offset * sizeof(LT), 1 /* post mode */);                   \
  }

VSTAI_POST(int8_t, s8)
VSTAI_POST(uint8_t, u8)
VSTAI_POST(int16_t, s16)
VSTAI_POST(uint16_t, u16)
VSTAI_POST(int32_t, s32)
VSTAI_POST(uint32_t, u32)
VSTAI_POST(half, f16)
VSTAI_POST(float, f32)
VSTAI_POST(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTAI_POST(bfloat16_t, bf16)
#endif
#undef VSTAI_POST

//----------------------------------------------------------------------------//
//  vstas
//----------------------------------------------------------------------------//
#define VSTAS(LT, ST)                                                          \
  CCE_INTRINSIC void vstas(vector_align data, __ubuf__ LT *base,               \
                           int32_t offset /* in unit of element */) {          \
    __builtin_cce_vstas_##ST(data, base, offset * sizeof(LT),                  \
                             0 /* pre mode*/);                                 \
  }

VSTAS(int8_t, s8)
VSTAS(uint8_t, u8)
VSTAS(int16_t, s16)
VSTAS(uint16_t, u16)
VSTAS(uint32_t, u32)
VSTAS(half, f16)
VSTAS(float, f32)
VSTAS(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTAS(bfloat16_t, bf16)
#endif
#undef VSTAS

#define VSTAS_POST(LT, ST)                                                     \
  template <class T>                                                           \
  CCE_INTRINSIC void vstas(vector_align data, __ubuf__ LT *&base,              \
                           int32_t offset /* in unit of element */, T post) {  \
    static_assert(std::is_same<T, PostUpdateType>::value,                      \
                  "The last argument can only be 'POST_UPDATE'.");             \
    base = (__ubuf__ LT *)__builtin_cce_vstas_post_##ST(                       \
        data, base, offset * sizeof(LT), 1 /* post mode */);                   \
  }

VSTAS_POST(int8_t, s8)
VSTAS_POST(uint8_t, u8)
VSTAS_POST(int16_t, s16)
VSTAS_POST(uint16_t, u16)
VSTAS_POST(int32_t, s32)
VSTAS_POST(uint32_t, u32)
VSTAS_POST(half, f16)
VSTAS_POST(float, f32)
VSTAS_POST(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTAS_POST(bfloat16_t, bf16)
#endif
#undef VSTAS_POST

//----------------------------------------------------------------------------//
//  vstur
//----------------------------------------------------------------------------//
#define INVALID_VALUE_VSTUR "The 4th argument of vstur is not valid"

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
#define VSTUR(LT, ST)                                                          \
  template <class T>                                                           \
  CCE_INTRINSIC void vstur(vector_align &alignData, vector_##ST src,           \
                           __ubuf__ LT *base, T post) {                        \
    static_assert(std::is_same<T, NoPostUpdateType>::value ||                  \
                      std::is_same<T, PostUpdateType>::value,                  \
                  INVALID_VALUE_VSTUR);                                        \
    alignData = __builtin_cce_vstur_##ST(src, base, alignData, 0 /* #loop */,  \
                                         (ULL)post.value);                     \
    return;                                                                    \
  }

VSTUR(int8_t, s8)
VSTUR(uint8_t, u8)
VSTUR(int16_t, s16)
VSTUR(uint16_t, u16)
VSTUR(int32_t, s32)
VSTUR(uint32_t, u32)
VSTUR(half, f16)
VSTUR(float, f32)
#elif defined(__DAV_M300__) || defined(__DAV_L300__) ||                        \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VSTUR(LT, ST)                                                          \
  template <class T>                                                           \
  CCE_INTRINSIC void vstur(vector_align &alignData, vector_##ST src,           \
                           __ubuf__ LT *base, T post) {                        \
    static_assert(std::is_same<T, NoPostUpdateType>::value ||                  \
                      std::is_same<T, PostUpdateType>::value,                  \
                  INVALID_VALUE_VSTUR);                                        \
    alignData = __builtin_cce_vstur_##ST(src, base, alignData,                 \
                                         (ULL)post.value, 0 /* #loop */);      \
    return;                                                                    \
  }
VSTUR(int8_t, s8)
VSTUR(uint8_t, u8)
VSTUR(int16_t, s16)
VSTUR(uint16_t, u16)
VSTUR(int32_t, s32)
VSTUR(uint32_t, u32)
VSTUR(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTUR(bfloat16_t, bf16)
#endif
VSTUR(half, f16)
VSTUR(float, f32)
#endif
//----------------------------------------------------------------------------//
//  vstar
//----------------------------------------------------------------------------//
#define VSTAR(LT, ST)                                                          \
  CCE_INTRINSIC void vstar(vector_align data, __ubuf__ LT *base) {             \
    __builtin_cce_vstar_##ST(data, base, 0 /* #loop */);                       \
  }

VSTAR(int8_t, s8)
VSTAR(uint8_t, u8)
VSTAR(int16_t, s16)
VSTAR(uint16_t, u16)
VSTAR(int32_t, s32)
VSTAR(uint32_t, u32)
VSTAR(half, f16)
VSTAR(float, f32)
VSTAR(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSTAR(bfloat16_t, bf16)
#endif
#undef VSTAR

//----------------------------------------------------------------------------//
//  sprclr
//----------------------------------------------------------------------------//
enum class Spr {
  SPR_AR_VALUE = 74,
};

typedef std::integral_constant<Spr, Spr::SPR_AR_VALUE> SPR_AR_TYPE;
#define SPR_AR SPR_AR_TYPE()

#define INVALID_VALUE_SPRCLR "the argument of sprclr is not valid"

template <class T> CCE_INTRINSIC void sprclr(T spr_id) {
  static_assert(std::is_same<T, SPR_AR_TYPE>::value, INVALID_VALUE_SPRCLR);
  __builtin_cce_sprclr((uint16_t)spr_id.value);
  return;
}

//----------------------------------------------------------------------------//
//  sprsti
//----------------------------------------------------------------------------//
#define INVALID_VALUE_SPRSTI "the argument of sprsti is not valid"

template <class T>
CCE_INTRINSIC void sprsti(T spr_id, __ubuf__ uint32_t *base, int32_t offset) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_SPRSTI);
  return __builtin_cce_sprsti((uint16_t)spr_id.value, base, offset,
                              0 /* post */);
}

template <class T0, class T1>
CCE_INTRINSIC void sprsti(T0 spr_id, __ubuf__ uint32_t *&base, int32_t offset,
                          T1 post) {
  static_assert(std::is_class<T0>::value, INVALID_VALUE_SPRSTI);
  static_assert(std::is_same<T1, PostUpdateType>::value ||
                    std::is_same<T1, NoPostUpdateType>::value,
                "Mode can only be 'POST_UPDATE' or 'NO_POST_UPDATE'.");
  if (std::is_same<T1, PostUpdateType>::value) {
    base = (__ubuf__ uint32_t *)__builtin_cce_sprsti_post(
        (uint16_t)spr_id.value, base, offset, 1 /* post */);
  } else {
    __builtin_cce_sprsti((uint16_t)spr_id.value, base, offset, 0 /* post */);
  }
  return;
}

//----------------------------------------------------------------------------//
//  sprsts
//----------------------------------------------------------------------------//
#define INVALID_VALUE_SPRSTS "the argument of sprsts is not valid"

template <class T>
CCE_INTRINSIC void sprsts(T spr_id, __ubuf__ uint32_t *base, int32_t offset) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_SPRSTS);
  return __builtin_cce_sprsts((uint16_t)spr_id.value, base, offset,
                              0 /* post */);
}

template <class T0, class T1>
CCE_INTRINSIC void sprsts(T0 spr_id, __ubuf__ uint32_t *&base, int32_t offset,
                          T1 post) {
  static_assert(std::is_class<T0>::value, INVALID_VALUE_SPRSTS);
  static_assert(std::is_same<T1, PostUpdateType>::value ||
                    std::is_same<T1, NoPostUpdateType>::value,
                "Mode can only be 'POST_UPDATE' or 'NO_POST_UPDATE'.");
  if (std::is_same<T1, PostUpdateType>::value) {
    base = (__ubuf__ uint32_t *)__builtin_cce_sprsts_post(
        (uint16_t)spr_id.value, base, offset, 1 /* post */);
  } else {
    __builtin_cce_sprsts((uint16_t)spr_id.value, base, offset, 0 /* post */);
  }
  return;
}

//----------------------------------------------------------------------------//
//  vscatter
//----------------------------------------------------------------------------//
#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
#define VSCATTER(STYPE, DATATYPE, INDEXTYPE, NUM)                              \
  CCE_INTRINSIC void vscatter(vector_##DATATYPE data, __ubuf__ STYPE *base,    \
                              vector_##INDEXTYPE index, vector_bool mask) {    \
    return __builtin_cce_vscatter_##NUM##DATATYPE(data, base, index,           \
                                                  /*loop*/ 0, mask);           \
  }
VSCATTER(int8_t, s8, u16, v256)
VSCATTER(uint8_t, u8, u16, v256)
VSCATTER(int16_t, s16, u16, v128)
VSCATTER(uint16_t, u16, u16, v128)
VSCATTER(int32_t, s32, u32, v64)
VSCATTER(uint32_t, u32, u32, v64)
VSCATTER(half, f16, u16, v128)
VSCATTER(float, f32, u32, v64)
#undef VSCATTER
#elif defined(__DAV_M300__) || defined(__DAV_L300__) ||                        \
    defined(__DAV_L300_VEC__) || defined(__DAV_C310__) ||                      \
    defined(__DAV_T300__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VSCATTER(STYPE, DATATYPE, INDEXTYPE, NUM)                              \
  CCE_INTRINSIC void vscatter(vector_##DATATYPE data, __ubuf__ STYPE *base,    \
                              vector_##INDEXTYPE index, vector_bool mask) {    \
    return __builtin_cce_vscatter_##NUM##DATATYPE##_v300(data, base, index,    \
                                                         mask);                \
  }
VSCATTER(int8_t, s8, u16, v256)
VSCATTER(uint8_t, u8, u16, v256)
VSCATTER(int16_t, s16, u16, v128)
VSCATTER(uint16_t, u16, u16, v128)
VSCATTER(int32_t, s32, u32, v64)
VSCATTER(uint32_t, u32, u32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VSCATTER(bfloat16_t, bf16, u16, v128)
#endif
VSCATTER(half, f16, u16, v128)
VSCATTER(float, f32, u32, v64)
#undef VSCATTER
#endif

//----------------------------------------------------------------------------//
//  vsld
//----------------------------------------------------------------------------//
enum class Stride {
  STRIDE_S3_B16,
  STRIDE_S4_B64,
  STRIDE_S8_B32,
  STRIDE_S2_B64
};
typedef std::integral_constant<Stride, Stride::STRIDE_S3_B16> S3_B16_Type;
typedef std::integral_constant<Stride, Stride::STRIDE_S4_B64> S4_B64_Type;
typedef std::integral_constant<Stride, Stride::STRIDE_S8_B32> S8_B32_Type;
typedef std::integral_constant<Stride, Stride::STRIDE_S2_B64> S2_B64_Type;
#define S3_B16 S3_B16_Type()
#define S4_B64 S4_B64_Type()
#define S8_B32 S8_B32_Type()
#define S2_B64 S2_B64_Type()

#define INVALID_VALUE_VSLD "The 4th argument of vsld is not valid"

#if defined(__DAV_L210__) || defined(__DAV_M300__) || defined(__DAV_L300__) || \
    defined(__DAV_L300_VEC__)
#define VSLD(LT, ST, NUM)                                                      \
  template <class T>                                                           \
  CCE_INTRINSIC void vsld(vector_##ST &dst, __ubuf__ LT *base,                 \
                          vector_address offset, T stride) {                   \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VSLD);                \
    static_assert(std::is_same<T, S3_B16_Type>::value ||                       \
                      std::is_same<T, S4_B64_Type>::value ||                   \
                      std::is_same<T, S8_B32_Type>::value ||                   \
                      std::is_same<T, S2_B64_Type>::value,                     \
                  INVALID_VALUE_VSLD);                                         \
    dst = __builtin_cce_vsld_##NUM##ST(base, offset, (ULL)stride.value,        \
                                       0 /* #loop */);                         \
    return;                                                                    \
  }

VSLD(int8_t, s8, v256)
VSLD(uint8_t, u8, v256)
VSLD(int16_t, s16, v128)
VSLD(uint16_t, u16, v128)
VSLD(int32_t, s32, v64)
VSLD(uint32_t, u32, v64)
VSLD(half, f16, v128)
VSLD(float, f32, v64)
VSLD(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) || defined(__DAV_L300_VEC__)
VSLD(bfloat16_t, bf16, v128)
#endif
#undef VSLD
#elif defined(__DAV_M210_VEC__)
#define VSLD(LT, ST, NUM)                                                      \
  template <class T>                                                           \
  CCE_INTRINSIC void vsld(vector_##ST &dst, __ubuf__ LT *base,                 \
                          vector_address offset, T stride) {                   \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VSLD);                \
    static_assert(std::is_same<T, S3_B16_Type>::value ||                       \
                      std::is_same<T, S4_B64_Type>::value,                     \
                  INVALID_VALUE_VSLD);                                         \
    dst = __builtin_cce_vsld_##NUM##ST(base, offset, (ULL)stride.value,        \
                                       0 /* #loop */);                         \
    return;                                                                    \
  }

VSLD(int8_t, s8, v256)
VSLD(uint8_t, u8, v256)
VSLD(int16_t, s16, v128)
VSLD(uint16_t, u16, v128)
VSLD(int32_t, s32, v64)
VSLD(uint32_t, u32, v64)
VSLD(half, f16, v128)
VSLD(float, f32, v64)
#undef VSLD
#endif

//----------------------------------------------------------------------------//
//  vsst
//----------------------------------------------------------------------------//
#if defined(__DAV_L210__) || defined(__DAV_M300__) || defined(__DAV_L300__) || \
    defined(__DAV_L300_VEC__)
enum class StrideVSST { STRIDE_VSST_S8_B16 };
typedef std::integral_constant<StrideVSST, StrideVSST::STRIDE_VSST_S8_B16>
    S8_B16_Type;
#define S8_B16 S8_B16_Type()

#define INVALID_VALUE_VSST "The 4th argument of vsst is not valid"

#define VSST(LT, ST, NUM)                                                      \
  template <class T>                                                           \
  CCE_INTRINSIC void vsst(vector_##ST src, __ubuf__ LT *base,                  \
                          vector_address offset, T stride) {                   \
    static_assert(std::is_same<T, S8_B16_Type>::value, INVALID_VALUE_VSST);    \
    return __builtin_cce_vsst_##NUM##ST(src, base, offset, (ULL)stride.value,  \
                                        0 /* #loop */);                        \
  }

VSST(int8_t, s8, v256)
VSST(uint8_t, u8, v256)
VSST(int16_t, s16, v128)
VSST(uint16_t, u16, v128)
VSST(int32_t, s32, v64)
VSST(uint32_t, u32, v64)
VSST(half, f16, v128)
VSST(float, f32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) || defined(__DAV_L300_VEC__)
VSST(bfloat16_t, bf16, v128)
#endif
#undef VSST
#endif

enum class Part_T { P0, P1, P2, P3 };
typedef std::integral_constant<Part_T, Part_T::P0> PartP0Type;
typedef std::integral_constant<Part_T, Part_T::P1> PartP1Type;
typedef std::integral_constant<Part_T, Part_T::P2> PartP2Type;
typedef std::integral_constant<Part_T, Part_T::P3> PartP3Type;
#define PART_P0 PartP0Type()
#define PART_P1 PartP1Type()
#define PART_P2 PartP2Type()
#define PART_P3 PartP3Type()

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) ||                      \
    defined(__DAV_T210__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_L310__)
//----------------------------------------------------------------------------//
//  prpnset
//----------------------------------------------------------------------------//
#define PRPNSET(TYPE)                                                          \
  CCE_INTRINSIC void prpnset_##TYPE(Part_T part, vector_bool pg) {             \
    __builtin_cce_prpnset_##TYPE((uint16_t)part, pg);                          \
    return;                                                                    \
  }

PRPNSET(f16)
#undef PRPNSET
#endif

//----------------------------------------------------------------------------//
//  vgatherb
//----------------------------------------------------------------------------//
#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) ||                      \
    defined(__DAV_T210__) || defined(__DAV_M300__) || defined(__DAV_L300__) || \
    defined(__DAV_L300_VEC__) || defined(__DAV_L310__) ||                      \
    defined(__DAV_M310__)
#define VGATHERB(ST, NUM, LT)                                                  \
  CCE_INTRINSIC void vgatherb(                                                 \
      vector_##ST &dst, __ubuf__ uint32_t *indexAddrBase,                      \
      vector_address addrOffset, uint32_t indexOffset) {                       \
    if (isV300Target() || isV310Target()) {                                    \
      vector_u32 vnOffset = __builtin_cce_vldx1_v64u32(                        \
          indexAddrBase, addrOffset, (ULL)NORM.value, 0);                      \
      dst = __builtin_cce_vgatherb_v300_##NUM##ST((__ubuf__ LT *)indexOffset,  \
                                                  vnOffset);                   \
    } else {                                                                   \
      dst = __builtin_cce_vgatherb_##NUM##ST(indexAddrBase, addrOffset,        \
                                             indexOffset, 0 /* #loop */);      \
    }                                                                          \
    return;                                                                    \
  }

VGATHERB(s8, v256, int8_t)
VGATHERB(u8, v256, uint8_t)
VGATHERB(s16, v128, int16_t)
VGATHERB(u16, v128, uint16_t)
VGATHERB(s32, v64, int32_t)
VGATHERB(u32, v64, uint32_t)
VGATHERB(f16, v128, half)
VGATHERB(f32, v64, float)
VGATHERB(s64, v32, int64_t)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VGATHERB(bf16, v128, bfloat16_t)
#endif
#endif
#undef VGATHERB

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VGATHERB(LT, ST, NUM)                                                  \
  CCE_INTRINSIC void vgatherb(vector_##ST &dst, __ubuf__ LT *base,             \
                              vector_u32 indexOffset) {                        \
    dst = __builtin_cce_vgatherb_v300_##NUM##ST(base, indexOffset);            \
    return;                                                                    \
  }
VGATHERB(int8_t, s8, v256)
VGATHERB(uint8_t, u8, v256)
VGATHERB(int16_t, s16, v128)
VGATHERB(uint16_t, u16, v128)
VGATHERB(int32_t, s32, v64)
VGATHERB(uint32_t, u32, v64)
VGATHERB(int64_t, s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VGATHERB(bfloat16_t, bf16, v128)
#endif
VGATHERB(half, f16, v128)
VGATHERB(float, f32, v64)
#undef VGATHERB
#endif

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
//----------------------------------------------------------------------------//
//  vgather2
//  For type = b8, read VL/2 indexes, each index is an unsigned 16-bit integer.
//  For 8-bit element, the result in the destination vector register is
//  zero-extended to 16-bit.
//----------------------------------------------------------------------------//
#define VGATHER2(LT, VT, OT, ST, NUM)                                          \
  template <class T = std::integral_constant<Mode, Mode::UNKNOWN_VALUE>>       \
  CCE_INTRINSIC void vgather2(vector_##VT &dst, __ubuf__ LT *base,             \
                              vector_##OT indexOffset, vector_bool mask,       \
                              T mode = MODE_UNKNOWN) {                         \
    static_assert(mode.value == MODE_ZEROING.value ||                          \
                      mode.value == MODE_UNKNOWN.value,                        \
                  "The last argument can only be 'MODE_ZEROING', "             \
                  "'MODE_UNKNOWN' or empty.");                                 \
    dst = __builtin_cce_vgather2_v300_##NUM##ST(base, indexOffset, mask);      \
    return;                                                                    \
  }
VGATHER2(int8_t, s16, u16, s8, v256)
VGATHER2(uint8_t, u16, u16, u8, v256)
VGATHER2(int16_t, s16, u16, s16, v128)
VGATHER2(uint16_t, u16, u16, u16, v128)
VGATHER2(int32_t, s32, u32, s32, v64)
VGATHER2(uint32_t, u32, u32, u32, v64)
VGATHER2(half, f16, u16, f16, v128)
VGATHER2(float, f32, u32, f32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VGATHER2(bfloat16_t, bf16, u16, bf16, v128)
#endif
#undef VGATHER2
#endif

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) ||                      \
    defined(__DAV_T210__) || defined(__DAV_M300__) || defined(__DAV_L300__) || \
    defined(__DAV_L300_VEC__) || defined(__DAV_L310__)
#define VGATHER2(INDEXT, LT, VT, ST, NUM, LD_INDEX_NUM, LD_VT)                 \
  CCE_INTRINSIC void vgather2(                                                 \
      vector_##VT &dst, __ubuf__ INDEXT *indexAddrBase,                        \
      vector_address addrOffset, __ubuf__ LT *offsetAddr) {                    \
    if (isV300Target() || isV310Target()) {                                    \
      vector_##VT vnOffset = __builtin_cce_vldx1_##LD_INDEX_NUM##LD_VT(        \
          indexAddrBase, addrOffset, (ULL)NORM.value, 0);                      \
      vector_bool mask = pge_b8(PAT_ALL);                                      \
      vnOffset = __builtin_cce_vshrs_##LD_INDEX_NUM##LD_VT##_x(                \
          vnOffset, (int16_t)(sizeof(INDEXT) / 2), mask);                      \
      dst = __builtin_cce_vgather2_v300_##NUM##ST(offsetAddr, vnOffset, mask); \
    } else {                                                                   \
      dst = __builtin_cce_vgather2_##NUM##ST(indexAddrBase, addrOffset,        \
                                             offsetAddr, 0 /* #loop */);       \
    }                                                                          \
    return;                                                                    \
  }

VGATHER2(uint16_t, int8_t, s16, s8, v256, v128, u16)
VGATHER2(uint16_t, uint8_t, u16, u8, v256, v128, u16)
VGATHER2(uint16_t, int16_t, s16, s16, v128, v128, u16)
VGATHER2(uint16_t, uint16_t, u16, u16, v128, v128, u16)
VGATHER2(uint32_t, int32_t, s32, s32, v64, v64, u32)
VGATHER2(uint32_t, uint32_t, u32, u32, v64, v64, u32)
VGATHER2(uint32_t, float, f32, f32, v64, v64, u32)
VGATHER2(uint16_t, half, f16, f16, v128, v128, u16)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VGATHER2(uint16_t, bfloat16_t, bf16, bf16, v128, v128, u16)
#endif
#endif
#undef VGATHER2

//----------------------------------------------------------------------------//
// vgather2_bc
// For Vn contains VL/4 indexes, each index is an unsigned 32-bit integer.
// If type=b16, each gathered element occupied 32bit with upper 16bit zero.
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VGATHER2_BC(LT, VT, OT, ST, NUM)                                       \
  CCE_INTRINSIC void vgather2_bc(vector_##VT &dst, __ubuf__ LT *base,          \
                                 vector_##OT indexOffset, vector_bool mask) {  \
    dst = __builtin_cce_vgather2_bc_##NUM##ST(base, indexOffset, mask);        \
    return;                                                                    \
  }

VGATHER2_BC(int16_t, s16, u32, s16, v128)
VGATHER2_BC(uint16_t, u16, u32, u16, v128)
VGATHER2_BC(int32_t, s32, u32, s32, v64)
VGATHER2_BC(uint32_t, u32, u32, u32, v64)
VGATHER2_BC(half, f16, u32, f16, v128)
VGATHER2_BC(float, f32, u32, f32, v64)
#undef VGATHER2_BC
#endif

//----------------------------------------------------------------------------//
//  vfcvt
//  For 32-bit to 16-bit and 16-bit to 8-bit data width conversion,
//  the destination element is placed into the even part or odd part
//  of each wider element position while the other part remains its old values.
//  So it should be merging mode.
//  But sometimes the user just want to get only one part of the odd and even,
//  so it also need to provide unknown mode.
//----------------------------------------------------------------------------//
enum class ROUND { R, A, F, C, Z, O };
typedef std::integral_constant<ROUND, ROUND::R> RoundRType;
typedef std::integral_constant<ROUND, ROUND::A> RoundAType;
typedef std::integral_constant<ROUND, ROUND::F> RoundFType;
typedef std::integral_constant<ROUND, ROUND::C> RoundCType;
typedef std::integral_constant<ROUND, ROUND::Z> RoundZType;
typedef std::integral_constant<ROUND, ROUND::O> RoundOType;
#define ROUND_R RoundRType()
#define ROUND_A RoundAType()
#define ROUND_F RoundFType()
#define ROUND_C RoundCType()
#define ROUND_Z RoundZType()
#define ROUND_O RoundOType()

enum class Part { EVEN, ODD };
typedef std::integral_constant<Part, Part::EVEN> PartEvenType;
typedef std::integral_constant<Part, Part::ODD> PartOddType;
#define PART_EVEN PartEvenType()
#define PART_ODD PartOddType()

enum class RoundingSaturation { RS_DISABLE_VALUE, RS_ENABLE_VALUE };
typedef std::integral_constant<RoundingSaturation,
                               RoundingSaturation::RS_DISABLE_VALUE>
    RSDisableType;
typedef std::integral_constant<RoundingSaturation,
                               RoundingSaturation::RS_ENABLE_VALUE>
    RSEnableType;
#define RS_DISABLE RSDisableType()
#define RS_ENABLE RSEnableType()

#define VFCVT(TO, RND, NAME)                                                   \
  template <class T1, class T2, class T3 = Mode_Unknown_Type>                  \
  CCE_INTRINSIC void vfcvt(vector_##TO &dst, vector_f32 src, T1 rnd, T2 part,  \
                           T3 mode = MODE_UNKNOWN) {                           \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T3>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, Round##RND##Type>::value,               \
                  "The 3rd argument of this vfcvt (f322" #TO                   \
                  ") can only be: ROUND_R, ROUND_" #RND);                      \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  "The 4th argument of this vfcvt (f322" #TO                   \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The last argument of this vfcvt (f322" #TO ") can only be: "          \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    const unsigned round =                                                     \
        ((ULL)rnd.value == 4) ? 1 : (((ULL)rnd.value == 5) ? 1 : 0);           \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask1 = pset_b8(PAT_ALL);                                    \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask = pset_b8(PAT_M4);                                    \
        if (part.value == Part::ODD) {                                         \
          mask = __builtin_cce_pnot_z(mask, mask1);                            \
        }                                                                      \
        vector_##TO dstTmp = __builtin_cce_##NAME##_f322##TO##_x(              \
            src, mask1, (ULL)rnd.value,                                        \
            (ULL)RoundingSaturation::RS_DISABLE_VALUE, (ULL)part.value);       \
        dst = __builtin_cce_vmov_v128##TO##_m(dst, dstTmp, mask);              \
      } else {                                                                 \
        dst = __builtin_cce_##NAME##_f322##TO##_x(                             \
            src, mask1, (ULL)rnd.value,                                        \
            (ULL)RoundingSaturation::RS_DISABLE_VALUE, (ULL)part.value);       \
      }                                                                        \
    } else {                                                                   \
      dst =                                                                    \
          (mode.value == MODE_MERGING.value)                                   \
              ? __builtin_cce_vfcvt_f322##TO##_m(dst, src, round,              \
                                                 (ULL)part.value)              \
              : __builtin_cce_vfcvt_f322##TO##_x(src, round, (ULL)part.value); \
    }                                                                          \
  }
VFCVT(f16, O, vcvtff)
VFCVT(s16, Z, vcvtfi)
#undef VFCVT

#define VFCVT(TO)                                                              \
  template <class T1, class T2, class T3 = Mode_Unknown_Type>                  \
  CCE_INTRINSIC void vfcvt(vector_##TO &dst, vector_f16 src, T1 rnd, T2 part,  \
                           T3 mode = MODE_UNKNOWN) {                           \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T3>::value, "the last argument is not valid"); \
    static_assert(                                                             \
        std::is_same<T1, RoundRType>::value ||                                 \
            std::is_same<T1, RoundAType>::value ||                             \
            std::is_same<T1, RoundFType>::value ||                             \
            std::is_same<T1, RoundCType>::value ||                             \
            std::is_same<T1, RoundZType>::value,                               \
        "The 4th argument of this vfcvt (f162" #TO                             \
        ") can only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");         \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  "The 5th argument of this vfcvt (f162" #TO                   \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The last argument of this vfcvt (f162" #TO ") can only be: "          \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask1 = pset_b8(PAT_ALL);                                    \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask = pset_b8(PAT_H);                                     \
        mask = __builtin_cce_punpack(mask, (ULL)LOWER.value);                  \
        if (part.value == Part::ODD) {                                         \
          mask = __builtin_cce_pnot_z(mask, mask1);                            \
        }                                                                      \
        vector_##TO dstTmp = __builtin_cce_vcvtfi_f162##TO##_x(                \
            src, mask1, (ULL)rnd.value,                                        \
            (ULL)RoundingSaturation::RS_DISABLE_VALUE, (ULL)part.value);       \
        dst = __builtin_cce_vmov_v256##TO##_m(dst, dstTmp, mask);              \
      } else {                                                                 \
        dst = __builtin_cce_vcvtfi_f162##TO##_x(                               \
            src, mask1, (ULL)rnd.value,                                        \
            (ULL)RoundingSaturation::RS_DISABLE_VALUE, (ULL)part.value);       \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vfcvt_f162##TO##_m(dst, src, (ULL)rnd.value,   \
                                                   (ULL)part.value)            \
                : __builtin_cce_vfcvt_f162##TO##_x(src, (ULL)rnd.value,        \
                                                   (ULL)part.value);           \
    }                                                                          \
  }
VFCVT(s8)
VFCVT(u8)
#undef VFCVT

// narrower to wider
#define VFCVT(TO)                                                              \
  template <class T1, class T2>                                                \
  CCE_INTRINSIC void vfcvt(vector_##TO &dst, vector_f16 src, T1 rnd,           \
                           T2 part) {                                          \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 5th argument is not valid");  \
    static_assert(                                                             \
        std::is_same<T1, RoundRType>::value ||                                 \
            std::is_same<T1, RoundAType>::value ||                             \
            std::is_same<T1, RoundFType>::value ||                             \
            std::is_same<T1, RoundCType>::value ||                             \
            std::is_same<T1, RoundZType>::value,                               \
        "The 4th argument of this vfcvt (f162" #TO                             \
        ") can only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");         \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  "The 5th argument of this vfcvt (f162" #TO                   \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    if (isV300Target() || isV310Target()) {                                    \
      vector_bool mask = pset_b16(PAT_ALL);                                    \
      dst = __builtin_cce_vcvtfi_f162##TO##_x(src, mask, (ULL)rnd.value,       \
                                              (ULL)part.value);                \
    } else {                                                                   \
      dst =                                                                    \
          __builtin_cce_vfcvt_f162##TO(src, (ULL)rnd.value, (ULL)part.value);  \
    }                                                                          \
  }
VFCVT(s32)
#undef VFCVT

#define VFCVT(FROM, TO, NUM)                                                   \
  template <class T3 = Mode_Unknown_Type>                                      \
  CCE_INTRINSIC void vfcvt(vector_##TO &dst, vector_##FROM src,                \
                           vector_bool mask, T3 mode = MODE_UNKNOWN) {         \
    static_assert(std::is_class<T3>::value, "the 4th argument is not valid");  \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The 4th argument of this vfcvt (" #FROM "2" #TO ") can only be: "     \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_##TO dstTmp = __builtin_cce_vcvtfi_##FROM##2##TO##_x(           \
            src, mask, (ULL)ROUND_R.value, (ULL)RS_DISABLE.value);             \
        dst = __builtin_cce_vmov_##NUM##TO##_m(dst, dstTmp, mask);             \
      } else {                                                                 \
        dst = __builtin_cce_vcvtfi_##FROM##2##TO##_x(                          \
            src, mask, (ULL)ROUND_R.value, (ULL)RS_DISABLE.value);             \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vfcvt_##FROM##2##TO##_m(dst, src, mask)        \
                : __builtin_cce_vfcvt_##FROM##2##TO##_x(src, mask);            \
    }                                                                          \
  }
VFCVT(f16, s16, v128)
#undef VFCVT

#define VFCVT(FROM, TO, NUM)                                                   \
  template <class T3 = Mode_Unknown_Type>                                      \
  CCE_INTRINSIC void vfcvt(vector_##TO &dst, vector_##FROM src,                \
                           vector_bool mask, T3 mode = MODE_UNKNOWN) {         \
    static_assert(std::is_class<T3>::value, "the 4th argument is not valid");  \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The 4th argument of this vfcvt (" #FROM "2" #TO ") can only be: "     \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_##TO dstTmp = __builtin_cce_vcvtif_##FROM##2##TO##_x(           \
            src, mask, (ULL)ROUND_R.value);                                    \
        dst = __builtin_cce_vmov_##NUM##TO##_m(dst, dstTmp, mask);             \
      } else {                                                                 \
        dst = __builtin_cce_vcvtif_##FROM##2##TO##_x(src, mask,                \
                                                     (ULL)ROUND_R.value);      \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vfcvt_##FROM##2##TO##_m(dst, src, mask)        \
                : __builtin_cce_vfcvt_##FROM##2##TO##_x(src, mask);            \
    }                                                                          \
  }
VFCVT(s16, f16, v128)
VFCVT(s32, f32, v64)
#undef VFCVT

#define VFCVT(FROM, TO, NUM)                                                   \
  template <class T2, class T3 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vfcvt(vector_##TO &dst, vector_##FROM src,                \
                           vector_bool mask, T2 rnd, T3 mode = MODE_UNKNOWN) { \
    static_assert(std::is_class<T2>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T3>::value, "the 5th argument is not valid");  \
    static_assert(                                                             \
        std::is_same<T2, RoundRType>::value ||                                 \
            std::is_same<T2, RoundAType>::value ||                             \
            std::is_same<T2, RoundFType>::value ||                             \
            std::is_same<T2, RoundCType>::value ||                             \
            std::is_same<T2, RoundZType>::value,                               \
        "The 4th argument of this vfcvt (f" #FROM "2" #TO                      \
        ") can only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");         \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The 5th argument of this vfcvt (" #FROM "2" #TO ") can only be: "     \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_##TO dstTmp = __builtin_cce_vcvtfi_##FROM##2##TO##_x(           \
            src, mask, (ULL)rnd.value, (ULL)RS_DISABLE.value);                 \
        dst = __builtin_cce_vmov_##NUM##TO##_m(dst, dstTmp, mask);             \
      } else {                                                                 \
        dst = __builtin_cce_vcvtfi_##FROM##2##TO##_x(                          \
            src, mask, (ULL)rnd.value, (ULL)RS_DISABLE.value);                 \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vfcvt_##FROM##2##TO##_m(dst, src,              \
                                                        (ULL)rnd.value, mask)  \
                : __builtin_cce_vfcvt_##FROM##2##TO##_x(src, (ULL)rnd.value,   \
                                                        mask);                 \
    }                                                                          \
  }
VFCVT(f32, s32, v64)
#undef VFCVT

#define VFCVT(FROM, TO, NAME, TYPE)                                            \
  template <class T2>                                                          \
  CCE_INTRINSIC void vfcvt(vector_##TO &dst, vector_##FROM src, T2 part) {     \
    static_assert(std::is_class<T2>::value, "the 4th argument is not valid");  \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  "The 4th argument of this vfcvt (" #FROM "2" #TO             \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask = pset_##TYPE(PAT_ALL);                                 \
      dst = __builtin_cce_##NAME##_##FROM##2##TO##_x(src, mask,                \
                                                     (ULL)part.value);         \
    } else {                                                                   \
      dst = __builtin_cce_vfcvt_##FROM##2##TO(src, (ULL)part.value);           \
    }                                                                          \
  }
VFCVT(f16, f32, vcvtff, b16)
VFCVT(s16, f32, vcvtif, b16)
VFCVT(u8, f16, vcvtif, b8)
VFCVT(s8, f16, vcvtif, b8)
#undef VFCVT

//----------------------------------------------------------------------------//
//  vsfcvt
//----------------------------------------------------------------------------//
#define VSFCVT(TO, RND)                                                        \
  template <class T1, class T2, class T3 = Mode_Unknown_Type>                  \
  CCE_INTRINSIC void vsfcvt(vector_##TO &dst, vector_f32 src, T1 rnd, T2 part, \
                            T3 mode = MODE_UNKNOWN) {                          \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T3>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, RoundZType>::value,                     \
                  "The 3rd argument of this vsfcvt (f322" #TO                  \
                  ") can only be: ROUND_R, ROUND_" #RND);                      \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  "The 4th argument of this vsfcvt (f322" #TO                  \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The last argument of this vsfcvt (f322" #TO ") can only be: "         \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    const unsigned round = ((ULL)rnd.value == 4) ? 1 : (ULL)rnd.value;         \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask1 = pset_b8(PAT_ALL);                                    \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask = pset_b8(PAT_M4);                                    \
        if (part.value == Part::ODD) {                                         \
          mask = __builtin_cce_pnot_z(mask, mask1);                            \
        }                                                                      \
        vector_##TO dstTmp = __builtin_cce_vcvtfi_f322##TO##_x(                \
            src, mask1, (ULL)rnd.value,                                        \
            (ULL)RoundingSaturation::RS_ENABLE_VALUE, (ULL)part.value);        \
        dst = __builtin_cce_vmov_v128s16_m(dst, dstTmp, mask);                 \
      } else {                                                                 \
        dst = __builtin_cce_vcvtfi_f322##TO##_x(                               \
            src, mask1, (ULL)rnd.value,                                        \
            (ULL)RoundingSaturation::RS_ENABLE_VALUE, (ULL)part.value);        \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vsfcvt_f322##TO##_m(dst, src, round,           \
                                                    (ULL)part.value)           \
                : __builtin_cce_vsfcvt_f322##TO##_x(src, round,                \
                                                    (ULL)part.value);          \
    }                                                                          \
  }
VSFCVT(s16, Z)
#undef VSFCVT

#define VSFCVT(TO)                                                             \
  template <class T1, class T2, class T3 = Mode_Unknown_Type>                  \
  CCE_INTRINSIC void vsfcvt(vector_##TO &dst, vector_f16 src, T1 rnd, T2 part, \
                            T3 mode = MODE_UNKNOWN) {                          \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T3>::value, "the last argument is not valid"); \
    static_assert(                                                             \
        std::is_same<T1, RoundRType>::value ||                                 \
            std::is_same<T1, RoundAType>::value ||                             \
            std::is_same<T1, RoundFType>::value ||                             \
            std::is_same<T1, RoundCType>::value ||                             \
            std::is_same<T1, RoundZType>::value,                               \
        "The 3rd argument of this vsfcvt (f162" #TO                            \
        ") can only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");         \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  "The 4th argument of this vsfcvt (f162" #TO                  \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The last argument of this vsfcvt (f162" #TO ") can only be: "         \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask1 = pset_b8(PAT_ALL);                                    \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask = pset_b8(PAT_H);                                     \
        mask = __builtin_cce_punpack(mask, (ULL)LOWER.value);                  \
        if (part.value == Part::ODD) {                                         \
          mask = __builtin_cce_pnot_z(mask, mask1);                            \
        }                                                                      \
        vector_##TO dstTmp = __builtin_cce_vcvtfi_f162##TO##_x(                \
            src, mask1, (ULL)rnd.value,                                        \
            (ULL)RoundingSaturation::RS_ENABLE_VALUE, (ULL)part.value);        \
        dst = __builtin_cce_vmov_v256##TO##_m(dst, dstTmp, mask);              \
      } else {                                                                 \
        dst = __builtin_cce_vcvtfi_f162##TO##_x(                               \
            src, mask1, (ULL)rnd.value,                                        \
            (ULL)RoundingSaturation::RS_ENABLE_VALUE, (ULL)part.value);        \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vsfcvt_f162##TO##_m(dst, src, (ULL)rnd.value,  \
                                                    (ULL)part.value)           \
                : __builtin_cce_vsfcvt_f162##TO##_x(src, (ULL)rnd.value,       \
                                                    (ULL)part.value);          \
    }                                                                          \
  }
VSFCVT(s8)
VSFCVT(u8)
#undef VSFCVT

#define VSFCVT(FROM, TO, NUM)                                                  \
  template <class T3 = Mode_Unknown_Type>                                      \
  CCE_INTRINSIC void vsfcvt(vector_##TO &dst, vector_##FROM src,               \
                            vector_bool mask, T3 mode = MODE_UNKNOWN) {        \
    static_assert(std::is_class<T3>::value, "the 4th argument is not valid");  \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The 4th argument of this vsfcvt (" #FROM "2" #TO ") can only be: "    \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_##TO dstTmp = __builtin_cce_vcvtfi_##FROM##2##TO##_x(           \
            src, mask, (ULL)ROUND_R.value, (ULL)RS_ENABLE.value);              \
        dst = __builtin_cce_vmov_##NUM##TO##_m(dst, dstTmp, mask);             \
      } else {                                                                 \
        dst = __builtin_cce_vcvtfi_##FROM##2##TO##_x(                          \
            src, mask, (ULL)ROUND_R.value, (ULL)RS_ENABLE.value);              \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vsfcvt_##FROM##2##TO##_m(dst, src, mask)       \
                : __builtin_cce_vsfcvt_##FROM##2##TO##_x(src, mask);           \
    }                                                                          \
  }
VSFCVT(f16, s16, v128)
#undef VSFCVT

#define VSFCVT(FROM, TO, NUM)                                                  \
  template <class T2, class T3 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vsfcvt(vector_##TO &dst, vector_##FROM src,               \
                            vector_bool mask, T2 rnd,                          \
                            T3 mode = MODE_UNKNOWN) {                          \
    static_assert(std::is_class<T2>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T3>::value, "the 5th argument is not valid");  \
    static_assert(                                                             \
        std::is_same<T2, RoundRType>::value ||                                 \
            std::is_same<T2, RoundAType>::value ||                             \
            std::is_same<T2, RoundFType>::value ||                             \
            std::is_same<T2, RoundCType>::value ||                             \
            std::is_same<T2, RoundZType>::value,                               \
        "The 4th argument of this vsfcvt (f" #FROM "2" #TO                     \
        ") can only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");         \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The 5th argument of this vsfcvt (" #FROM "2" #TO ") can only be: "    \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_##TO dstTmp = __builtin_cce_vcvtfi_##FROM##2##TO##_x(           \
            src, mask, (ULL)rnd.value, (ULL)RS_ENABLE.value);                  \
        dst = __builtin_cce_vmov_##NUM##TO##_m(dst, dstTmp, mask);             \
      } else {                                                                 \
        dst = __builtin_cce_vcvtfi_##FROM##2##TO##_x(                          \
            src, mask, (ULL)rnd.value, (ULL)RS_ENABLE.value);                  \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vsfcvt_##FROM##2##TO##_m(dst, src,             \
                                                         (ULL)rnd.value, mask) \
                : __builtin_cce_vsfcvt_##FROM##2##TO##_x(src, (ULL)rnd.value,  \
                                                         mask);                \
    }                                                                          \
  }
VSFCVT(f32, s32, v64)
#undef VSFCVT

//----------------------------------------------------------------------------//
//  vcvt
//----------------------------------------------------------------------------//
#define VCVT(FROM, TO)                                                         \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src, T1 part,        \
                          T2 mode = MODE_UNKNOWN) {                            \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  "The 3rd argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The last argument of this vcvt (" #FROM "2" #TO ") can only be: "     \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask1 = pset_b8(PAT_ALL);                                    \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask = pset_b8(PAT_H);                                     \
        mask = __builtin_cce_punpack(mask, (ULL)LOWER.value);                  \
        if (part.value == Part::ODD) {                                         \
          mask = __builtin_cce_pnot_z(mask, mask1);                            \
        }                                                                      \
        vector_##TO dstTmp = __builtin_cce_vcvtii_##FROM##2##TO##_x(           \
            src, mask1, (ULL)RoundingSaturation::RS_DISABLE_VALUE,             \
            (ULL)part.value);                                                  \
        dst = __builtin_cce_vmov_v256##TO##_m(dst, dstTmp, mask);              \
      } else {                                                                 \
        dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(                          \
            src, mask1, (ULL)RoundingSaturation::RS_DISABLE_VALUE,             \
            (ULL)part.value);                                                  \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vcvt_##FROM##2##TO##_m(dst, src,               \
                                                       (ULL)part.value)        \
                : __builtin_cce_vcvt_##FROM##2##TO##_x(src, (ULL)part.value);  \
    }                                                                          \
  }
VCVT(u16, u8)
VCVT(s16, u8)
#undef VCVT

#define VCVT(FROM, TO)                                                         \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src, T1 part,        \
                          T2 mode = MODE_UNKNOWN) {                            \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  "The 3rd argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The last argument of this vcvt (" #FROM "2" #TO ") can only be: "     \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask1 = pset_b8(PAT_ALL);                                    \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask = pset_b8(PAT_M4);                                    \
        if (part.value == Part::ODD) {                                         \
          mask = __builtin_cce_pnot_z(mask, mask1);                            \
        }                                                                      \
        vector_##TO dstTmp = __builtin_cce_vcvtii_##FROM##2##TO##_x(           \
            src, mask1, (ULL)RoundingSaturation::RS_DISABLE_VALUE,             \
            (ULL)part.value);                                                  \
        dst = __builtin_cce_vmov_v128##TO##_m(dst, dstTmp, mask);              \
      } else {                                                                 \
        dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(                          \
            src, mask1, (ULL)RoundingSaturation::RS_DISABLE_VALUE,             \
            (ULL)part.value);                                                  \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vcvt_##FROM##2##TO##_m(dst, src,               \
                                                       (ULL)part.value)        \
                : __builtin_cce_vcvt_##FROM##2##TO##_x(src, (ULL)part.value);  \
    }                                                                          \
  }
VCVT(u32, u16)
VCVT(u32, s16)
VCVT(s32, u16)
VCVT(s32, s16)
#undef VCVT

#define VCVT(FROM, TO, TYPE)                                                   \
  template <class T1>                                                          \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src, T1 part) {      \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  "The 3rd argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    if (isV300Target() || isV310Target()) {                                    \
      vector_bool mask = pset_##TYPE(PAT_ALL);                                 \
      dst =                                                                    \
          __builtin_cce_vcvtii_##FROM##2##TO##_x(src, mask, (ULL)part.value);  \
    } else {                                                                   \
      dst = __builtin_cce_vcvt_##FROM##2##TO(src, (ULL)part.value);            \
    }                                                                          \
  }
VCVT(u8, u16, b8)
VCVT(s8, s16, b8)
VCVT(u16, u32, b16)
VCVT(s16, u32, b16)
VCVT(s16, s32, b16)
#undef VCVT

#define VCVT(FROM, TO)                                                         \
  template <class T1>                                                          \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src, T1 pp) {        \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_same<T1, PartP0Type>::value ||                       \
                      std::is_same<T1, PartP1Type>::value ||                   \
                      std::is_same<T1, PartP2Type>::value ||                   \
                      std::is_same<T1, PartP3Type>::value,                     \
                  "The 3rd argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_P0, PART_P1, PART_P2, PART_P3");        \
    if (isV300Target() || isV310Target()) {                                    \
      vector_bool mask = pset_b8(PAT_ALL);                                    \
      dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(src, mask, (ULL)pp.value);  \
    } else {                                                                   \
      dst = __builtin_cce_vcvt_##FROM##2##TO(src, (ULL)pp.value);              \
    }                                                                          \
  }
VCVT(u8, u32)
VCVT(s8, s32)
#undef VCVT

//----------------------------------------------------------------------------//
// For compatibling with v300 VCVT, using pset + pslide create mask;
// The magic number 1, 2, 3 are representing right shift number.
//----------------------------------------------------------------------------//
#define VCVT(FROM, TO)                                                         \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src, T1 pp,          \
                          T2 mode = MODE_UNKNOWN) {                            \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_same<T1, PartP0Type>::value ||                       \
                      std::is_same<T1, PartP1Type>::value ||                   \
                      std::is_same<T1, PartP2Type>::value ||                   \
                      std::is_same<T1, PartP3Type>::value,                     \
                  "The 3rd argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_P0, PART_P1, PART_P2, PART_P3");        \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The last argument of this vcvt (" #FROM "2" #TO ") can only be: "     \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask1 = pset_b32(PAT_ALL);                                    \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask;                                                      \
        if (pp.value == Part_T::P0) {                                          \
          mask = pset_b8(PAT_M4);                                              \
        } else if (pp.value == Part_T::P1) {                                   \
          vector_bool maskM4Src0 = pset_b8(PAT_M4);                            \
          vector_bool maskM4Src1 = pset_b8(PAT_M4);                            \
          mask = __builtin_cce_pslide_b8(maskM4Src0, maskM4Src1, 3);           \
        } else if (pp.value == Part_T::P2) {                                   \
          vector_bool maskM4Src0 = pset_b8(PAT_M4);                            \
          vector_bool maskM4Src1 = pset_b8(PAT_M4);                            \
          mask = __builtin_cce_pslide_b8(maskM4Src0, maskM4Src1, 2);           \
        } else if (pp.value == Part_T::P3) {                                   \
          vector_bool maskM4Src0 = pset_b8(PAT_M4);                            \
          vector_bool maskM4Src1 = pset_b8(PAT_M4);                            \
          mask = __builtin_cce_pslide_b8(maskM4Src0, maskM4Src1, 1);           \
        }                                                                      \
        vector_##TO dstTmp = __builtin_cce_vcvtii_##FROM##2##TO##_x(           \
            src, mask1, (ULL)RoundingSaturation::RS_DISABLE_VALUE,             \
            (ULL)pp.value);                                                    \
        dst = __builtin_cce_vmov_v256##TO##_m(dst, dstTmp, mask);              \
      } else {                                                                 \
        dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(                          \
            src, mask1, (ULL)RoundingSaturation::RS_DISABLE_VALUE,             \
            (ULL)pp.value);                                                    \
      }                                                                        \
    } else {                                                                   \
      dst =                                                                    \
          (mode.value == MODE_MERGING.value)                                   \
              ? __builtin_cce_vcvt_##FROM##2##TO##_m(dst, src, (ULL)pp.value)  \
              : __builtin_cce_vcvt_##FROM##2##TO##_x(src, (ULL)pp.value);      \
    }                                                                          \
  }
VCVT(u32, u8)
VCVT(s32, u8)
#undef VCVT

//----------------------------------------------------------------------------//
//  vscvt
//----------------------------------------------------------------------------//
#define VSCVT(FROM, TO)                                                        \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vscvt(vector_##TO &dst, vector_##FROM src, T1 part,       \
                           T2 mode = MODE_UNKNOWN) {                           \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  "The 3rd argument of this vscvt (" #FROM "2" #TO             \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The last argument of this vscvt (" #FROM "2" #TO ") can only be: "    \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask1 = pset_b8(PAT_ALL);                                    \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask = pset_b8(PAT_H);                                     \
        mask = __builtin_cce_punpack(mask, (ULL)LOWER.value);                  \
        if (part.value == Part::ODD) {                                         \
          mask = __builtin_cce_pnot_z(mask, mask1);                            \
        }                                                                      \
        vector_##TO dstTmp = __builtin_cce_vcvtii_##FROM##2##TO##_x(           \
            src, mask1, (ULL)RoundingSaturation::RS_ENABLE_VALUE,              \
            (ULL)part.value);                                                  \
        dst = __builtin_cce_vmov_v256##TO##_m(dst, dstTmp, mask);              \
      } else {                                                                 \
        dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(                          \
            src, mask1, (ULL)RoundingSaturation::RS_ENABLE_VALUE,              \
            (ULL)part.value);                                                  \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vscvt_##FROM##2##TO##_m(dst, src,              \
                                                        (ULL)part.value)       \
                : __builtin_cce_vscvt_##FROM##2##TO##_x(src, (ULL)part.value); \
    }                                                                          \
  }
VSCVT(u16, u8)
VSCVT(s16, u8)
#undef VSCVT

#define VSCVT(FROM, TO)                                                        \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vscvt(vector_##TO &dst, vector_##FROM src, T1 part,       \
                           T2 mode = MODE_UNKNOWN) {                           \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  "The 3rd argument of this vscvt (" #FROM "2" #TO             \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The last argument of this vscvt (" #FROM "2" #TO ") can only be: "    \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask1 = pset_b8(PAT_ALL);                                    \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask = pset_b8(PAT_M4);                                    \
        if (part.value == Part::ODD) {                                         \
          mask = __builtin_cce_pnot_z(mask, mask1);                            \
        }                                                                      \
        vector_##TO dstTmp = __builtin_cce_vcvtii_##FROM##2##TO##_x(           \
            src, mask1, (ULL)RoundingSaturation::RS_ENABLE_VALUE,              \
            (ULL)part.value);                                                  \
        dst = __builtin_cce_vmov_v128##TO##_m(dst, dstTmp, mask);              \
      } else {                                                                 \
        dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(                          \
            src, mask1, (ULL)RoundingSaturation::RS_ENABLE_VALUE,              \
            (ULL)part.value);                                                  \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vscvt_##FROM##2##TO##_m(dst, src,              \
                                                        (ULL)part.value)       \
                : __builtin_cce_vscvt_##FROM##2##TO##_x(src, (ULL)part.value); \
    }                                                                          \
  }
VSCVT(u32, u16)
VSCVT(u32, s16)
VSCVT(s32, u16)
VSCVT(s32, s16)
#undef VSCVT

//----------------------------------------------------------------------------//
// For compatibling with v300 VSCVT, using pset + pslide create mask;
// The magic number 1, 2, 3 are representing right shift number.
//----------------------------------------------------------------------------//
#define VSCVT(FROM, TO)                                                        \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vscvt(vector_##TO &dst, vector_##FROM src, T1 pp,         \
                           T2 mode = MODE_UNKNOWN) {                           \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_same<T1, PartP0Type>::value ||                       \
                      std::is_same<T1, PartP1Type>::value ||                   \
                      std::is_same<T1, PartP2Type>::value ||                   \
                      std::is_same<T1, PartP3Type>::value,                     \
                  "The 3rd argument of this vscvt (" #FROM "2" #TO             \
                  ") can only be: PART_P0, PART_P1, PART_P2, PART_P3");        \
    static_assert(                                                             \
        mode.value == MODE_UNKNOWN.value || mode.value == MODE_MERGING.value,  \
        "The last argument of this vscvt (" #FROM "2" #TO ") can only be: "    \
        "MODE_UNKNOWN, MODE_MERGING");                                         \
    if (isSoftwareMergeMode()) {                                               \
      vector_bool mask1 = pset_b32(PAT_ALL);                                    \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask;                                                      \
        if (pp.value == Part_T::P0) {                                          \
          mask = pset_b8(PAT_M4);                                              \
        } else if (pp.value == Part_T::P1) {                                   \
          vector_bool maskM4Src0 = pset_b8(PAT_M4);                            \
          vector_bool maskM4Src1 = pset_b8(PAT_M4);                            \
          mask = __builtin_cce_pslide_b8(maskM4Src0, maskM4Src1, 3);           \
        } else if (pp.value == Part_T::P2) {                                   \
          vector_bool maskM4Src0 = pset_b8(PAT_M4);                            \
          vector_bool maskM4Src1 = pset_b8(PAT_M4);                            \
          mask = __builtin_cce_pslide_b8(maskM4Src0, maskM4Src1, 2);           \
        } else if (pp.value == Part_T::P3) {                                   \
          vector_bool maskM4Src0 = pset_b8(PAT_M4);                            \
          vector_bool maskM4Src1 = pset_b8(PAT_M4);                            \
          mask = __builtin_cce_pslide_b8(maskM4Src0, maskM4Src1, 1);           \
        }                                                                      \
        vector_##TO dstTmp = __builtin_cce_vcvtii_##FROM##2##TO##_x(           \
            src, mask1, (ULL)RoundingSaturation::RS_ENABLE_VALUE,              \
            (ULL)pp.value);                                                    \
        dst = __builtin_cce_vmov_v256##TO##_m(dst, dstTmp, mask);              \
      } else {                                                                 \
        dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(                          \
            src, mask1, (ULL)RoundingSaturation::RS_ENABLE_VALUE,              \
            (ULL)pp.value);                                                    \
      }                                                                        \
    } else {                                                                   \
      dst =                                                                    \
          (mode.value == MODE_MERGING.value)                                   \
              ? __builtin_cce_vscvt_##FROM##2##TO##_m(dst, src, (ULL)pp.value) \
              : __builtin_cce_vscvt_##FROM##2##TO##_x(src, (ULL)pp.value);     \
    }                                                                          \
  }
VSCVT(u32, u8)
VSCVT(s32, u8)
#undef VSCVT

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
//----------------------------------------------------------------------------//
//  VCVTFI
//----------------------------------------------------------------------------//
#define VCVTFI_SAT_PART(FROM, TO)                                              \
  template <class T1, class T2, class T3,                                      \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src,                 \
                          vector_bool mask, T1 rnd, T2 sat, T3 part,           \
                          Tm mode = MODE_ZEROING) {                            \
    static_assert(std::is_class<T1>::value, "the 4th argument is invalid");    \
    static_assert(std::is_class<T2>::value, "the 5th argument is invalid");    \
    static_assert(std::is_class<T3>::value, "the 6th argument is invalid");    \
    static_assert(std::is_class<Tm>::value, "the last argument is invalid");   \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, RoundAType>::value ||                   \
                      std::is_same<T1, RoundFType>::value ||                   \
                      std::is_same<T1, RoundCType>::value ||                   \
                      std::is_same<T1, RoundZType>::value,                     \
                  "The 4th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  "only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");     \
    static_assert(std::is_same<T2, RSDisableType>::value ||                    \
                      std::is_same<T2, RSEnableType>::value,                   \
                  "The 5th argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: RS_DISABLE, RS_ENABLE");                     \
    static_assert(std::is_same<T3, PartEvenType>::value ||                     \
                      std::is_same<T3, PartOddType>::value,                    \
                  "The 6th argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtfi_##FROM##2##TO##_x(                              \
        src, mask, (ULL)rnd.value, (ULL)sat.value, (ULL)part.value);           \
    return;                                                                    \
  }
VCVTFI_SAT_PART(f32, s16)
VCVTFI_SAT_PART(f16, u8)
VCVTFI_SAT_PART(f16, s8)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_C310__)
VCVTFI_SAT_PART(f32, s64)
VCVTFI_SAT_PART(bf16, s32)
#endif
#undef VCVTFI_SAT_PART

#define VCVTFI_SAT(FROM, TO)                                                   \
  template <class T1, class T2,                                                \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src,                 \
                          vector_bool mask, T1 rnd, T2 sat,                    \
                          Tm mode = MODE_ZEROING) {                            \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 5th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, RoundAType>::value ||                   \
                      std::is_same<T1, RoundFType>::value ||                   \
                      std::is_same<T1, RoundCType>::value ||                   \
                      std::is_same<T1, RoundZType>::value,                     \
                  "The 4th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  "only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");     \
    static_assert(std::is_same<T2, RSDisableType>::value ||                    \
                      std::is_same<T2, RSEnableType>::value,                   \
                  "The 5th argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: RS_DISABLE, RS_ENABLE");                     \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtfi_##FROM##2##TO##_x(src, mask, (ULL)rnd.value,    \
                                                 (ULL)sat.value);              \
    return;                                                                    \
  }
VCVTFI_SAT(f32, s32)
VCVTFI_SAT(f16, s16)
#undef VCVTFI_SAT

#define VCVTFI_PART(FROM, TO)                                                  \
  template <class T1, class T2,                                                \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src,                 \
                          vector_bool mask, T1 rnd, T2 part,                   \
                          Tm mode = MODE_ZEROING) {                            \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 5th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, RoundAType>::value ||                   \
                      std::is_same<T1, RoundFType>::value ||                   \
                      std::is_same<T1, RoundCType>::value ||                   \
                      std::is_same<T1, RoundZType>::value,                     \
                  "The 4th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  "only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");     \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  "The 5th argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtfi_##FROM##2##TO##_x(src, mask, (ULL)rnd.value,    \
                                                 (ULL)part.value);             \
    return;                                                                    \
  }
VCVTFI_PART(f16, s32)
#undef VCVTFI_PART

#define VCVTFI_SAT_PP(FROM, TO)                                                \
  template <class T1, class T2, class T3,                                      \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt_f162s4(vector_##TO &dst, vector_##FROM src,          \
                                 vector_bool mask, T1 rnd, T2 sat, T3 pp,      \
                                 Tm mode = MODE_ZEROING) {                     \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 5th argument is not valid");  \
    static_assert(std::is_class<T3>::value, "the 6th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, RoundAType>::value ||                   \
                      std::is_same<T1, RoundFType>::value ||                   \
                      std::is_same<T1, RoundCType>::value ||                   \
                      std::is_same<T1, RoundZType>::value,                     \
                  "The 4th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  " only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");    \
    static_assert(std::is_same<T2, RSDisableType>::value ||                    \
                      std::is_same<T2, RSEnableType>::value,                   \
                  "The 5th argument of this vcvt (" #FROM "2" #TO              \
                  " ) can only be: RS_DISABLE, RS_ENABLE");                    \
    static_assert(std::is_same<T3, PartP0Type>::value ||                       \
                      std::is_same<T3, PartP1Type>::value ||                   \
                      std::is_same<T3, PartP2Type>::value ||                   \
                      std::is_same<T3, PartP3Type>::value,                     \
                  "The 6th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  "only be: PART_P0, PART_P1, PART_P2, PART_P3");              \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtfi_##FROM##2##TO##_x(                              \
        src, mask, (ULL)rnd.value, (ULL)sat.value, (ULL)pp.value);             \
    return;                                                                    \
  }
VCVTFI_SAT_PP(f16, s4x2)
#undef VCVTFI_SAT_PP

//----------------------------------------------------------------------------//
//  VCVTFF
//----------------------------------------------------------------------------//
#define VCVTFF_RND_F322F16(TO)                                                 \
  template <class T1, class T2, class T3,                                      \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_f32 src, vector_bool mask,  \
                          T1 rnd, T2 sat, T3 part, Tm mode = MODE_ZEROING) {   \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 5th argument is not valid");  \
    static_assert(std::is_class<T3>::value, "the 6th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, RoundAType>::value ||                   \
                      std::is_same<T1, RoundFType>::value ||                   \
                      std::is_same<T1, RoundCType>::value ||                   \
                      std::is_same<T1, RoundZType>::value ||                   \
                      std::is_same<T1, RoundOType>::value,                     \
                  "The 4th argument of this vcvt (f322" #TO ") can only be: "  \
                  "ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z, ROUND_O");     \
    static_assert(std::is_same<T2, RSDisableType>::value ||                    \
                      std::is_same<T2, RSEnableType>::value,                   \
                  "The 5th argument of this vcvt (f322" #TO                    \
                  ") can only be: RS_DISABLE, RS_ENABLE");                     \
    static_assert(std::is_same<T3, PartEvenType>::value ||                     \
                      std::is_same<T3, PartOddType>::value,                    \
                  "The 6th argument of this vcvt (f322" #TO                    \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtff_f322##TO##_x(src, mask, (ULL)rnd.value,         \
                                            (ULL)sat.value, (ULL)part.value);  \
    return;                                                                    \
  }
VCVTFF_RND_F322F16(f16)
#undef VCVTFF_RND_F322F16

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
#define VCVTFF_RND(TO)                                                         \
  template <class T1, class T2, class T3,                                      \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_f32 src, vector_bool mask,  \
                          T1 rnd, T2 sat, T3 part, Tm mode = MODE_ZEROING) {   \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 5th argument is not valid");  \
    static_assert(std::is_class<T3>::value, "the 6th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, RoundAType>::value ||                   \
                      std::is_same<T1, RoundFType>::value ||                   \
                      std::is_same<T1, RoundCType>::value ||                   \
                      std::is_same<T1, RoundZType>::value,                     \
                  "The 4th argument of this vcvt (f322" #TO ") can only be: "  \
                  "ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");              \
    static_assert(std::is_same<T2, RSDisableType>::value ||                    \
                      std::is_same<T2, RSEnableType>::value,                   \
                  "The 5th argument of this vcvt (f322" #TO                    \
                  ") can only be: RS_DISABLE, RS_ENABLE");                     \
    static_assert(std::is_same<T3, PartEvenType>::value ||                     \
                      std::is_same<T3, PartOddType>::value,                    \
                  "The 6th argument of this vcvt (f322" #TO                    \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtff_f322##TO##_x(src, mask, (ULL)rnd.value,         \
                                            (ULL)sat.value, (ULL)part.value);  \
    return;                                                                    \
  }
VCVTFF_RND(bf16)
#undef VCVTFF_RND
#endif

#define VCVTFF(FROM)                                                           \
  template <class T1,                                                          \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_f32 &dst, vector_##FROM src,                  \
                          vector_bool mask, T1 part, Tm mode = MODE_ZEROING) { \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  "The 4th argument of this vcvt (" #FROM "2f32"               \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtff_##FROM##2f32_x(src, mask, (ULL)part.value);     \
  }
VCVTFF(f16)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VCVTFF(bf16)
#endif
#undef VCVTFF

//----------------------------------------------------------------------------//
//  VCVTIF
//----------------------------------------------------------------------------//
#define VCVTIF_PART(FROM, TO)                                                  \
  template <class T1,                                                          \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src,                 \
                          vector_bool mask, T1 part, Tm mode = MODE_ZEROING) { \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  "The 4th argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtif_##FROM##2##TO##_x(src, mask, (ULL)part.value);  \
    return;                                                                    \
  }
VCVTIF_PART(u8, f16)
VCVTIF_PART(s8, f16)
VCVTIF_PART(s16, f32)
#undef VCVTIF_PART

#define VCVTIF_RND(FROM, TO)                                                   \
  template <class T1,                                                          \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src,                 \
                          vector_bool mask, T1 rnd, Tm mode = MODE_ZEROING) {  \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, RoundAType>::value ||                   \
                      std::is_same<T1, RoundFType>::value ||                   \
                      std::is_same<T1, RoundCType>::value ||                   \
                      std::is_same<T1, RoundZType>::value,                     \
                  "The 4th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  " only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");    \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtif_##FROM##2##TO##_x(src, mask, (ULL)rnd.value);   \
    return;                                                                    \
  }
VCVTIF_RND(s16, f16)
VCVTIF_RND(s32, f32)
#undef VCVTIF_RND

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_C310__)
#define VCVTIF_RND_PART(FROM, TO)                                              \
  template <class T1, class T2,                                                \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src,                 \
                          vector_bool mask, T1 rnd, T2 part,                   \
                          Tm mode = MODE_ZEROING) {                            \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 5th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, RoundAType>::value ||                   \
                      std::is_same<T1, RoundFType>::value ||                   \
                      std::is_same<T1, RoundCType>::value ||                   \
                      std::is_same<T1, RoundZType>::value,                     \
                  "The 4th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  " only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");    \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  "The 5th argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtif_##FROM##2##TO##_x(src, mask, (ULL)rnd.value,    \
                                                 (ULL)part.value);             \
    return;                                                                    \
  }
VCVTIF_RND_PART(s64, f32)
#undef VCVTIF_RND_PART
#endif

#define VCVTIF_PP(FROM, TO)                                                    \
  template <class T1,                                                          \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt_s42f16(vector_##TO &dst, vector_##FROM src,          \
                                 vector_bool mask, T1 pp,                      \
                                 Tm mode = MODE_ZEROING) {                     \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartP0Type>::value ||                       \
                      std::is_same<T1, PartP1Type>::value ||                   \
                      std::is_same<T1, PartP2Type>::value ||                   \
                      std::is_same<T1, PartP3Type>::value,                     \
                  "The 4th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  "only be: PART_P0, PART_P1, PART_P2, PART_P3");              \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtif_##FROM##2##TO##_x(src, mask, (ULL)pp.value);    \
    return;                                                                    \
  }
VCVTIF_PP(s4x2, f16)
#undef VCVTIF_PP

//----------------------------------------------------------------------------//
//  VCVTII
//----------------------------------------------------------------------------//
#define VCVTII_PART(FROM, TO)                                                  \
  template <class T1,                                                          \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src,                 \
                          vector_bool mask, T1 part, Tm mode = MODE_ZEROING) { \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  "The 4th argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(src, mask, (ULL)part.value);  \
    return;                                                                    \
  }
VCVTII_PART(u8, u16)
VCVTII_PART(s8, s16)
VCVTII_PART(u16, u32)
VCVTII_PART(s16, u32)
VCVTII_PART(s16, s32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_C310__) ||                      \
    defined(__DAV_L310__)
VCVTII_PART(s32, s64)
#endif
#undef VCVTII_PART

#define VCVTII_SAT_PART(FROM, TO)                                              \
  template <class T1, class T2,                                                \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src,                 \
                          vector_bool mask, T1 sat, T2 part,                   \
                          Tm mode = MODE_ZEROING) {                            \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 5th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RSDisableType>::value ||                    \
                      std::is_same<T1, RSEnableType>::value,                   \
                  "The 4th argument of this vcvt (" #FROM "2" #TO              \
                  " ) can only be: RS_DISABLE, RS_ENABLE");                    \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  "The 5th argument of this vcvt (" #FROM "2" #TO              \
                  ") can only be: PART_EVEN, PART_ODD");                       \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(src, mask, (ULL)sat.value,    \
                                                 (ULL)part.value);             \
    return;                                                                    \
  }
VCVTII_SAT_PART(u16, u8)
VCVTII_SAT_PART(s16, u8)
VCVTII_SAT_PART(u32, u16)
VCVTII_SAT_PART(u32, s16)
VCVTII_SAT_PART(s32, u16)
VCVTII_SAT_PART(s32, s16)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_C310__) ||                      \
    defined(__DAV_L310__)
VCVTII_SAT_PART(s64, s32)
#endif
#undef VCVTII_SAT_PART

#define VCVTII_PP(FROM, TO)                                                    \
  template <class T1,                                                          \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src,                 \
                          vector_bool mask, T1 pp, Tm mode = MODE_ZEROING) {   \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartP0Type>::value ||                       \
                      std::is_same<T1, PartP1Type>::value ||                   \
                      std::is_same<T1, PartP2Type>::value ||                   \
                      std::is_same<T1, PartP3Type>::value,                     \
                  "The 4th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  "only be: PART_P0, PART_P1, PART_P2, PART_P3");              \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(src, mask, (ULL)pp.value);    \
    return;                                                                    \
  }
VCVTII_PP(u8, u32)
VCVTII_PP(s8, s32)
#undef VCVTII_PP

#define VCVTII_SAT_PP(FROM, TO)                                                \
  template <class T1, class T2,                                                \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_##FROM src,                 \
                          vector_bool mask, T1 sat, T2 pp,                     \
                          Tm mode = MODE_ZEROING) {                            \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 5th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RSDisableType>::value ||                    \
                      std::is_same<T1, RSEnableType>::value,                   \
                  "The 4th argument of this vcvt (" #FROM "2" #TO              \
                  " ) can only be: RS_DISABLE, RS_ENABLE");                    \
    static_assert(std::is_same<T2, PartP0Type>::value ||                       \
                      std::is_same<T2, PartP1Type>::value ||                   \
                      std::is_same<T2, PartP2Type>::value ||                   \
                      std::is_same<T2, PartP3Type>::value,                     \
                  "The 5th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  "only be: PART_P0, PART_P1, PART_P2, PART_P3");              \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(src, mask, (ULL)sat.value,    \
                                                 (ULL)pp.value);               \
    return;                                                                    \
  }
VCVTII_SAT_PP(u32, u8)
VCVTII_SAT_PP(s32, u8)
#undef VCVTII_SAT_PP

#define VCVTII_PP_S4(FROM, TO)                                                 \
  template <class T1,                                                          \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt_s42s16(vector_##TO &dst, vector_##FROM src,          \
                                 vector_bool mask, T1 pp,                      \
                                 Tm mode = MODE_ZEROING) {                     \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartP0Type>::value ||                       \
                      std::is_same<T1, PartP1Type>::value ||                   \
                      std::is_same<T1, PartP2Type>::value ||                   \
                      std::is_same<T1, PartP3Type>::value,                     \
                  "The 4th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  "only be: PART_P0, PART_P1, PART_P2, PART_P3");              \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(src, mask, (ULL)pp.value);    \
    return;                                                                    \
  }
#if defined(__DAV_M310__) || defined(__DAV_L310__)
VCVTII_PP_S4(s4x2, s16)
#endif
#undef VCVTII_PP_S4

#define VCVTII_SAT_PP_S4(FROM, TO)                                             \
  template <class T1, class T2,                                                \
            class Tm = std::integral_constant<Mode, Mode::ZEROING_VALUE>>      \
  CCE_INTRINSIC void vcvt_s162s4(vector_##TO &dst, vector_##FROM src,          \
                                 vector_bool mask, T1 sat, T2 pp,              \
                                 Tm mode = MODE_ZEROING) {                     \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 5th argument is not valid");  \
    static_assert(std::is_class<Tm>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, RSDisableType>::value ||                    \
                      std::is_same<T1, RSEnableType>::value,                   \
                  "The 4th argument of this vcvt (" #FROM "2" #TO              \
                  " ) can only be: RS_DISABLE, RS_ENABLE");                    \
    static_assert(std::is_same<T2, PartP0Type>::value ||                       \
                      std::is_same<T2, PartP1Type>::value ||                   \
                      std::is_same<T2, PartP2Type>::value ||                   \
                      std::is_same<T2, PartP3Type>::value,                     \
                  "The 5th argument of this vcvt (" #FROM "2" #TO ") can "     \
                  "only be: PART_P0, PART_P1, PART_P2, PART_P3");              \
    static_assert(mode.value == MODE_ZEROING.value,                            \
                  "The last argument can only be 'MODE_ZEROING' or empty.");   \
    dst = __builtin_cce_vcvtii_##FROM##2##TO##_x(src, mask, (ULL)sat.value,    \
                                                 (ULL)pp.value);               \
    return;                                                                    \
  }
#if defined(__DAV_M310__) || defined(__DAV_L310__)
VCVTII_SAT_PP_S4(s16, s4x2)
#endif
#undef VCVTII_SAT_PP_S4

#endif

//----------------------------------------------------------------------------//
//  vtrc
//----------------------------------------------------------------------------//
#define VTRC(T, NUM)                                                           \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vtrc(vector_##T &dst, vector_##T src, T1 rnd,             \
                          vector_bool mask, T2 mode = MODE_UNKNOWN) {          \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 4th argument is not valid");  \
    static_assert(std::is_same<T1, RoundRType>::value ||                       \
                      std::is_same<T1, RoundAType>::value ||                   \
                      std::is_same<T1, RoundFType>::value ||                   \
                      std::is_same<T1, RoundCType>::value ||                   \
                      std::is_same<T1, RoundZType>::value,                     \
                  "The 3rd argument of vtrc can only be: ROUND_R, ROUND_A, "   \
                  "ROUND_F, ROUND_C, ROUND_Z");                                \
    static_assert(mode.value == MODE_ZEROING.value ||                          \
                      mode.value == MODE_UNKNOWN.value ||                      \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      vector_##T dstTmp =                                                      \
          __builtin_cce_vtrc_##T##_x(src, (ULL)rnd.value, mask);               \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vmov_##NUM##T##_m(dst, dstTmp, mask)           \
                : dstTmp;                                                      \
    } else {                                                                   \
      dst = (mode.value == MODE_UNKNOWN.value)                                 \
                ? __builtin_cce_vtrc_##T##_x(src, (ULL)rnd.value, mask)        \
                : __builtin_cce_vtrc_##T##_m(dst, src, (ULL)rnd.value, mask);  \
    }                                                                          \
    return;                                                                    \
  }
VTRC(f16, v128)
VTRC(f32, v64)
#undef VTRC

//----------------------------------------------------------------------------//
//  vcmp
//----------------------------------------------------------------------------//
#define VCMP(OP, T)                                                            \
  CCE_INTRINSIC void vcmp_##OP(vector_bool &dst, vector_##T src1,              \
                               vector_##T src2, vector_bool mask) {            \
    dst = __builtin_cce_vcmp_##OP##_##T##_z(src1, src2, mask);                 \
    return;                                                                    \
  }
VCMP(eq, u8)
VCMP(ne, u8)
VCMP(gt, u8)
VCMP(ge, u8)
VCMP(lt, u8)
VCMP(le, u8)

VCMP(eq, s8)
VCMP(ne, s8)
VCMP(gt, s8)
VCMP(ge, s8)
VCMP(lt, s8)
VCMP(le, s8)

VCMP(eq, u16)
VCMP(ne, u16)
VCMP(gt, u16)
VCMP(ge, u16)
VCMP(lt, u16)
VCMP(le, u16)

VCMP(eq, s16)
VCMP(ne, s16)
VCMP(gt, s16)
VCMP(ge, s16)
VCMP(lt, s16)
VCMP(le, s16)

VCMP(eq, u32)
VCMP(ne, u32)
VCMP(gt, u32)
VCMP(ge, u32)
VCMP(lt, u32)
VCMP(le, u32)

VCMP(eq, s32)
VCMP(ne, s32)
VCMP(gt, s32)
VCMP(ge, s32)
VCMP(lt, s32)
VCMP(le, s32)

VCMP(eq, f16)
VCMP(ne, f16)
VCMP(gt, f16)
VCMP(ge, f16)
VCMP(lt, f16)
VCMP(le, f16)

VCMP(eq, f32)
VCMP(ne, f32)
VCMP(gt, f32)
VCMP(ge, f32)
VCMP(lt, f32)
VCMP(le, f32)
#undef VCMP

//----------------------------------------------------------------------------//
//  vcmps
//----------------------------------------------------------------------------//
#define VCMPS(OP, ST, LT)                                                      \
  CCE_INTRINSIC void vcmps_##OP(vector_bool &dst, vector_##ST src1, LT src2,   \
                                vector_bool mask) {                            \
    dst = __builtin_cce_vcmps_##OP##_##ST##_z(src1, src2, mask);               \
    return;                                                                    \
  }

VCMPS(eq, u8, uint8_t)
VCMPS(ne, u8, uint8_t)
VCMPS(gt, u8, uint8_t)
VCMPS(ge, u8, uint8_t)
VCMPS(lt, u8, uint8_t)
VCMPS(le, u8, uint8_t)

VCMPS(eq, s8, int8_t)
VCMPS(ne, s8, int8_t)
VCMPS(gt, s8, int8_t)
VCMPS(ge, s8, int8_t)
VCMPS(lt, s8, int8_t)
VCMPS(le, s8, int8_t)

VCMPS(eq, u16, uint16_t)
VCMPS(ne, u16, uint16_t)
VCMPS(gt, u16, uint16_t)
VCMPS(ge, u16, uint16_t)
VCMPS(lt, u16, uint16_t)
VCMPS(le, u16, uint16_t)

VCMPS(eq, s16, int16_t)
VCMPS(ne, s16, int16_t)
VCMPS(gt, s16, int16_t)
VCMPS(ge, s16, int16_t)
VCMPS(lt, s16, int16_t)
VCMPS(le, s16, int16_t)

VCMPS(eq, u32, uint32_t)
VCMPS(ne, u32, uint32_t)
VCMPS(gt, u32, uint32_t)
VCMPS(ge, u32, uint32_t)
VCMPS(lt, u32, uint32_t)
VCMPS(le, u32, uint32_t)

VCMPS(eq, s32, int32_t)
VCMPS(ne, s32, int32_t)
VCMPS(gt, s32, int32_t)
VCMPS(ge, s32, int32_t)
VCMPS(lt, s32, int32_t)
VCMPS(le, s32, int32_t)

VCMPS(eq, f16, half)
VCMPS(ne, f16, half)
VCMPS(gt, f16, half)
VCMPS(ge, f16, half)
VCMPS(lt, f16, half)
VCMPS(le, f16, half)

VCMPS(eq, f32, float)
VCMPS(ne, f32, float)
VCMPS(gt, f32, float)
VCMPS(ge, f32, float)
VCMPS(lt, f32, float)
VCMPS(le, f32, float)

//----------------------------------------------------------------------------//
//  vlda
//----------------------------------------------------------------------------//
#define VLDA(LT, ST)                                                           \
  CCE_INTRINSIC void vlda(vector_align &dst, __ubuf__ LT *base,                \
                          vector_address offset) {                             \
    dst = __builtin_cce_vlda_##ST(base, offset, 0 /* #loop */);                \
  }

VLDA(int8_t, s8)
VLDA(uint8_t, u8)
VLDA(int16_t, s16)
VLDA(uint16_t, u16)
VLDA(int32_t, s32)
VLDA(uint32_t, u32)
VLDA(half, f16)
VLDA(float, f32)
VLDA(int64_t, s64)
#if defined(__DAV_L300__) || defined(__DAV_L300_VEC__) ||                      \
    defined(__DAV_M300__) || defined(__DAV_T300__) || defined(__DAV_C310__)
VLDA(bfloat16_t, bf16)
#endif
#undef VLDA

//----------------------------------------------------------------------------//
//  vldas
//----------------------------------------------------------------------------//
#define VLDAS(LT, ST)                                                          \
  CCE_INTRINSIC void vldas(vector_align &dst, __ubuf__ LT *base) {             \
    dst = __builtin_cce_vldas_##ST(base);                                      \
  }

VLDAS(int8_t, s8)
VLDAS(uint8_t, u8)
VLDAS(int16_t, s16)
VLDAS(uint16_t, u16)
VLDAS(int32_t, s32)
VLDAS(uint32_t, u32)
VLDAS(half, f16)
VLDAS(float, f32)
VLDAS(int64_t, s64)
#if defined(__DAV_L300__) || defined(__DAV_L300_VEC__) ||                      \
    defined(__DAV_M300__) || defined(__DAV_T300__) || defined(__DAV_C310__)
VLDAS(bfloat16_t, bf16)
#endif
#undef VLDAS

//----------------------------------------------------------------------------//
//  vldu
//----------------------------------------------------------------------------//
#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
#define VLDU(LT, ST)                                                           \
  CCE_INTRINSIC void vldu(vector_##ST &dst, vector_align &alignData,           \
                          vector_address &offset, __ubuf__ LT *base,           \
                          uint16_t inc /* in unit of element */) {             \
    struct {                                                                   \
      vector_##ST vecData;                                                     \
      vector_align_data alignData;                                             \
      vector_address offset;                                                   \
    } ret;                                                                     \
    __builtin_cce_vldu_##ST(&ret, base, offset, alignData, inc * sizeof(LT),   \
                            0 /* #loop */);                                    \
    dst = ret.vecData;                                                         \
    alignData = ret.alignData;                                                 \
    offset = ret.offset;                                                       \
    return;                                                                    \
  }

VLDU(int8_t, s8)
VLDU(uint8_t, u8)
VLDU(int16_t, s16)
VLDU(uint16_t, u16)
VLDU(int32_t, s32)
VLDU(uint32_t, u32)
VLDU(half, f16)
VLDU(float, f32)
#undef VLDU
#elif defined(__DAV_M300__) || defined(__DAV_L300__) ||                        \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VLDU(LT, ST)                                                           \
  CCE_INTRINSIC void vldu(vector_##ST &dst, vector_align &alignData,           \
                          vector_address &offset, __ubuf__ LT *base,           \
                          uint32_t inc /* in unit of element */) {             \
    struct {                                                                   \
      vector_##ST vecData;                                                     \
      vector_align_data alignData;                                             \
      vector_address offset;                                                   \
    } ret;                                                                     \
    __builtin_cce_vldu_v300_##ST(&ret, base, offset, alignData,                \
                                 inc * sizeof(LT), 0 /* #loop */);             \
    dst = ret.vecData;                                                         \
    alignData = ret.alignData;                                                 \
    offset = ret.offset;                                                       \
    return;                                                                    \
  }
VLDU(int8_t, s8)
VLDU(uint8_t, u8)
VLDU(int16_t, s16)
VLDU(uint16_t, u16)
VLDU(int32_t, s32)
VLDU(uint32_t, u32)
VLDU(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VLDU(bfloat16_t, bf16)
#endif
VLDU(half, f16)
VLDU(float, f32)
#undef VLDU
#endif

//----------------------------------------------------------------------------//
//  vldus
//  - the value of 'base' is added up by the acutal base pointer and 'inc'.
//  - the no post-update interface is just implemented on software level.
//  the return value of no post-update interface is a struct same with that
//  of po-update interface. however, 'ret.base' is not actually useful. the
//  reason to keep it is to simplify the IR replacement later work.
//----------------------------------------------------------------------------//
#define VLDUS(LT, ST)                                                          \
  CCE_INTRINSIC void vldus(vector_##ST &dst, vector_align &alignData,          \
                           __ubuf__ LT *base) {                                \
    struct {                                                                   \
      vector_##ST vecData;                                                     \
      vector_align_data alignData;                                             \
      __ubuf__ LT *base;                                                       \
    } ret;                                                                     \
    __builtin_cce_vldus_##ST(&ret, base, alignData);                           \
    dst = ret.vecData;                                                         \
    alignData = ret.alignData;                                                 \
    base = ret.base;                                                           \
    return;                                                                    \
  }                                                                            \
  template <class T>                                                           \
  CCE_INTRINSIC void vldus(vector_##ST &dst, vector_align &alignData,          \
                           __ubuf__ LT *&base,                                 \
                           uint32_t inc /* in unit of element */, T post) {    \
    static_assert(std::is_same<T, PostUpdateType>::value,                      \
                  "The last argument can only be 'POST_UPDATE'.");             \
    struct {                                                                   \
      vector_##ST vecData;                                                     \
      vector_align_data alignData;                                             \
      __ubuf__ LT *base;                                                       \
    } ret;                                                                     \
    __builtin_cce_vldus_post_##ST(&ret, base, alignData, inc * sizeof(LT));    \
    dst = ret.vecData;                                                         \
    alignData = ret.alignData;                                                 \
    base = ret.base;                                                           \
    return;                                                                    \
  }

VLDUS(int8_t, s8)
VLDUS(uint8_t, u8)
VLDUS(int16_t, s16)
VLDUS(uint16_t, u16)
VLDUS(int32_t, s32)
VLDUS(uint32_t, u32)
VLDUS(half, f16)
VLDUS(float, f32)
VLDUS(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VLDUS(bfloat16_t, bf16)
#endif

#undef VLDUS

//----------------------------------------------------------------------------//
//  vldui
//  - the value of 'base' is added up by the acutal base pointer and 'inc'.
//  - the no post-update interface is just implemented on software level.
//  the return value of no post-update interface is a struct same with that
//  of po-update interface. however, 'ret.base' is not actually useful. the
//  reason to keep it is to simplify the IR replacement work later.
//----------------------------------------------------------------------------//
#define VLDUI(LT, ST)                                                          \
  DEPRECATED CCE_INTRINSIC void vldui(                                         \
      vector_##ST &dst, vector_align &alignData, __ubuf__ LT *base) {          \
    struct {                                                                   \
      vector_##ST vecData;                                                     \
      vector_align_data alignData;                                             \
      __ubuf__ LT *base;                                                       \
    } ret;                                                                     \
    __builtin_cce_vldui_##ST(&ret, base, alignData);                           \
    dst = ret.vecData;                                                         \
    alignData = ret.alignData;                                                 \
    base = ret.base;                                                           \
    return;                                                                    \
  }                                                                            \
  template <class T>                                                           \
  DEPRECATED CCE_INTRINSIC void vldui(                                         \
      vector_##ST &dst, vector_align &alignData, __ubuf__ LT *&base,           \
      uint16_t inc /* in unit of element */, T post) {                         \
    static_assert(std::is_same<T, PostUpdateType>::value,                      \
                  "The last argument can only be 'POST_UPDATE'.");             \
    struct {                                                                   \
      vector_##ST vecData;                                                     \
      vector_align_data alignData;                                             \
      __ubuf__ LT *base;                                                       \
    } ret;                                                                     \
    __builtin_cce_vldui_post_##ST(&ret, base, alignData, inc * sizeof(LT));    \
    dst = ret.vecData;                                                         \
    alignData = ret.alignData;                                                 \
    base = ret.base;                                                           \
    return;                                                                    \
  }

VLDUI(int8_t, s8)
VLDUI(uint8_t, u8)
VLDUI(int16_t, s16)
VLDUI(uint16_t, u16)
VLDUI(int32_t, s32)
VLDUI(uint32_t, u32)
VLDUI(half, f16)
VLDUI(float, f32)
VLDUI(int64_t, s64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VLDUI(bfloat16_t, bf16)
#endif

#undef VLDUI

//----------------------------------------------------------------------------//
//  vsel
//----------------------------------------------------------------------------//
#define VSEL(TYPE, NUM)                                                        \
  CCE_INTRINSIC void vsel(vector_##TYPE &dst, vector_##TYPE src0,              \
                          vector_##TYPE src1, vector_bool mask) {              \
    dst = __builtin_cce_vsel_##NUM##TYPE(src0, src1, mask);                    \
    return;                                                                    \
  }

VSEL(u8, v256)
VSEL(s8, v256)
VSEL(u16, v128)
VSEL(s16, v128)
VSEL(f16, v128)
VSEL(s32, v64)
VSEL(u32, v64)
VSEL(f32, v64)
#undef VSEL

//----------------------------------------------------------------------------//
//  vmov
//----------------------------------------------------------------------------//
#define VMOV_PG(TYPE, NUM)                                                     \
  template <class Tm = std::integral_constant<Mode, Mode::MERGING_VALUE>>      \
  CCE_INTRINSIC void vmov(vector_##TYPE &dst, vector_##TYPE src,               \
                          vector_bool mask, Tm mode = MODE_MERGING) {          \
    static_assert(mode.value == MODE_MERGING.value,                            \
                  "The last argument can only be 'MODE_MERGING' or empty.");   \
    dst = __builtin_cce_vmov_##NUM##TYPE##_m(dst, src, mask);                  \
    return;                                                                    \
  }
VMOV_PG(u8, v256)
VMOV_PG(s8, v256)
VMOV_PG(u16, v128)
VMOV_PG(s16, v128)
VMOV_PG(f16, v128)
VMOV_PG(s32, v64)
VMOV_PG(u32, v64)
VMOV_PG(f32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VMOV_PG(bf16, v128)
#endif
#undef VMOV_PG

#define VMOV(TYPE, NUM)                                                        \
  CCE_INTRINSIC void vmov(vector_##TYPE &dst, vector_##TYPE src) {             \
    dst = __builtin_cce_vmov_##NUM##TYPE(src);                                 \
    return;                                                                    \
  }
VMOV(u8, v256)
VMOV(s8, v256)
VMOV(u16, v128)
VMOV(s16, v128)
VMOV(f16, v128)
VMOV(s32, v64)
VMOV(u32, v64)
VMOV(f32, v64)
VMOV(s64, v32)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VMOV(bf16, v128)
#endif
#undef VMOV

#ifdef ENABLE_MOVVP
//----------------------------------------------------------------------------//
//  MOVVP
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define MOVVP(TYPE, NUM)                                                       \
  CCE_INTRINSIC void movvp(vector_bool &dst, vector_##TYPE src,                \
                           int16_t part) {                                     \
    dst = __builtin_cce_movvp_##NUM##TYPE(src, part);                          \
    return;                                                                    \
  }

MOVVP(u16, v128)
MOVVP(u32, v64)
#undef MOVVP
#endif
#endif // ENABLE_MOVVP

//----------------------------------------------------------------------------//
//  PLT
//----------------------------------------------------------------------------//
#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
#define PLT_V210(TYPE, SCALAR)                                                 \
  CCE_INTRINSIC vector_bool plt_##TYPE(SCALAR scalar) {                        \
    return __builtin_cce_plt_##TYPE(scalar);                                   \
  }

PLT_V210(b8, uint32_t)
PLT_V210(b16, uint32_t)
PLT_V210(b32, uint32_t)
#undef PLT_V210
#endif

// marked POST_UPDATE for post-update interface
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define PLT_V300(TYPE, SCALAR)                                                 \
  template <class T>                                                           \
  CCE_INTRINSIC vector_bool plt_##TYPE(SCALAR &scalar, T post) {               \
    static_assert(std::is_same<T, PostUpdateType>::value,                      \
                  "The last argument can only be 'POST_UPDATE'.");             \
    struct {                                                                   \
      vector_bool mask;                                                        \
      SCALAR scalar_out;                                                       \
    } ret;                                                                     \
    __builtin_cce_plt_##TYPE##_v300(&ret, scalar);                             \
    scalar = ret.scalar_out;                                                   \
    return ret.mask;                                                           \
  }

PLT_V300(b8, uint32_t)
PLT_V300(b16, uint32_t)
PLT_V300(b32, uint32_t)
#undef PLT_V300
#endif

#if defined(__DAV_C310__) || defined(__DAV_M310__)
template <class T>
CCE_INTRINSIC vector_bool plt_2xvl_b64(uint32_t &scalar, T post) {
  static_assert(std::is_same<T, PostUpdateType>::value,
                "The last argument can only be 'POST_UPDATE'.");
  struct {
    vector_bool mask;
    uint32_t scalar_out;
  } ret;
  __builtin_cce_plt_b32_v300(&ret, scalar);
  scalar = ret.scalar_out;
  return ret.mask;
}
#endif

//----------------------------------------------------------------------------//
//  PLTM
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define PLTM(TYPE, SCALAR16BIT, SCALAR32BIT)                                   \
  CCE_INTRINSIC vector_bool pltm_##TYPE(SCALAR16BIT scalar0,                   \
                                        SCALAR32BIT scalar1) {                 \
    return __builtin_cce_pltm_##TYPE##_v300(scalar0, scalar1);                 \
  }

PLTM(b8, uint16_t, uint32_t)
PLTM(b16, uint16_t, uint32_t)
PLTM(b32, uint16_t, uint32_t)
#undef PLTM
#endif

#if defined(__DAV_C310__) || defined(__DAV_M310__)
CCE_INTRINSIC vector_bool pltm_2xvl_b64(uint16_t scalar0, uint32_t scalar1) {
  return __builtin_cce_pltm_b32_v300(scalar0, scalar1);
}
#endif

//----------------------------------------------------------------------------//
//  vselr
//----------------------------------------------------------------------------//
#define VSELR(TYPE1, TYPE2, NUM)                                               \
  CCE_INTRINSIC void vselr(vector_##TYPE1 &dst, vector_##TYPE1 src0,           \
                           vector_##TYPE2 src1) {                              \
    dst = __builtin_cce_vselr_##NUM##TYPE1(src0, src1);                        \
    return;                                                                    \
  }

VSELR(u8, u8, v256)
VSELR(s8, u8, v256)
VSELR(u16, u16, v128)
VSELR(s16, u16, v128)
#if defined(__DAV_C310__) || defined(__DAV_M310__)
VSELR(u32, u32, v64)
VSELR(s32, u32, v64)
#endif
VSELR(f16, u16, v128)
#undef VSELR

//----------------------------------------------------------------------------//
//  vdupi
//----------------------------------------------------------------------------//
#define VDUPI_B8(VTYPE, STYPE)                                                 \
  CCE_INTRINSIC void DEPRECATED vdupi_##VTYPE(vector_##VTYPE &dst,             \
                                              STYPE src) {                     \
    dst = __builtin_cce_vdupi_v256##VTYPE(src);                                \
    return;                                                                    \
  }

VDUPI_B8(u8, uint8_t)
VDUPI_B8(s8, int8_t)
#undef VDUPI_B8

#define VDUPI_B16(VTYPE, STYPE16, STYPE8)                                      \
  CCE_INTRINSIC void DEPRECATED vdupi_##VTYPE(vector_##VTYPE &dst,             \
                                              STYPE16 src) {                   \
    dst = __builtin_cce_vdupi_v128u16_x((src & 0x00FF), 0);                    \
    dst = __builtin_cce_vdupi_v128u16_m(dst, (src & 0xFF00) >> 8, 1);          \
    return;                                                                    \
  }

VDUPI_B16(u16, uint16_t, uint8_t)
VDUPI_B16(s16, int16_t, uint8_t)
#undef VDUPI_B16

//----------------------------------------------------------------------------//
//  vdups
//----------------------------------------------------------------------------//
#define VDUPS(VTYPE, NUM, STYPE)                                               \
  template <class T>                                                           \
  CCE_INTRINSIC void vdup(vector_##VTYPE &dst, STYPE src, vector_bool mask,    \
                          T mode) {                                            \
    static_assert(mode.value == MODE_MERGING.value ||                          \
                      mode.value == MODE_ZEROING.value,                        \
                  "The 4th argument of this vdup can only be: "                \
                  "MODE_MERGING, MODE_ZEROING");                               \
    dst = (mode.value == MODE_MERGING.value)                                   \
              ? __builtin_cce_vdups_##NUM##VTYPE(dst, src, mask, 0)            \
              : __builtin_cce_vdups_##NUM##VTYPE##_z(src, mask, 1);            \
    return;                                                                    \
  }

VDUPS(u8, v256, uint8_t)
VDUPS(s8, v256, int8_t)
VDUPS(u16, v128, uint16_t)
VDUPS(s16, v128, int16_t)
VDUPS(f16, v128, half)
VDUPS(u32, v64, uint32_t)
VDUPS(s32, v64, int32_t)
VDUPS(f32, v64, float)
#undef VDUPS

//----------------------------------------------------------------------------//
//  vdup/vdupm
//----------------------------------------------------------------------------//
#define VDUP(TYPE, NUM)                                                        \
  template <class T1, class T2>                                                \
  CCE_INTRINSIC void vdup(vector_##TYPE &dst, vector_##TYPE src,               \
                          vector_bool mask, T1 POS, T2 mode) {                 \
    static_assert(std::is_same<T1, Lowest_Type>::value ||                      \
                      std::is_same<T1, Highest_Type>::value,                   \
                  "The 4th argument of this vdup can only be: "                \
                  "POS_LOWEST, POS_HIGHEST");                                  \
    static_assert(mode.value == MODE_MERGING.value ||                          \
                      mode.value == MODE_ZEROING.value,                        \
                  "The 5th argument of this vdup can only be: "                \
                  "MODE_MERGING, MODE_ZEROING");                               \
    if (POS.value == POS_LOWEST.value) {                                       \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vdup_##NUM##TYPE(dst, src, mask, 0)            \
                : __builtin_cce_vdup_##NUM##TYPE##_z(src, mask, 1);            \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vdupm_##NUM##TYPE(dst, src, mask, 0)           \
                : __builtin_cce_vdupm_##NUM##TYPE##_z(src, mask, 1);           \
    }                                                                          \
    return;                                                                    \
  }

VDUP(u8, v256)
VDUP(s8, v256)
VDUP(u16, v128)
VDUP(s16, v128)
VDUP(f16, v128)
VDUP(u32, v64)
VDUP(s32, v64)
VDUP(f32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
VDUP(bf16, v128)
#endif
#undef VDUP

//----------------------------------------------------------------------------//
//  vpack
//----------------------------------------------------------------------------//
#define VPACK(FROM, TO)                                                        \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vpack(vector_##TO &dst, vector_##FROM src, T1 part,       \
                           T2 mode = MODE_UNKNOWN) {                           \
    static_assert(std::is_class<T1>::value, "the 3rd argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 4th argument is not valid");  \
    static_assert(                                                             \
        std::is_same<T1, Lower_Type>::value ||                                 \
            std::is_same<T1, Higher_Type>::value,                              \
        "The 3rd argument of this vpack can only be: LOWER, HIGHER");          \
    static_assert(mode.value == MODE_ZEROING.value ||                          \
                      mode.value == MODE_UNKNOWN.value ||                      \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask = pset_b8(PAT_H);                                     \
        vector_##TO dstTmp =                                                   \
            __builtin_cce_vpack_##FROM##2##TO##_x(src, (ULL)part.value);       \
        if (part.value == HIGHER.value) {                                      \
          vector_bool mask1 = pset_b8(PAT_ALL);                                \
          mask = __builtin_cce_pnot_z(mask, mask1);                            \
        }                                                                      \
        vmov(dst, dstTmp, mask);                                               \
      } else {                                                                 \
        dst = __builtin_cce_vpack_##FROM##2##TO##_x(src, (ULL)part.value);     \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_UNKNOWN.value)                                 \
                ? __builtin_cce_vpack_##FROM##2##TO##_x(src, (ULL)part.value)  \
                : __builtin_cce_vpack_##FROM##2##TO##_m(dst, src,              \
                                                        (ULL)part.value);      \
    }                                                                          \
    return;                                                                    \
  }

VPACK(u16, u8)
VPACK(s16, u8)
VPACK(u32, u16)
VPACK(s32, u16)
#undef VPACK

//----------------------------------------------------------------------------//
//  vzunpack && vsunpack
//----------------------------------------------------------------------------//
#define VUNPACK(FROM, TO, S)                                                   \
  template <class T>                                                           \
  CCE_INTRINSIC void vunpack(vector_##TO &dst, vector_##FROM src, T part) {    \
    static_assert(std::is_class<T>::value, "the 3rd argument is not valid");   \
    static_assert(                                                             \
        std::is_same<T, Lower_Type>::value ||                                  \
            std::is_same<T, Higher_Type>::value,                               \
        "The 3rd argument of this vunpack can only be: LOWER, HIGHER");        \
    dst = (S == 1)                                                             \
              ? __builtin_cce_vsunpack_##FROM##2##TO(src, (ULL)part.value)     \
              : __builtin_cce_vzunpack_##FROM##2##TO(src, (ULL)part.value);    \
    return;                                                                    \
  }

VUNPACK(u8, u16, 0)
VUNPACK(s8, s16, 1)
VUNPACK(u16, u32, 0)
VUNPACK(s16, s32, 1)
#undef VUNPACK

enum class PatVcp {
  CHN4TO8,
  CHN4TO16,
  CHN4TO32,
  CHN8TO16,
  CHN8TO32,
};

typedef std::integral_constant<PatVcp, PatVcp::CHN4TO8> PatCHN4TO8Type;
typedef std::integral_constant<PatVcp, PatVcp::CHN4TO16> PatCHN4TO16Type;
typedef std::integral_constant<PatVcp, PatVcp::CHN4TO32> PatCHN4TO32Type;
typedef std::integral_constant<PatVcp, PatVcp::CHN8TO16> PatCHN8TO16Type;
typedef std::integral_constant<PatVcp, PatVcp::CHN8TO32> PatCHN8TO32Type;

#define PAT_CHN4TO8 PatCHN4TO8Type()
#define PAT_CHN4TO16 PatCHN4TO16Type()
#define PAT_CHN4TO32 PatCHN4TO32Type()
#define PAT_CHN8TO16 PatCHN8TO16Type()
#define PAT_CHN8TO32 PatCHN8TO32Type()

//----------------------------------------------------------------------------//
//  vcp
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define INVALID_VALUE_VCP "The 2th argument of vcp is not valid"
#define ERROR_VALUE_VCP                                                        \
  "The 2th argument of vcp can only be: PAT_CHN4TO8, PAT_CHN4TO16, "           \
  "PAT_CHN4TO32, PAT_CHN8TO16, PAT_CHN8TO32"

#define VCP(TYPE, NUM)                                                         \
  template <class T> CCE_INTRINSIC void vcp(vector_##TYPE &dst, T pat) {       \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VCP);                 \
    static_assert(std::is_same<T, PatCHN4TO8Type>::value ||                    \
                      std::is_same<T, PatCHN4TO16Type>::value ||               \
                      std::is_same<T, PatCHN4TO32Type>::value ||               \
                      std::is_same<T, PatCHN8TO16Type>::value ||               \
                      std::is_same<T, PatCHN8TO32Type>::value,                 \
                  ERROR_VALUE_VCP);                                            \
    dst = __builtin_cce_vcp_##NUM##TYPE((ULL)pat.value);                       \
    return;                                                                    \
  }

VCP(s32, v64)
VCP(u32, v64)
VCP(s16, v128)
VCP(u16, v128)
#undef VCP
#endif

//----------------------------------------------------------------------------//
//  vabs/vrelu/vexp/vsqrt/vrsqrt/vrec
//  vln/vneg/vbcnt/vcls/vnot
//----------------------------------------------------------------------------//
#define UNARY_OP(OP_NAME, DTYPE, STYPE, NUM)                                   \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(vector_##DTYPE &dst, vector_##STYPE src,          \
                             vector_bool mask, T mode = MODE_UNKNOWN) {        \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      vector_##DTYPE dstTmp =                                                  \
          __builtin_cce_##OP_NAME##_##NUM##STYPE##_x(src, mask);               \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vmov_##NUM##DTYPE##_m(dst, dstTmp, mask)       \
                : dstTmp;                                                      \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_##OP_NAME##_##NUM##STYPE##_m(dst, src, mask)   \
                : __builtin_cce_##OP_NAME##_##NUM##STYPE##_x(src, mask);       \
    }                                                                          \
    return;                                                                    \
  }

UNARY_OP(vabs, s8, s8, v256)
UNARY_OP(vabs, s16, s16, v128)
UNARY_OP(vabs, s32, s32, v64)
UNARY_OP(vabs, f16, f16, v128)
UNARY_OP(vabs, f32, f32, v64)

UNARY_OP(vrelu, f16, f16, v128)
UNARY_OP(vrelu, f32, f32, v64)
UNARY_OP(vrelu, s32, s32, v64)

UNARY_OP(vexp, f16, f16, v128)
UNARY_OP(vexp, f32, f32, v64)

UNARY_OP(vsqrt, f16, f16, v128)
UNARY_OP(vsqrt, f32, f32, v64)

UNARY_OP(vrsqrt, f16, f16, v128)
UNARY_OP(vrsqrt, f32, f32, v64)

UNARY_OP(vrec, f16, f16, v128)
UNARY_OP(vrec, f32, f32, v64)

UNARY_OP(vln, f16, f16, v128)
UNARY_OP(vln, f32, f32, v64)

UNARY_OP(vneg, s8, s8, v256)
UNARY_OP(vneg, s16, s16, v128)
UNARY_OP(vneg, s32, s32, v64)
UNARY_OP(vneg, f16, f16, v128)
UNARY_OP(vneg, f32, f32, v64)

UNARY_OP(vbcnt, s8, u8, v256)
UNARY_OP(vbcnt, s8, s8, v256)
UNARY_OP(vbcnt, s16, u16, v128)
UNARY_OP(vbcnt, s16, s16, v128)
UNARY_OP(vbcnt, s32, u32, v64)
UNARY_OP(vbcnt, s32, s32, v64)

UNARY_OP(vcls, s8, s8, v256)
UNARY_OP(vcls, s16, s16, v128)
UNARY_OP(vcls, s32, s32, v64)
UNARY_OP(vcls, u8, u8, v256)
UNARY_OP(vcls, u16, u16, v128)
UNARY_OP(vcls, u32, u32, v64)

UNARY_OP(vnot, u8, u8, v256)
UNARY_OP(vnot, s8, s8, v256)
UNARY_OP(vnot, u16, u16, v128)
UNARY_OP(vnot, s16, s16, v128)
UNARY_OP(vnot, f16, f16, v128)
UNARY_OP(vnot, s32, s32, v64)
UNARY_OP(vnot, u32, u32, v64)
UNARY_OP(vnot, f32, f32, v64)
#undef UNARY_OP

//----------------------------------------------------------------------------//
//  vadd/vsub/vmul/vdiv/vabsdif/vmax/vmin/vand
//  vsadd/vssub/vdivf/vshl/vshr/vrnd/vprelu/vor/vxor
//----------------------------------------------------------------------------//
#define BINARY_OP(OP_NAME, TYPE1, TYPE2, NUM)                                  \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(vector_##TYPE1 &dst, vector_##TYPE1 src0,         \
                             vector_##TYPE2 src1, vector_bool mask,            \
                             T mode = MODE_UNKNOWN) {                          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      vector_##TYPE1 dstTmp =                                                  \
          __builtin_cce_##OP_NAME##_##NUM##TYPE1##_x(src0, src1, mask);        \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vmov_##NUM##TYPE1##_m(dst, dstTmp, mask)       \
                : dstTmp;                                                      \
    } else {                                                                   \
      dst =                                                                    \
          (mode.value == MODE_MERGING.value)                                   \
              ? __builtin_cce_##OP_NAME##_##NUM##TYPE1##_m(dst, src0, src1,    \
                                                           mask)               \
              : __builtin_cce_##OP_NAME##_##NUM##TYPE1##_x(src0, src1, mask);  \
    }                                                                          \
    return;                                                                    \
  }

BINARY_OP(vadd, s32, s32, v64)
BINARY_OP(vadd, u32, u32, v64)
BINARY_OP(vadd, s16, s16, v128)
BINARY_OP(vadd, u16, u16, v128)
BINARY_OP(vadd, s8, s8, v256)
BINARY_OP(vadd, u8, u8, v256)
BINARY_OP(vadd, f16, f16, v128)
BINARY_OP(vadd, f32, f32, v64)

BINARY_OP(vsub, s32, s32, v64)
BINARY_OP(vsub, u32, u32, v64)
BINARY_OP(vsub, s16, s16, v128)
BINARY_OP(vsub, u16, u16, v128)
BINARY_OP(vsub, s8, s8, v256)
BINARY_OP(vsub, u8, u8, v256)
BINARY_OP(vsub, f16, f16, v128)
BINARY_OP(vsub, f32, f32, v64)

BINARY_OP(vmul, s32, s32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
BINARY_OP(vmul, u32, u32, v64)
#endif
BINARY_OP(vmul, s16, s16, v128)
BINARY_OP(vmul, u16, u16, v128)
BINARY_OP(vmul, s8, s8, v256)
BINARY_OP(vmul, u8, u8, v256)
BINARY_OP(vmul, f32, f32, v64)
BINARY_OP(vmul, f16, f16, v128)

BINARY_OP(vdiv, s32, s32, v64);
BINARY_OP(vdiv, u32, u32, v64);
BINARY_OP(vdiv, s16, s16, v128);
BINARY_OP(vdiv, u16, u16, v128);
BINARY_OP(vdiv, f16, f16, v128);
BINARY_OP(vdiv, f32, f32, v64);

BINARY_OP(vabsdif, u8, u8, v256)
BINARY_OP(vabsdif, s8, s8, v256)
BINARY_OP(vabsdif, u16, u16, v128)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
BINARY_OP(vabsdif, u32, u32, v64)
#endif
BINARY_OP(vabsdif, s16, s16, v128)
BINARY_OP(vabsdif, f16, f16, v128)
BINARY_OP(vabsdif, s32, s32, v64)
BINARY_OP(vabsdif, f32, f32, v64)

BINARY_OP(vmax, s8, s8, v256)
BINARY_OP(vmax, s16, s16, v128)
BINARY_OP(vmax, s32, s32, v64)
BINARY_OP(vmax, u8, u8, v256)
BINARY_OP(vmax, u16, u16, v128)
BINARY_OP(vmax, u32, u32, v64)
BINARY_OP(vmax, f16, f16, v128)
BINARY_OP(vmax, f32, f32, v64)

BINARY_OP(vmin, s8, s8, v256)
BINARY_OP(vmin, s16, s16, v128)
BINARY_OP(vmin, s32, s32, v64)
BINARY_OP(vmin, u8, u8, v256)
BINARY_OP(vmin, u16, u16, v128)
BINARY_OP(vmin, u32, u32, v64)
BINARY_OP(vmin, f16, f16, v128)
BINARY_OP(vmin, f32, f32, v64)

BINARY_OP(vdivf, u16, u16, v128)

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
BINARY_OP(vshl, s8, s8, v256)
#endif
BINARY_OP(vshl, s16, s16, v128)
BINARY_OP(vshl, s32, s32, v64)
BINARY_OP(vshl, u8, s8, v256)
BINARY_OP(vshl, u16, s16, v128)
BINARY_OP(vshl, u32, s32, v64)

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
BINARY_OP(vshr, s8, s8, v256)
#endif
BINARY_OP(vshr, s16, s16, v128)
BINARY_OP(vshr, s32, s32, v64)
BINARY_OP(vshr, u8, s8, v256)
BINARY_OP(vshr, u16, s16, v128)
BINARY_OP(vshr, u32, s32, v64)

BINARY_OP(vrnd, s16, u16, v128)
BINARY_OP(vrnd, s32, u32, v64)

BINARY_OP(vprelu, f16, f16, v128)
BINARY_OP(vprelu, f32, f32, v64)

BINARY_OP(vsadd, s16, s16, v128)

BINARY_OP(vssub, s16, s16, v128)

BINARY_OP(vand, s32, s32, v64)
BINARY_OP(vand, u32, u32, v64)
BINARY_OP(vand, s16, s16, v128)
BINARY_OP(vand, u16, u16, v128)
BINARY_OP(vand, s8, s8, v256)
BINARY_OP(vand, u8, u8, v256)
BINARY_OP(vand, f32, f32, v64)
BINARY_OP(vand, f16, f16, v128)

BINARY_OP(vor, s32, s32, v64)
BINARY_OP(vor, u32, u32, v64)
BINARY_OP(vor, s16, s16, v128)
BINARY_OP(vor, u16, u16, v128)
BINARY_OP(vor, s8, s8, v256)
BINARY_OP(vor, u8, u8, v256)
BINARY_OP(vor, f32, f32, v64)
BINARY_OP(vor, f16, f16, v128)

BINARY_OP(vxor, s32, s32, v64)
BINARY_OP(vxor, u32, u32, v64)
BINARY_OP(vxor, s16, s16, v128)
BINARY_OP(vxor, u16, u16, v128)
BINARY_OP(vxor, s8, s8, v256)
BINARY_OP(vxor, u8, u8, v256)
BINARY_OP(vxor, f32, f32, v64)
BINARY_OP(vxor, f16, f16, v128)
#undef BINARY_OP

//----------------------------------------------------------------------------//
//  vmull
//----------------------------------------------------------------------------//
#if defined(__DAV_C310__) || defined(__DAV_M310__)
#define VMULL(TYPE, NUM)                                                       \
  DEPRECATED_AFTER_V210 CCE_INTRINSIC void vmull(                              \
      vector_##TYPE &dst0, vector_##TYPE &dst1, vector_##TYPE src0,            \
      vector_##TYPE src1, vector_bool mask) {                                  \
    vector_##TYPE##x2_t __ret;                                                 \
    __builtin_cce_vmull_##NUM##TYPE(&__ret, src0, src1, mask);                 \
    dst0 = __ret.val[0];                                                       \
    dst1 = __ret.val[1];                                                       \
    return;                                                                    \
  }
VMULL(s32, v64)
VMULL(u32, v64)
#undef VMULL

//----------------------------------------------------------------------------//
//  vaddc/vsubc
//----------------------------------------------------------------------------//
#define CARRY_BINARY_OP(OP_NAME, TYPE, NUM)                                    \
  DEPRECATED_AFTER_V210 CCE_INTRINSIC void OP_NAME(                            \
      vector_bool &carryp, vector_##TYPE &dst, vector_##TYPE src0,             \
      vector_##TYPE src1, vector_bool mask) {                                  \
    struct {                                                                   \
      vector_##TYPE dst_;                                                      \
      vector_bool carryp_;                                                     \
    } ret;                                                                     \
    __builtin_cce_##OP_NAME##_##NUM##TYPE(&ret, src0, src1, mask);             \
    dst = ret.dst_;                                                            \
    carryp = ret.carryp_;                                                      \
    return;                                                                    \
  }

CARRY_BINARY_OP(vaddc, s32, v64)
CARRY_BINARY_OP(vaddc, u32, v64)
CARRY_BINARY_OP(vsubc, s32, v64)
CARRY_BINARY_OP(vsubc, u32, v64)
#undef CARRY_BINARY_OP

//----------------------------------------------------------------------------//
//  vaddcs/vsubcs
//----------------------------------------------------------------------------//
#define CARRY_TERNARY_OP(OP_NAME, TYPE, NUM)                                   \
  DEPRECATED_AFTER_V210 CCE_INTRINSIC void OP_NAME(                            \
      vector_bool &carryp, vector_##TYPE &dst, vector_##TYPE src0,             \
      vector_##TYPE src1, vector_bool carrysrcp, vector_bool mask) {           \
    struct {                                                                   \
      vector_##TYPE dst_;                                                      \
      vector_bool carryp_;                                                     \
    } ret;                                                                     \
    __builtin_cce_##OP_NAME##_##NUM##TYPE(&ret, src0, src1, carrysrcp, mask);  \
    dst = ret.dst_;                                                            \
    carryp = ret.carryp_;                                                      \
    return;                                                                    \
  }

CARRY_TERNARY_OP(vaddcs, s32, v64)
CARRY_TERNARY_OP(vaddcs, u32, v64)
CARRY_TERNARY_OP(vsubcs, s32, v64)
CARRY_TERNARY_OP(vsubcs, u32, v64)
#undef CARRY_TERNARY_OP
#endif

//----------------------------------------------------------------------------//
//  vmula/vmadd/vadd3/vadif/vsad
//----------------------------------------------------------------------------//
#define TERNARY_OP(OP_NAME, TYPE, NUM)                                         \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(vector_##TYPE &dst, vector_##TYPE src0,           \
                             vector_##TYPE src1, vector_bool mask,             \
                             T mode = MODE_UNKNOWN) {                          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_##TYPE dstTmp = __builtin_cce_vmov_##NUM##TYPE(dst);            \
        dstTmp = __builtin_cce_##OP_NAME##_##NUM##TYPE##_m(dstTmp, src0, src1, \
                                                           mask);              \
        dst = __builtin_cce_vmov_##NUM##TYPE##_m(dst, dstTmp, mask);           \
      } else {                                                                 \
        dst =                                                                  \
            __builtin_cce_##OP_NAME##_##NUM##TYPE##_m(dst, src0, src1, mask);  \
      }                                                                        \
    } else {                                                                   \
      dst = __builtin_cce_##OP_NAME##_##NUM##TYPE##_m(dst, src0, src1, mask);  \
    }                                                                          \
    return;                                                                    \
  }

TERNARY_OP(vmula, s32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
TERNARY_OP(vmula, u32, v64)
#endif
TERNARY_OP(vmula, s16, v128)
TERNARY_OP(vmula, u16, v128)
TERNARY_OP(vmula, s8, v256)
TERNARY_OP(vmula, u8, v256)
TERNARY_OP(vmula, f32, v64)
TERNARY_OP(vmula, f16, v128)

TERNARY_OP(vmadd, f16, v128)
TERNARY_OP(vmadd, f32, v64)

TERNARY_OP(vadd3, u8, v256)
TERNARY_OP(vadd3, s8, v256)
TERNARY_OP(vadd3, u16, v128)
TERNARY_OP(vadd3, s16, v128)
TERNARY_OP(vadd3, u32, v64)
TERNARY_OP(vadd3, s32, v64)

TERNARY_OP(vadif, u8, v256)
TERNARY_OP(vadif, s8, v256)
TERNARY_OP(vadif, u16, v128)
TERNARY_OP(vadif, s16, v128)
TERNARY_OP(vadif, u32, v64)
TERNARY_OP(vadif, s32, v64)

TERNARY_OP(vsad, u8, v256)
TERNARY_OP(vsad, s8, v256)
TERNARY_OP(vsad, u16, v128)
TERNARY_OP(vsad, s16, v128)
TERNARY_OP(vsad, u32, v64)
TERNARY_OP(vsad, s32, v64)
#undef TERNARY_OP

//----------------------------------------------------------------------------//
//  vcadd
//----------------------------------------------------------------------------//
#define VCADD(TYPE_DST, TYPE_SRC, NUM, NUM_DST, PTYPE)                         \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void vcadd(vector_##TYPE_DST &dst, vector_##TYPE_SRC src,      \
                           vector_bool mask, T mode = MODE_UNKNOWN) {          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      vector_##TYPE_DST dstTmp =                                               \
          __builtin_cce_vcadd_##NUM##TYPE_SRC##_x(src, mask);                  \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask1 = pge_##PTYPE(PAT_VL1);                              \
        dst = __builtin_cce_vmov_##NUM_DST##TYPE_DST##_m(dst, dstTmp, mask1);  \
      } else {                                                                 \
        dst = dstTmp;                                                          \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vcadd_##NUM##TYPE_SRC##_m(dst, src, mask)      \
                : __builtin_cce_vcadd_##NUM##TYPE_SRC##_x(src, mask);          \
    }                                                                          \
    return;                                                                    \
  }

VCADD(s16, s8, v256, v128, b16)
VCADD(s32, s16, v128, v64, b32)
VCADD(s32, s32, v64, v64, b32)
VCADD(u16, u8, v256, v128, b16)
VCADD(u32, u16, v128, v64, b32)
VCADD(u32, u32, v64, v64, b32)
VCADD(f16, f16, v128, v128, b16)
VCADD(f32, f32, v64, v64, b32)
#undef VCADD

//----------------------------------------------------------------------------//
//  vcmax/vcmin
//----------------------------------------------------------------------------//
#define VCMAX_MIN(OP_NAME, TYPE, NUM, PTYPE)                                   \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(vector_##TYPE &dst, vector_##TYPE src,            \
                             vector_bool mask, T mode = MODE_UNKNOWN) {        \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      vector_##TYPE dstTmp =                                                   \
          __builtin_cce_##OP_NAME##_##NUM##TYPE##_x(src, mask);                \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask1 = pge_##PTYPE(PAT_VL2);                              \
        dst = __builtin_cce_vmov_##NUM##TYPE##_m(dst, dstTmp, mask1);          \
      } else {                                                                 \
        dst = dstTmp;                                                          \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_##OP_NAME##_##NUM##TYPE##_m(dst, src, mask)    \
                : __builtin_cce_##OP_NAME##_##NUM##TYPE##_x(src, mask);        \
    }                                                                          \
    return;                                                                    \
  }

VCMAX_MIN(vcmax, s8, v256, b8)
VCMAX_MIN(vcmax, s16, v128, b16)
VCMAX_MIN(vcmax, s32, v64, b32)
VCMAX_MIN(vcmax, u8, v256, b8)
VCMAX_MIN(vcmax, u16, v128, b16)
VCMAX_MIN(vcmax, u32, v64, b32)
VCMAX_MIN(vcmax, f16, v128, b16)
VCMAX_MIN(vcmax, f32, v64, b32)

VCMAX_MIN(vcmin, s8, v256, b8)
VCMAX_MIN(vcmin, s16, v128, b16)
VCMAX_MIN(vcmin, s32, v64, b32)
VCMAX_MIN(vcmin, u8, v256, b8)
VCMAX_MIN(vcmin, u16, v128, b16)
VCMAX_MIN(vcmin, u32, v64, b32)
VCMAX_MIN(vcmin, f16, v128, b16)
VCMAX_MIN(vcmin, f32, v64, b32)
#undef VCMAX_MIN

//----------------------------------------------------------------------------//
//  vcbmax/vcbmin
//----------------------------------------------------------------------------//
#define VCBMAX_MIN(OP_NAME, TYPE, NUM, PTYPE)                                  \
  template <class Tm = Mode_Unknown_Type>                                      \
  CCE_INTRINSIC void OP_NAME(vector_##TYPE &dst, vector_bool &dst1,            \
                             vector_##TYPE src, vector_bool pg,                \
                             Tm mode = MODE_UNKNOWN) {                         \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    struct {                                                                   \
      vector_##TYPE dst_;                                                      \
      vector_bool dst1_;                                                       \
    } ret;                                                                     \
    if (isSoftwareMergeMode()) {                                               \
      __builtin_cce_##OP_NAME##_##NUM##TYPE##_x(&ret, src, pg);                \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask1 = pge_##PTYPE(PAT_VL1);                              \
        dst = __builtin_cce_vmov_##NUM##TYPE##_m(dst, ret.dst_, mask1);        \
      } else {                                                                 \
        dst = ret.dst_;                                                        \
      }                                                                        \
      dst1 = ret.dst1_;                                                        \
    } else {                                                                   \
      __builtin_cce_##OP_NAME##_##NUM##TYPE##_m(&ret, dst, dst1, src, pg);     \
      dst = ret.dst_;                                                          \
      dst1 = ret.dst1_;                                                        \
    }                                                                          \
    return;                                                                    \
  }

VCBMAX_MIN(vcbmax, s16, v128, b16)
VCBMAX_MIN(vcbmax, s8, v256, b8)
VCBMAX_MIN(vcbmax, u16, v128, b16)
VCBMAX_MIN(vcbmax, u8, v256, b8)
VCBMAX_MIN(vcbmax, s32, v64, b32)
VCBMAX_MIN(vcbmax, u32, v64, b32)
VCBMAX_MIN(vcbmax, f16, v128, b16)
VCBMAX_MIN(vcbmax, f32, v64, b32)

VCBMAX_MIN(vcbmin, s16, v128, b16)
VCBMAX_MIN(vcbmin, s8, v256, b8)
VCBMAX_MIN(vcbmin, u16, v128, b16)
VCBMAX_MIN(vcbmin, u8, v256, b8)
VCBMAX_MIN(vcbmin, s32, v64, b32)
VCBMAX_MIN(vcbmin, u32, v64, b32)
VCBMAX_MIN(vcbmin, f16, v128, b16)
VCBMAX_MIN(vcbmin, f32, v64, b32)
#undef VCBMAX_MIN

//----------------------------------------------------------------------------//
//  vsqz
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VSQZ(TYPE, NUM)                                                        \
  template <class T1>                                                          \
  CCE_INTRINSIC void vsqz(vector_##TYPE &dst, vector_##TYPE src,               \
                          vector_bool mask, T1 store) {                        \
    static_assert(std::is_same<T1, NoStoredType>::value ||                     \
                      std::is_same<T1, StoredType>::value,                     \
                  "The 4th argument of this vsqz can only be: "                \
                  "MODE_NO_STORED, MODE_STORED");                              \
    dst =                                                                      \
        __builtin_cce_vsqz_##NUM##TYPE##_x_v300(src, mask, (ULL)store.value);  \
    return;                                                                    \
  }
VSQZ(s8, v256)
VSQZ(s16, v128)
VSQZ(s32, v64)
VSQZ(u8, v256)
VSQZ(u16, v128)
VSQZ(u32, v64)
VSQZ(f16, v128)
VSQZ(f32, v64)
#undef VSQZ
#endif

#define VSQZ(TYPE, NUM)                                                        \
  CCE_INTRINSIC void vsqz(vector_##TYPE &dst, vector_##TYPE src,               \
                          vector_bool mask) {                                  \
    if (isV300Target() || isV310Target()) {                                    \
      dst = __builtin_cce_vsqz_##NUM##TYPE##_x_v300(src, mask,                 \
                                                    (ULL)MODE_STORED.value);   \
    } else {                                                                   \
      dst = __builtin_cce_vsqz_##NUM##TYPE##_m(dst, src, mask);                \
    }                                                                          \
    return;                                                                    \
  }
VSQZ(s8, v256)
VSQZ(s16, v128)
VSQZ(s32, v64)
VSQZ(u8, v256)
VSQZ(u16, v128)
VSQZ(u32, v64)
VSQZ(f16, v128)
VSQZ(f32, v64)
#undef VSQZ

//----------------------------------------------------------------------------//
//  vusqz
//  Vd need be initialized as all '0' before calling this VUSQZ, so it's merging
//  mode. And it's to calcaulte the number of '1' in Pg, so the Vd type could
//  only be s8/s16/s32.
//----------------------------------------------------------------------------//
#define VUSQZ(TYPE, NUM)                                                       \
  CCE_INTRINSIC void vusqz(vector_##TYPE &dst, vector_bool mask) {             \
    dst = __builtin_cce_vusqz_##NUM##TYPE##_m(dst, mask);                      \
    return;                                                                    \
  }

VUSQZ(s8, v256)
VUSQZ(s16, v128)
VUSQZ(s32, v64)
VUSQZ(u8, v256)
VUSQZ(u16, v128)
VUSQZ(u32, v64)
#undef VUSQZ

//----------------------------------------------------------------------------//
//  vslide
//----------------------------------------------------------------------------//
#define VSLIDE(TYPE, NUM)                                                      \
  CCE_INTRINSIC void vslide(vector_##TYPE &dst, vector_##TYPE src0,            \
                            vector_##TYPE src1, int16_t slideAmount) {         \
    dst = __builtin_cce_vslide_##NUM##TYPE(src0, src1, slideAmount);           \
    return;                                                                    \
  }

VSLIDE(s32, v64)
VSLIDE(u32, v64)
VSLIDE(s16, v128)
VSLIDE(u16, v128)
VSLIDE(s8, v256)
VSLIDE(u8, v256)
VSLIDE(f32, v64)
VSLIDE(f16, v128)
#undef VSLIDE

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
//----------------------------------------------------------------------------//
//  vintegral
//
// VINTEGRALS1-S3 always use together, so we provide one interface for user.
// Also we provide three interfaces separately for VINTEGRALS1-S3.
// So user can do some optimization.
//----------------------------------------------------------------------------//
CCE_INTRINSIC void vintegral(vector_u16 &dst0, vector_u16 &dst1, vector_u8 src0,
                             vector_u16 src1, vector_u16 src2) {
  __builtin_cce_vintegrals1(src0);
  __builtin_cce_vintegrals2();
  vector_u16x2_t __ret;
  __builtin_cce_vintegrals3(&__ret, src1, src2);
  dst0 = __ret.val[0];
  dst1 = __ret.val[1];
  return;
}

CCE_INTRINSIC void vintegrals1(vector_u8 src) {
  return __builtin_cce_vintegrals1(src);
}

CCE_INTRINSIC void vintegrals2() { return __builtin_cce_vintegrals2(); }

CCE_INTRINSIC void vintegrals3(vector_u16 &dst0, vector_u16 &dst1,
                               vector_u16 src1, vector_u16 src2) {
  vector_u16x2_t __ret;
  __builtin_cce_vintegrals3(&__ret, src1, src2);
  dst0 = __ret.val[0];
  dst1 = __ret.val[1];
  return;
}
#endif

#ifdef ENABLE_VINTEGRALV2
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
//----------------------------------------------------------------------------//
//  vintegralv2 for v300
//----------------------------------------------------------------------------//
CCE_INTRINSIC void vintegralv2(vector_u16x2_t &dst, vector_u8 src) {
  __builtin_cce_vintegralv2(&dst, src);
  return;
}
#endif
#endif // ENABLE_VINTEGRALV2

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
//----------------------------------------------------------------------------//
// DHISTV2&CHISTV2
//----------------------------------------------------------------------------//
enum class Bin {
  BIN_VALUE_0,
  BIN_VALUE_1,
  BIN_VALUE_2,
  BIN_VALUE_3,
};
typedef std::integral_constant<Bin, Bin::BIN_VALUE_0> Bin_N0_Type;
typedef std::integral_constant<Bin, Bin::BIN_VALUE_1> Bin_N1_Type;
typedef std::integral_constant<Bin, Bin::BIN_VALUE_2> Bin_N2_Type;
typedef std::integral_constant<Bin, Bin::BIN_VALUE_3> Bin_N3_Type;
#define Bin_N0 Bin_N0_Type()
#define Bin_N1 Bin_N1_Type()
#define Bin_N2 Bin_N2_Type()
#define Bin_N3 Bin_N3_Type()

template <class T>
CCE_INTRINSIC void dhistv2(vector_u16 &dst, vector_u8 src, vector_bool mask,
                           T bin) {
  static_assert(std::is_class<T>::value, "The 4th argument is not valid.");
  static_assert(std::is_same<T, Bin_N0_Type>::value ||
                    std::is_same<T, Bin_N1_Type>::value ||
                    std::is_same<T, Bin_N2_Type>::value ||
                    std::is_same<T, Bin_N3_Type>::value,
                "The 4th argument can only be Bin_N0_Type, Bin_N1_Type, "
                "Bin_N2_Type, Bin_N3_Type.");
  dst = __builtin_cce_dhistv2_m(dst, src, mask, (ULL)bin.value);
  return;
}

template <class T>
CCE_INTRINSIC void chistv2(vector_u16 &dst, vector_u8 src, vector_bool mask,
                           T bin) {
  static_assert(std::is_class<T>::value, "The 4th argument is not valid.");
  static_assert(std::is_same<T, Bin_N0_Type>::value ||
                    std::is_same<T, Bin_N1_Type>::value ||
                    std::is_same<T, Bin_N2_Type>::value ||
                    std::is_same<T, Bin_N3_Type>::value,
                "The 4th argument can only be Bin_N0_Type, Bin_N1_Type, "
                "Bin_N2_Type, Bin_N3_Type.");
  dst = __builtin_cce_chistv2_m(dst, src, mask, (ULL)bin.value);
  return;
}
#endif
//----------------------------------------------------------------------------//
//  vci
//----------------------------------------------------------------------------//
enum class Order { INC_ORDER_VALUE, DEC_ORDER_VALUE };
typedef std::integral_constant<Order, Order::INC_ORDER_VALUE> IncOrderType;
typedef std::integral_constant<Order, Order::DEC_ORDER_VALUE> DecOrderType;
#define INC_ORDER IncOrderType()
#define DEC_ORDER DecOrderType()
#define VCI(TYPE, NUM, IDX_TYPE)                                               \
  template <class T = IncOrderType>                                            \
  CCE_INTRINSIC void vci(vector_##TYPE &dst, IDX_TYPE index,                   \
                         T order = INC_ORDER) {                                \
    if (isV300Target() || isV310Target()) {                                    \
      dst = __builtin_cce_vci_##NUM##TYPE(index, (ULL)order.value);            \
    } else {                                                                   \
      dst = __builtin_cce_vci_##NUM##TYPE(index, 0);                           \
    }                                                                          \
    return;                                                                    \
  }

VCI(s8, v256, int8_t)
VCI(s16, v128, int16_t)
VCI(s32, v64, int32_t)
#undef VCI

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VCI(TYPE, NUM, IDX_TYPE)                                               \
  template <class T = IncOrderType>                                            \
  CCE_INTRINSIC void vci(vector_##TYPE &dst, IDX_TYPE index,                   \
                         T order = INC_ORDER) {                                \
    dst = __builtin_cce_vci_##NUM##TYPE##_##TYPE(index, (ULL)order.value);     \
    return;                                                                    \
  }

VCI(f16, v128, half)
VCI(f32, v64, float)
#undef VCI
#endif

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
#define VCI(TYPE, NUM, IDX_TYPE)                                               \
  CCE_INTRINSIC void vci(vector_##TYPE &dst, IDX_TYPE index) {                 \
    dst = __builtin_cce_vci_##NUM##TYPE(index, 0);                             \
    return;                                                                    \
  }

VCI(f16, v128, uint16_t)
VCI(f32, v64, uint32_t)
#undef VCI
#endif
//----------------------------------------------------------------------------//
// DHIST0-DHIST3
//----------------------------------------------------------------------------//
#define DHIST(FROM, TO, NUM)                                                   \
  CCE_INTRINSIC void dhist0(vector_##TO &dst, vector_##FROM src,               \
                            vector_bool mask) {                                \
    dst = __builtin_cce_dhist0_##NUM##TO##_m(dst, src, mask);                  \
    return;                                                                    \
  }                                                                            \
  CCE_INTRINSIC void dhist1(vector_##TO &dst, vector_##FROM src,               \
                            vector_bool mask) {                                \
    dst = __builtin_cce_dhist1_##NUM##TO##_m(dst, src, mask);                  \
    return;                                                                    \
  }                                                                            \
  CCE_INTRINSIC void dhist2(vector_##TO &dst, vector_##FROM src,               \
                            vector_bool mask) {                                \
    dst = __builtin_cce_dhist2_##NUM##TO##_m(dst, src, mask);                  \
    return;                                                                    \
  }                                                                            \
  CCE_INTRINSIC void dhist3(vector_##TO &dst, vector_##FROM src,               \
                            vector_bool mask) {                                \
    dst = __builtin_cce_dhist3_##NUM##TO##_m(dst, src, mask);                  \
    return;                                                                    \
  }
#if defined(__DAV_M210_VEC__)
DHIST(u8, u32, v256)
#elif defined(__DAV_L210__)
DHIST(u8, u16, v128)
#endif
#undef DHIST

//----------------------------------------------------------------------------//
// CHIST0-CHIST3
//----------------------------------------------------------------------------//
#define CHIST(FROM, TO, NUM)                                                   \
  CCE_INTRINSIC void chist0(vector_##TO &dst, vector_##FROM src,               \
                            vector_bool mask) {                                \
    dst = __builtin_cce_chist0_##NUM##TO##_m(dst, src, mask);                  \
    return;                                                                    \
  }                                                                            \
  CCE_INTRINSIC void chist1(vector_##TO &dst, vector_##FROM src,               \
                            vector_bool mask) {                                \
    dst = __builtin_cce_chist1_##NUM##TO##_m(dst, src, mask);                  \
    return;                                                                    \
  }                                                                            \
  CCE_INTRINSIC void chist2(vector_##TO &dst, vector_##FROM src,               \
                            vector_bool mask) {                                \
    dst = __builtin_cce_chist2_##NUM##TO##_m(dst, src, mask);                  \
    return;                                                                    \
  }                                                                            \
  CCE_INTRINSIC void chist3(vector_##TO &dst, vector_##FROM src,               \
                            vector_bool mask) {                                \
    dst = __builtin_cce_chist3_##NUM##TO##_m(dst, src, mask);                  \
    return;                                                                    \
  }
#if defined(__DAV_M210_VEC__)
CHIST(u8, u32, v256)
#elif defined(__DAV_L210__)
CHIST(u8, u16, v128)
#endif
#undef CHIST

//----------------------------------------------------------------------------//
//  vaxpy
//----------------------------------------------------------------------------//
#define VAXPY(TYPE, NUM, SCALAR)                                               \
  template <class Tm = Mode_Unknown_Type>                                      \
  CCE_INTRINSIC void vaxpy(vector_##TYPE &dst, vector_##TYPE src0,             \
                           SCALAR scalar, vector_bool mask,                    \
                           Tm mode = MODE_UNKNOWN) {                           \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_##TYPE dstTmp = __builtin_cce_vmov_##NUM##TYPE(dst);            \
        dstTmp =                                                               \
            __builtin_cce_vaxpy_##NUM##TYPE##_m(dstTmp, src0, scalar, mask);   \
        dst = __builtin_cce_vmov_##NUM##TYPE##_m(dst, dstTmp, mask);           \
      } else {                                                                 \
        dst = __builtin_cce_vaxpy_##NUM##TYPE##_m(dst, src0, scalar, mask);    \
      }                                                                        \
    } else {                                                                   \
      dst = __builtin_cce_vaxpy_##NUM##TYPE##_m(dst, src0, scalar, mask);      \
    }                                                                          \
    return;                                                                    \
  }

VAXPY(f32, v64, float)
VAXPY(f16, v128, half)
#undef VAXPY

//----------------------------------------------------------------------------//
//  vadds/vmuls/vlrelu/vmaxs/vmins/vshls/vshrs/vrnds/vsadds
//----------------------------------------------------------------------------//

#define VEC_SCALAR_OP(OP_NAME, STYPE, TYPE, NUM)                               \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(vector_##TYPE &dst, vector_##TYPE src0,           \
                             STYPE scalar, vector_bool mask,                   \
                             T mode = MODE_UNKNOWN) {                          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      vector_##TYPE dstTmp =                                                   \
          __builtin_cce_##OP_NAME##_##NUM##TYPE##_x(src0, scalar, mask);       \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vmov_##NUM##TYPE##_m(dst, dstTmp, mask)        \
                : dstTmp;                                                      \
    } else {                                                                   \
      dst =                                                                    \
          (mode.value == MODE_MERGING.value)                                   \
              ? __builtin_cce_##OP_NAME##_##NUM##TYPE##_m(dst, src0, scalar,   \
                                                          mask)                \
              : __builtin_cce_##OP_NAME##_##NUM##TYPE##_x(src0, scalar, mask); \
    }                                                                          \
    return;                                                                    \
  }

VEC_SCALAR_OP(vadds, int32_t, s32, v64)
VEC_SCALAR_OP(vadds, int16_t, s16, v128)
VEC_SCALAR_OP(vadds, int8_t, s8, v256)
VEC_SCALAR_OP(vadds, uint32_t, u32, v64)
VEC_SCALAR_OP(vadds, uint16_t, u16, v128)
VEC_SCALAR_OP(vadds, uint8_t, u8, v256)
VEC_SCALAR_OP(vadds, float, f32, v64)
VEC_SCALAR_OP(vadds, half, f16, v128)

VEC_SCALAR_OP(vmuls, int32_t, s32, v64)
VEC_SCALAR_OP(vmuls, int16_t, s16, v128)
VEC_SCALAR_OP(vmuls, int8_t, s8, v256)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
VEC_SCALAR_OP(vmuls, uint32_t, u32, v64)
#endif
VEC_SCALAR_OP(vmuls, uint16_t, u16, v128)
VEC_SCALAR_OP(vmuls, uint8_t, u8, v256)
VEC_SCALAR_OP(vmuls, float, f32, v64)
VEC_SCALAR_OP(vmuls, half, f16, v128)

VEC_SCALAR_OP(vlrelu, float, f32, v64)
VEC_SCALAR_OP(vlrelu, half, f16, v128)

VEC_SCALAR_OP(vmaxs, int8_t, s8, v256)
VEC_SCALAR_OP(vmaxs, int16_t, s16, v128)
VEC_SCALAR_OP(vmaxs, int32_t, s32, v64)
VEC_SCALAR_OP(vmaxs, uint8_t, u8, v256)
VEC_SCALAR_OP(vmaxs, uint16_t, u16, v128)
VEC_SCALAR_OP(vmaxs, uint32_t, u32, v64)
VEC_SCALAR_OP(vmaxs, half, f16, v128)
VEC_SCALAR_OP(vmaxs, float, f32, v64)

VEC_SCALAR_OP(vmins, int8_t, s8, v256)
VEC_SCALAR_OP(vmins, int16_t, s16, v128)
VEC_SCALAR_OP(vmins, int32_t, s32, v64)
VEC_SCALAR_OP(vmins, uint8_t, u8, v256)
VEC_SCALAR_OP(vmins, uint16_t, u16, v128)
VEC_SCALAR_OP(vmins, uint32_t, u32, v64)
VEC_SCALAR_OP(vmins, half, f16, v128)
VEC_SCALAR_OP(vmins, float, f32, v64)

VEC_SCALAR_OP(vshls, int16_t, s16, v128)
VEC_SCALAR_OP(vshls, int16_t, s32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
VEC_SCALAR_OP(vshls, int16_t, s8, v256)
#endif
VEC_SCALAR_OP(vshls, int16_t, u8, v256)
VEC_SCALAR_OP(vshls, int16_t, u16, v128)
VEC_SCALAR_OP(vshls, int16_t, u32, v64)

VEC_SCALAR_OP(vshrs, int16_t, s16, v128)
VEC_SCALAR_OP(vshrs, int16_t, s32, v64)
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
VEC_SCALAR_OP(vshrs, int16_t, s8, v256)
#endif
VEC_SCALAR_OP(vshrs, int16_t, u8, v256)
VEC_SCALAR_OP(vshrs, int16_t, u16, v128)
VEC_SCALAR_OP(vshrs, int16_t, u32, v64)

VEC_SCALAR_OP(vrnds, uint16_t, s16, v128)
VEC_SCALAR_OP(vrnds, uint16_t, s32, v64)

VEC_SCALAR_OP(vsadds, int16_t, s16, v128)
#undef VEC_SCALAR_OP

//----------------------------------------------------------------------------//
//  pnot
//----------------------------------------------------------------------------//
CCE_INTRINSIC void pnot(vector_bool &dst, vector_bool src, vector_bool mask) {
  dst = __builtin_cce_pnot_z(src, mask);
  return;
}

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
//----------------------------------------------------------------------------//
//  vext
//----------------------------------------------------------------------------//
CCE_INTRINSIC void vext(vector_bool mask) {
  __builtin_cce_vext(mask);
  return;
}

//----------------------------------------------------------------------------//
//  vextfa
//----------------------------------------------------------------------------//
CCE_INTRINSIC void vextfa(vector_bool mask) {
  __builtin_cce_vextfa(mask);
  return;
}
#endif

//----------------------------------------------------------------------------//
//  vavg
//----------------------------------------------------------------------------//
enum class Rnd { NO_RND_VALUE, RND_VALUE };
typedef std::integral_constant<Rnd, Rnd::NO_RND_VALUE> NoRNDType;
typedef std::integral_constant<Rnd, Rnd::RND_VALUE> RNDType;
#define NO_RND NoRNDType()
#define RND RNDType()

#define VAVG(TYPE, NUM)                                                        \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void vavg(vector_##TYPE &dst, vector_##TYPE src0,              \
                          vector_##TYPE src1, T1 rnd, vector_bool mask,        \
                          T2 mode = MODE_UNKNOWN) {                            \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the 6th argument is not valid");  \
    static_assert(std::is_same<T1, NoRNDType>::value ||                        \
                      std::is_same<T1, RNDType>::value,                        \
                  "The 4th argument of this vavg can only be: NO_RND, RND");   \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      vector_##TYPE dstTmp = __builtin_cce_vavg_##NUM##TYPE##_x(               \
          src0, src1, (ULL)rnd.value, mask);                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vmov_##NUM##TYPE##_m(dst, dstTmp, mask)        \
                : dstTmp;                                                      \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vavg_##NUM##TYPE##_m(dst, src0, src1,          \
                                                     (ULL)rnd.value, mask)     \
                : __builtin_cce_vavg_##NUM##TYPE##_x(src0, src1,               \
                                                     (ULL)rnd.value, mask);    \
    }                                                                          \
    return;                                                                    \
  }

VAVG(s16, v128)
VAVG(s8, v256)
VAVG(u16, v128)
VAVG(u8, v256)
#undef VAVG

//----------------------------------------------------------------------------//
//  vcgadd/vcgmax/vcgmin/vcpadd
//----------------------------------------------------------------------------//
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
#define VCG_P_OP(OP_NAME, TYPE, NUM)                                           \
  template <class Tm = Mode_Unknown_Type>                                      \
  CCE_INTRINSIC void OP_NAME(vector_##TYPE &dst, vector_##TYPE src,            \
                             vector_bool pg, Tm mode = MODE_UNKNOWN) {         \
    static_assert(                                                             \
        mode.value == MODE_ZEROING.value | mode.value == MODE_UNKNOWN.value,   \
        "The last argument can only be 'MODE_ZEROING' or MODE_UNKNOWN.");      \
    dst = __builtin_cce_##OP_NAME##_##NUM##TYPE##_x(src, pg);                  \
    return;                                                                    \
  }

VCG_P_OP(vcgadd, f16, v128)
VCG_P_OP(vcgadd, f32, v64)

VCG_P_OP(vcgmax, f16, v128)
VCG_P_OP(vcgmax, f32, v64)

VCG_P_OP(vcgmin, f16, v128)
VCG_P_OP(vcgmin, f32, v64)

VCG_P_OP(vcpadd, f16, v128)
VCG_P_OP(vcpadd, f32, v64)
#undef VCG_P_OP
#endif

//----------------------------------------------------------------------------//
//  pand, por, pxor
//----------------------------------------------------------------------------//
#define PREDICATE_Z(NAME)                                                      \
  CCE_INTRINSIC void NAME(vector_bool &dst, vector_bool src0,                  \
                          vector_bool src1, vector_bool mask) {                \
    dst = __builtin_cce_##NAME##_z(src0, src1, mask);                          \
    return;                                                                    \
  }

PREDICATE_Z(pand)
PREDICATE_Z(por)
PREDICATE_Z(pxor)

#undef PREDICATE_Z

//----------------------------------------------------------------------------//
//  pmov
//----------------------------------------------------------------------------//
CCE_INTRINSIC void pmov(vector_bool &dst, vector_bool src, vector_bool mask) {
  dst = __builtin_cce_pmov_z(src, mask);
  return;
}
CCE_INTRINSIC void pmov(vector_bool &dst, vector_bool src) {
  dst = __builtin_cce_pmov(src);
  return;
}

//----------------------------------------------------------------------------//
//  pslide
//----------------------------------------------------------------------------//
#define PSLIDE(TYPE)                                                           \
  CCE_INTRINSIC void pslide_##TYPE(vector_bool &dst, vector_bool src0,         \
                                   vector_bool src1, int16_t slideAmount) {    \
    dst = __builtin_cce_pslide_##TYPE(src0, src1, slideAmount);                \
    return;                                                                    \
  }

PSLIDE(b8)
PSLIDE(b16)
PSLIDE(b32)
#undef PSLIDE

//----------------------------------------------------------------------------//
//  pintlv
//----------------------------------------------------------------------------//
#define PINTLV(TYPE)                                                           \
  CCE_INTRINSIC void pintlv_##TYPE(vector_bool &dst0, vector_bool &dst1,       \
                                   vector_bool src0, vector_bool src1) {       \
    vector_boolx2_t __ret;                                                     \
    __builtin_cce_pintlv_##TYPE(&__ret, src0, src1);                           \
    dst0 = __ret.val[0];                                                       \
    dst1 = __ret.val[1];                                                       \
    return;                                                                    \
  }

PINTLV(b8)
PINTLV(b16)
PINTLV(b32)
#undef PINTLV

//----------------------------------------------------------------------------//
//  pdintlv
//----------------------------------------------------------------------------//
#define PDINTLV(TYPE)                                                          \
  CCE_INTRINSIC void pdintlv_##TYPE(vector_bool &dst0, vector_bool &dst1,      \
                                    vector_bool src0, vector_bool src1) {      \
    vector_boolx2_t __ret;                                                     \
    __builtin_cce_pdintlv_##TYPE(&__ret, src0, src1);                          \
    dst0 = __ret.val[0];                                                       \
    dst1 = __ret.val[1];                                                       \
    return;                                                                    \
  }

PDINTLV(b8)
PDINTLV(b16)
PDINTLV(b32)
#undef PDINTLV

//----------------------------------------------------------------------------//
//  psel
//----------------------------------------------------------------------------//
CCE_INTRINSIC void psel(vector_bool &dst, vector_bool src0, vector_bool src1,
                        vector_bool mask) {
  dst = __builtin_cce_psel(src0, src1, mask);
  return;
}

//----------------------------------------------------------------------------//
//  ppack
//----------------------------------------------------------------------------//
template <class T>
CCE_INTRINSIC void ppack(vector_bool &dst, vector_bool src, T part) {
  static_assert(std::is_class<T>::value, "the 3rd argument is not valid");
  static_assert(std::is_same<T, Lower_Type>::value ||
                    std::is_same<T, Higher_Type>::value,
                "The 3rd argument of this ppack can only be: LOWER, HIGHER");
  dst = __builtin_cce_ppack_z(src, (ULL)part.value);
  return;
}

//----------------------------------------------------------------------------//
//  punpack
//----------------------------------------------------------------------------//
template <class T>
CCE_INTRINSIC void punpack(vector_bool &dst, vector_bool src, T part) {
  static_assert(std::is_class<T>::value, "the 3rd argument is not valid");
  static_assert(std::is_same<T, Lower_Type>::value ||
                    std::is_same<T, Higher_Type>::value,
                "The 3rd argument of this punpack can only be: "
                "LOWER, HIGHER");
  dst = __builtin_cce_punpack(src, (ULL)part.value);
  return;
}

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
//----------------------------------------------------------------------------//
//  wdups
//----------------------------------------------------------------------------//
#define WDUPS(WTYPE, VTYPE, NUM, STYPE)                                        \
  CCE_INTRINSIC void wdups(wvector_##WTYPE &dst, STYPE src) {                  \
    dst = __builtin_cce_wdups_##NUM##VTYPE(src);                               \
    return;                                                                    \
  }

WDUPS(s24, u8, v256, uint8_t)
WDUPS(s24, s8, v256, int8_t)
WDUPS(s48, u16, v128, uint16_t)
WDUPS(s48, s16, v128, int16_t)
#undef WDUPS

//----------------------------------------------------------------------------//
//  wmov
//----------------------------------------------------------------------------//
#define WMOV(WTYPE, NUM)                                                       \
  template <class Tm = std::integral_constant<Mode, Mode::MERGING_VALUE>>      \
  CCE_INTRINSIC void wmov(wvector_##WTYPE &dst, wvector_##WTYPE src,           \
                          vector_bool mask, Tm mode = MODE_MERGING) {          \
    static_assert(mode.value == MODE_MERGING.value,                            \
                  "The last argument can only be 'MODE_MERGING' or empty.");   \
    dst = __builtin_cce_wmov_##NUM##WTYPE##_m(dst, src, mask);                 \
    return;                                                                    \
  }                                                                            \
  CCE_INTRINSIC void wmov(wvector_##WTYPE &dst, wvector_##WTYPE src) {         \
    dst = __builtin_cce_wmov_##NUM##WTYPE(src);                                \
    return;                                                                    \
  }

WMOV(s24, v256)
WMOV(s48, v128)
#undef WMOV
#endif

//----------------------------------------------------------------------------//
//  wadd
//----------------------------------------------------------------------------//
#define WIDE_BINARY_OP(OP_NAME, WTYPE, VTYPE, NUM)                             \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(wvector_##WTYPE &dst, vector_##VTYPE src0,        \
                             vector_##VTYPE src1, vector_bool mask,            \
                             T mode = MODE_UNKNOWN) {                          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      wvector_##WTYPE dstTmp =                                                 \
          __builtin_cce_##OP_NAME##_##NUM##VTYPE##_x(src0, src1, mask);        \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_wmov_##NUM##WTYPE##_m(dst, dstTmp, mask)       \
                : dstTmp;                                                      \
    } else {                                                                   \
      dst =                                                                    \
          (mode.value == MODE_MERGING.value)                                   \
              ? __builtin_cce_##OP_NAME##_##NUM##VTYPE##_m(dst, src0, src1,    \
                                                           mask)               \
              : __builtin_cce_##OP_NAME##_##NUM##VTYPE##_x(src0, src1, mask);  \
    }                                                                          \
    return;                                                                    \
  }

WIDE_BINARY_OP(wadd, s24, u8, v256)
WIDE_BINARY_OP(wadd, s48, u16, v128)
WIDE_BINARY_OP(wadd, s24, s8, v256)
WIDE_BINARY_OP(wadd, s48, s16, v128)

WIDE_BINARY_OP(wsub, s24, s8, v256)
WIDE_BINARY_OP(wsub, s48, s16, v128)
WIDE_BINARY_OP(wsub, s24, u8, v256)
WIDE_BINARY_OP(wsub, s48, u16, v128)
#undef WIDE_BINARY_OP

//----------------------------------------------------------------------------//
// wadda/wsuba/waddsub
//----------------------------------------------------------------------------//
#define WIDE_TERNARY_OP(OP_NAME, WTYPE, VTYPE, NUM)                            \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(wvector_##WTYPE &dst, vector_##VTYPE src0,        \
                             vector_##VTYPE src1, vector_bool mask,            \
                             T mode = MODE_UNKNOWN) {                          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        wvector_##WTYPE dstTmp = __builtin_cce_wmov_##NUM##WTYPE(dst);         \
        dstTmp =                                                               \
            __builtin_cce_##OP_NAME##_##NUM##VTYPE(dstTmp, src0, src1, mask);  \
        dst = __builtin_cce_wmov_##NUM##WTYPE##_m(dst, dstTmp, mask);          \
      } else {                                                                 \
        dst = __builtin_cce_##OP_NAME##_##NUM##VTYPE(dst, src0, src1, mask);   \
      }                                                                        \
    } else {                                                                   \
      dst = __builtin_cce_##OP_NAME##_##NUM##VTYPE(dst, src0, src1, mask);     \
    }                                                                          \
    return;                                                                    \
  }

WIDE_TERNARY_OP(wadda, s24, u8, v256)
WIDE_TERNARY_OP(wadda, s48, u16, v128)
WIDE_TERNARY_OP(wadda, s24, s8, v256)
WIDE_TERNARY_OP(wadda, s48, s16, v128)

WIDE_TERNARY_OP(wsuba, s24, u8, v256)
WIDE_TERNARY_OP(wsuba, s48, u16, v128)
WIDE_TERNARY_OP(wsuba, s24, s8, v256)
WIDE_TERNARY_OP(wsuba, s48, s16, v128)

WIDE_TERNARY_OP(waddsub, s24, s8, v256)
WIDE_TERNARY_OP(waddsub, s24, u8, v256)
WIDE_TERNARY_OP(waddsub, s48, s16, v128)
WIDE_TERNARY_OP(waddsub, s48, u16, v128)
#undef WIDE_TERNARY_OP

//----------------------------------------------------------------------------//
//  wmul
//----------------------------------------------------------------------------//
#define WMUL(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                                 \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void wmul(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,      \
                          vector_##SRC1TYPE src1, vector_bool mask,            \
                          T mode = MODE_UNKNOWN) {                             \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      wvector_##DSTTYPE dstTmp =                                               \
          __builtin_cce_wmul_##NUM##SRC0TYPE##SRC1TYPE##_x(src0, src1, mask);  \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_wmov_##NUM##DSTTYPE##_m(dst, dstTmp, mask)     \
                : dstTmp;                                                      \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_wmul_##NUM##SRC0TYPE##SRC1TYPE##_m(dst, src0,  \
                                                                   src1, mask) \
                : __builtin_cce_wmul_##NUM##SRC0TYPE##SRC1TYPE##_x(src0, src1, \
                                                                   mask);      \
    }                                                                          \
    return;                                                                    \
  }

WMUL(s24, u8, u8, v256)
WMUL(s24, u8, s8, v256)
WMUL(s24, s8, s8, v256)
WMUL(s48, u16, u16, v128)
WMUL(s48, u16, s16, v128)
WMUL(s48, s16, s16, v128)
#undef WMUL

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
//----------------------------------------------------------------------------//
//  Putting PART_EVEN and PART_ODD instrinsic together to implemet
//  a full element wmul
//----------------------------------------------------------------------------//
#define WMUL(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                                 \
  CCE_INTRINSIC void wmul(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,      \
                          vector_##SRC1TYPE##x2_t src1) {                      \
    vector_bool mask = pset_b8(PAT_M4);                                        \
    wvector_##DSTTYPE dstTmp =                                                 \
        __builtin_cce_wmul_##NUM##SRC0TYPE##SRC1TYPE##_v300(                   \
            src0, src1.val[0], (ULL)PART_EVEN.value);                          \
    dst = __builtin_cce_wmul_##NUM##SRC0TYPE##SRC1TYPE##_v300(                 \
        src0, src1.val[1], (ULL)PART_ODD.value);                               \
    dst = __builtin_cce_wmov_##NUM##DSTTYPE##_m(dst, dstTmp, mask);            \
    return;                                                                    \
  }

WMUL(s48, u16, u32, v128)
WMUL(s48, u16, s32, v128)
WMUL(s48, s16, u32, v128)
WMUL(s48, s16, s32, v128)
#undef WMUL

#define WMUL(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                                 \
  CCE_INTRINSIC void wmul(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,      \
                          vector_##SRC1TYPE##x2_t src1) {                      \
    dst = __builtin_cce_wmul_##NUM##SRC0TYPE##SRC1TYPE##_v300(                 \
        src0, src1.val[0], src1.val[1]);                                       \
    return;                                                                    \
  }

WMUL(s24, u8, s16, v256)
WMUL(s24, s8, s16, v256)
#undef WMUL
#endif

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
//----------------------------------------------------------------------------//
// For V210 we don't hide part parameter, V300 don't need to be compatible V210
//----------------------------------------------------------------------------//
#define WMUL_PART(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                            \
  template <class T1>                                                          \
  CCE_INTRINSIC void wmul(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,      \
                          vector_##SRC1TYPE src1, T1 part) {                   \
    static_assert(std::is_class<T1>::value, INVALID_VALUE_PART);               \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  INVALID_VALUE_PART);                                         \
    dst = __builtin_cce_wmul_##NUM##SRC0TYPE##SRC1TYPE##_x(dst, src0, src1,    \
                                                           (ULL)part.value);   \
    return;                                                                    \
  }

WMUL_PART(s24, u8, s16, v256)
WMUL_PART(s24, s8, s16, v256)
#undef WMUL_PART

#define WMUL_PART(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                            \
  template <class T1>                                                          \
  CCE_INTRINSIC void wmul(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,      \
                          vector_##SRC1TYPE src1, T1 part) {                   \
    static_assert(std::is_class<T1>::value, INVALID_VALUE_PART);               \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  INVALID_VALUE_PART);                                         \
    dst = __builtin_cce_wmul_##NUM##SRC0TYPE##SRC1TYPE##_x(src0, src1,         \
                                                           (ULL)part.value);   \
    return;                                                                    \
  }

WMUL_PART(s64, u16, u32, v128)
WMUL_PART(s64, u16, s32, v128)
WMUL_PART(s64, s16, u32, v128)
WMUL_PART(s64, s16, s32, v128)
#undef WMUL_PART
#endif

//----------------------------------------------------------------------------//
//  wmula
//----------------------------------------------------------------------------//
#define WMULA(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                                \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void wmula(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,     \
                           vector_##SRC1TYPE src1, vector_bool mask,           \
                           T mode = MODE_UNKNOWN) {                            \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        wvector_##DSTTYPE dstTmp = __builtin_cce_wmov_##NUM##DSTTYPE(dst);     \
        dstTmp = __builtin_cce_wmula_##NUM##SRC0TYPE##SRC1TYPE##_m(            \
            dstTmp, src0, src1, mask);                                         \
        dst = __builtin_cce_wmov_##NUM##DSTTYPE##_m(dst, dstTmp, mask);        \
      } else {                                                                 \
        dst = __builtin_cce_wmula_##NUM##SRC0TYPE##SRC1TYPE##_m(dst, src0,     \
                                                                src1, mask);   \
      }                                                                        \
    } else {                                                                   \
      dst = __builtin_cce_wmula_##NUM##SRC0TYPE##SRC1TYPE##_m(dst, src0, src1, \
                                                              mask);           \
    }                                                                          \
    return;                                                                    \
  }

WMULA(s24, u8, u8, v256)
WMULA(s24, u8, s8, v256)
WMULA(s24, s8, s8, v256)
WMULA(s48, u16, u16, v128)
WMULA(s48, u16, s16, v128)
WMULA(s48, s16, s16, v128)
#undef WMULA

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
//----------------------------------------------------------------------------//
//  Putting PART_EVEN and PART_ODD instrinsic together to implemet
//  a full element wmula
//----------------------------------------------------------------------------//
#define WMULA(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                                \
  CCE_INTRINSIC void wmula(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,     \
                           vector_##SRC1TYPE##x2_t src1) {                     \
    vector_bool mask = pset_b8(PAT_M4);                                        \
    wvector_##DSTTYPE dstTmp = __builtin_cce_wmov_##NUM##DSTTYPE(dst);         \
    dstTmp = __builtin_cce_wmula_##NUM##SRC0TYPE##SRC1TYPE##_v300(             \
        dstTmp, src0, src1.val[0], (ULL)PART_EVEN.value);                      \
    dst = __builtin_cce_wmula_##NUM##SRC0TYPE##SRC1TYPE##_v300(                \
        dst, src0, src1.val[1], (ULL)PART_ODD.value);                          \
    dst = __builtin_cce_wmov_##NUM##DSTTYPE##_m(dst, dstTmp, mask);            \
    return;                                                                    \
  }

WMULA(s48, u16, u32, v128)
WMULA(s48, u16, s32, v128)
WMULA(s48, s16, u32, v128)
WMULA(s48, s16, s32, v128)
#undef WMULA

#define WMULA(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                                \
  CCE_INTRINSIC void wmula(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,     \
                           vector_##SRC1TYPE##x2_t src1) {                     \
    dst = __builtin_cce_wmula_##NUM##SRC0TYPE##SRC1TYPE##_v300(                \
        dst, src0, src1.val[0], src1.val[1]);                                  \
    return;                                                                    \
  }

WMULA(s24, u8, s16, v256)
WMULA(s24, s8, s16, v256)
#undef WMULA
#endif

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
//----------------------------------------------------------------------------//
// For V210 we don't hide part parameter, V300 don't need to be compatible V210
//----------------------------------------------------------------------------//
#define WMULA_PART(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                           \
  template <class T>                                                           \
  CCE_INTRINSIC void wmula(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,     \
                           vector_##SRC1TYPE src1, T part) {                   \
    static_assert(std::is_class<T>::value, INVALID_VALUE_PART);                \
    static_assert(std::is_same<T, PartEvenType>::value ||                      \
                      std::is_same<T, PartOddType>::value,                     \
                  INVALID_VALUE_PART);                                         \
    dst = __builtin_cce_wmula_##NUM##SRC0TYPE##SRC1TYPE##_m(dst, src0, src1,   \
                                                            (ULL)part.value);  \
    return;                                                                    \
  }

WMULA_PART(s24, u8, s16, v256)
WMULA_PART(s24, s8, s16, v256)
WMULA_PART(s64, u16, u32, v128)
WMULA_PART(s64, u16, s32, v128)
WMULA_PART(s64, s16, u32, v128)
WMULA_PART(s64, s16, s32, v128)
#undef WMULA_PART
#endif

//----------------------------------------------------------------------------//
//  wmuls
//----------------------------------------------------------------------------//
#define WMULS(DSTTYPE, SRC0TYPE, SRC1TYPE, SCALAR, NUM)                        \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void wmuls(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,     \
                           SCALAR src1, vector_bool mask,                      \
                           T mode = MODE_UNKNOWN) {                            \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      wvector_##DSTTYPE dstTmp =                                               \
          __builtin_cce_wmuls_##NUM##SRC0TYPE##SRC1TYPE##_x(src0, src1, mask); \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_wmov_##NUM##DSTTYPE##_m(dst, dstTmp, mask)     \
                : dstTmp;                                                      \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_wmuls_##NUM##SRC0TYPE##SRC1TYPE##_m(           \
                      dst, src0, src1, mask)                                   \
                : __builtin_cce_wmuls_##NUM##SRC0TYPE##SRC1TYPE##_x(           \
                      src0, src1, mask);                                       \
    }                                                                          \
    return;                                                                    \
  }
WMULS(s24, u8, u8, uint8_t, v256)
WMULS(s24, u8, s8, int8_t, v256)
WMULS(s24, s8, s8, int8_t, v256)
WMULS(s48, u16, u16, uint16_t, v128)
WMULS(s48, u16, s16, int16_t, v128)
WMULS(s48, s16, s16, int16_t, v128)
#undef WMULS

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
//----------------------------------------------------------------------------//
//  Putting PART_EVEN and PART_ODD instrinsic together to implemet
//  a full element wmuls
//----------------------------------------------------------------------------//
#define WMULS(DSTTYPE, SRC0TYPE, SRC1TYPE, SCALAR, NUM)                        \
  CCE_INTRINSIC void wmuls(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,     \
                           SCALAR src1) {                                      \
    vector_bool mask = pset_b8(PAT_M4);                                        \
    wvector_##DSTTYPE dstTmp =                                                 \
        __builtin_cce_wmuls_##NUM##SRC0TYPE##SRC1TYPE##_v300(                  \
            src0, src1, (ULL)PART_EVEN.value);                                 \
    dst = __builtin_cce_wmuls_##NUM##SRC0TYPE##SRC1TYPE##_v300(                \
        src0, src1, (ULL)PART_ODD.value);                                      \
    dst = __builtin_cce_wmov_##NUM##DSTTYPE##_m(dst, dstTmp, mask);            \
    return;                                                                    \
  }

WMULS(s48, u16, u32, uint32_t, v128)
WMULS(s48, u16, s32, int32_t, v128)
WMULS(s48, s16, u32, uint32_t, v128)
WMULS(s48, s16, s32, int32_t, v128)
#undef WMULS

#define WMULS(DSTTYPE, SRC0TYPE, SRC1TYPE, SCALAR, NUM)                        \
  CCE_INTRINSIC void wmuls(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,     \
                           SCALAR src1) {                                      \
    dst = __builtin_cce_wmuls_##NUM##SRC0TYPE##SRC1TYPE##_v300(src0, src1);    \
    return;                                                                    \
  }

WMULS(s24, u8, s16, int16_t, v256)
WMULS(s24, s8, s16, int16_t, v256)
#undef WMULS
#endif

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
//----------------------------------------------------------------------------//
// For V210 we don't hide part parameter, V300 don't need to be compatible V210
//----------------------------------------------------------------------------//
#define WMULS_PART(DSTTYPE, SRC0TYPE, SRC1TYPE, SCALAR, NUM)                   \
  template <class T1>                                                          \
  CCE_INTRINSIC void wmuls(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,     \
                           SCALAR src1, T1 part) {                             \
    static_assert(std::is_class<T1>::value, INVALID_VALUE_PART);               \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  INVALID_VALUE_PART);                                         \
    dst = __builtin_cce_wmuls_##NUM##SRC0TYPE##SRC1TYPE##_x(dst, src0, src1,   \
                                                            (ULL)part.value);  \
    return;                                                                    \
  }

WMULS_PART(s24, u8, s16, int16_t, v256)
WMULS_PART(s24, s8, s16, int16_t, v256)
#undef WMULS_PART

#define WMULS_PART(DSTTYPE, SRC0TYPE, SRC1TYPE, SCALAR, NUM)                   \
  template <class T1>                                                          \
  CCE_INTRINSIC void wmuls(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,     \
                           SCALAR src1, T1 part) {                             \
    static_assert(std::is_class<T1>::value, INVALID_VALUE_PART);               \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  INVALID_VALUE_PART);                                         \
    dst = __builtin_cce_wmuls_##NUM##SRC0TYPE##SRC1TYPE##_x(src0, src1,        \
                                                            (ULL)part.value);  \
    return;                                                                    \
  }

WMULS_PART(s64, u16, u32, uint32_t, v128)
WMULS_PART(s64, u16, s32, int32_t, v128)
WMULS_PART(s64, s16, u32, uint32_t, v128)
WMULS_PART(s64, s16, s32, int32_t, v128)
#undef WMULS_PART
#endif

//----------------------------------------------------------------------------//
//  wmulas
//----------------------------------------------------------------------------//
#define WMULAS(DSTTYPE, SRC0TYPE, SRC1TYPE, SCALAR, NUM)                       \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void wmulas(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,    \
                            SCALAR src1, vector_bool mask,                     \
                            T mode = MODE_UNKNOWN) {                           \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      if (mode.value == MODE_MERGING.value) {                                  \
        wvector_##DSTTYPE dstTmp = __builtin_cce_wmov_##NUM##DSTTYPE(dst);     \
        dstTmp = __builtin_cce_wmulas_##NUM##SRC0TYPE##SRC1TYPE##_m(           \
            dstTmp, src0, src1, mask);                                         \
        dst = __builtin_cce_wmov_##NUM##DSTTYPE##_m(dst, dstTmp, mask);        \
      } else {                                                                 \
        dst = __builtin_cce_wmulas_##NUM##SRC0TYPE##SRC1TYPE##_m(dst, src0,    \
                                                                 src1, mask);  \
      }                                                                        \
    } else {                                                                   \
      dst = __builtin_cce_wmulas_##NUM##SRC0TYPE##SRC1TYPE##_m(dst, src0,      \
                                                               src1, mask);    \
    }                                                                          \
    return;                                                                    \
  }

WMULAS(s24, u8, u8, uint8_t, v256)
WMULAS(s24, u8, s8, int8_t, v256)
WMULAS(s24, s8, s8, int8_t, v256)
WMULAS(s48, u16, u16, uint16_t, v128)
WMULAS(s48, u16, s16, int16_t, v128)
WMULAS(s48, s16, s16, int16_t, v128)
#undef WMULAS

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
//----------------------------------------------------------------------------//
//  Putting PART_EVEN and PART_ODD instrinsic together to implemet
//  a full element wmulas
//----------------------------------------------------------------------------//
#define WMULAS(DSTTYPE, SRC0TYPE, SRC1TYPE, SCALAR, NUM)                       \
  CCE_INTRINSIC void wmulas(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,    \
                            SCALAR src1) {                                     \
    vector_bool mask = pset_b8(PAT_M4);                                        \
    wvector_##DSTTYPE dstTmp = __builtin_cce_wmov_##NUM##DSTTYPE(dst);         \
    dstTmp = __builtin_cce_wmulas_##NUM##SRC0TYPE##SRC1TYPE##_v300(            \
        dstTmp, src0, src1, (ULL)PART_EVEN.value);                             \
    dst = __builtin_cce_wmulas_##NUM##SRC0TYPE##SRC1TYPE##_v300(               \
        dst, src0, src1, (ULL)PART_ODD.value);                                 \
    dst = __builtin_cce_wmov_##NUM##DSTTYPE##_m(dst, dstTmp, mask);            \
    return;                                                                    \
  }

WMULAS(s48, u16, u32, uint32_t, v128)
WMULAS(s48, u16, s32, int32_t, v128)
WMULAS(s48, s16, u32, uint32_t, v128)
WMULAS(s48, s16, s32, int32_t, v128)
#undef WMULAS

#define WMULAS(DSTTYPE, SRC0TYPE, SRC1TYPE, SCALAR, NUM)                       \
  CCE_INTRINSIC void wmulas(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,    \
                            SCALAR src1) {                                     \
    dst = __builtin_cce_wmulas_##NUM##SRC0TYPE##SRC1TYPE##_v300(dst, src0,     \
                                                                src1);         \
    return;                                                                    \
  }

WMULAS(s24, u8, s16, int16_t, v256)
WMULAS(s24, s8, s16, int16_t, v256)
#undef WMULAS
#endif

#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
//----------------------------------------------------------------------------//
// For V210 we don't hide part parameter, V300 don't need to be compatible V210
//----------------------------------------------------------------------------//
#define WMULAS_PART(DSTTYPE, SRC0TYPE, SRC1TYPE, SCALAR, NUM)                  \
  template <class T>                                                           \
  CCE_INTRINSIC void wmulas(wvector_##DSTTYPE &dst, vector_##SRC0TYPE src0,    \
                            SCALAR src1, T part) {                             \
    static_assert(std::is_class<T>::value, INVALID_VALUE_PART);                \
    static_assert(std::is_same<T, PartEvenType>::value ||                      \
                      std::is_same<T, PartOddType>::value,                     \
                  INVALID_VALUE_PART);                                         \
    dst = __builtin_cce_wmulas_##NUM##SRC0TYPE##SRC1TYPE##_m(dst, src0, src1,  \
                                                             (ULL)part.value); \
    return;                                                                    \
  }
WMULAS_PART(s24, u8, s16, int16_t, v256)
WMULAS_PART(s24, s8, s16, int16_t, v256)
WMULAS_PART(s64, u16, u32, uint32_t, v128)
WMULAS_PART(s64, u16, s32, int32_t, v128)
WMULAS_PART(s64, s16, u32, uint32_t, v128)
WMULAS_PART(s64, s16, s32, int32_t, v128)
#undef WMULAS_PART
#endif

//----------------------------------------------------------------------------//
//  wpack
//----------------------------------------------------------------------------//
#define INVALID_VALUE_RS "The value of RS is not valid"
#define WPACK(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                                \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void wpack(vector_##DSTTYPE &dst, wvector_##SRC0TYPE src0,     \
                           vector_##SRC1TYPE src1, T1 rs, vector_bool mask,    \
                           T2 mode = MODE_UNKNOWN) {                           \
    static_assert(std::is_same<T1, RSDisableType>::value ||                    \
                      std::is_same<T1, RSEnableType>::value,                   \
                  INVALID_VALUE_RS);                                           \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      vector_##DSTTYPE dstTmp;                                                 \
      if (rs.value == RS_DISABLE.value) {                                      \
        dstTmp = __builtin_cce_wpack_rs0_##NUM##SRC0TYPE##2##DSTTYPE##_x(      \
            src0, src1, (ULL)rs.value, mask);                                  \
      } else if (rs.value == RS_ENABLE.value) {                                \
        dstTmp = __builtin_cce_wpack_rs1_##NUM##SRC0TYPE##2##DSTTYPE##_x(      \
            src0, src1, (ULL)rs.value, mask);                                  \
      }                                                                        \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vmov_##NUM##DSTTYPE##_m(dst, dstTmp, mask)     \
                : dstTmp;                                                      \
    } else {                                                                   \
      if (rs.value == RS_DISABLE.value) {                                      \
        dst = (mode.value == MODE_MERGING.value)                               \
                  ? __builtin_cce_wpack_rs0_##NUM##SRC0TYPE##2##DSTTYPE##_m(   \
                        dst, src0, src1, (ULL)rs.value, mask)                  \
                  : __builtin_cce_wpack_rs0_##NUM##SRC0TYPE##2##DSTTYPE##_x(   \
                        src0, src1, (ULL)rs.value, mask);                      \
      } else if (rs.value == RS_ENABLE.value) {                                \
        dst = (mode.value == MODE_MERGING.value)                               \
                  ? __builtin_cce_wpack_rs1_##NUM##SRC0TYPE##2##DSTTYPE##_m(   \
                        dst, src0, src1, (ULL)rs.value, mask)                  \
                  : __builtin_cce_wpack_rs1_##NUM##SRC0TYPE##2##DSTTYPE##_x(   \
                        src0, src1, (ULL)rs.value, mask);                      \
      }                                                                        \
    }                                                                          \
    return;                                                                    \
  }                                                                            \
                                                                               \
  template <class T>                                                           \
  CCE_INTRINSIC void wpack(vector_##DSTTYPE &dst, wvector_##SRC0TYPE src0,     \
                           vector_##SRC1TYPE src1, T rs) {                     \
    static_assert(std::is_same<T, RSDisableType>::value ||                     \
                      std::is_same<T, RSEnableType>::value,                    \
                  INVALID_VALUE_RS);                                           \
    if (rs.value == RS_DISABLE.value) {                                        \
      dst = __builtin_cce_wpack_rs0_##NUM##SRC0TYPE##2##DSTTYPE##_u(           \
          src0, src1, (ULL)rs.value);                                          \
    } else if (rs.value == RS_ENABLE.value) {                                  \
      dst = __builtin_cce_wpack_rs1_##NUM##SRC0TYPE##2##DSTTYPE##_u(           \
          src0, src1, (ULL)rs.value);                                          \
    }                                                                          \
    return;                                                                    \
  }

WPACK(s8, s24, u8, v256)
WPACK(u8, s24, u8, v256)
WPACK(s16, s48, u16, v128)
WPACK(u16, s48, u16, v128)
#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
WPACK(s32, s64, u32, v64)
#endif
#undef WPACK

#define WPACK_PART(DSTTYPE, SRC0TYPE, SRC1TYPE, NUM)                           \
  template <class T1, class T2>                                                \
  CCE_INTRINSIC void wpack(vector_##DSTTYPE &dst, wvector_##SRC0TYPE src0,     \
                           vector_##SRC1TYPE src1, T1 rs, T2 part) {           \
    static_assert(std::is_same<T1, RSDisableType>::value ||                    \
                      std::is_same<T1, RSEnableType>::value,                   \
                  INVALID_VALUE_RS);                                           \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  INVALID_VALUE_PART);                                         \
    if (rs.value == RS_DISABLE.value) {                                        \
      dst = __builtin_cce_wpack_rs0_##NUM##SRC0TYPE##2##DSTTYPE##_u(           \
          src0, src1, (ULL)rs.value, (ULL)part.value);                         \
    } else if (rs.value == RS_ENABLE.value) {                                  \
      dst = __builtin_cce_wpack_rs1_##NUM##SRC0TYPE##2##DSTTYPE##_u(           \
          src0, src1, (ULL)rs.value, (ULL)part.value);                         \
    }                                                                          \
    return;                                                                    \
  }

WPACK_PART(s16, s24, u16, v256)
WPACK_PART(u16, s24, u16, v256)
WPACK_PART(s32, s48, u32, v128)
#undef WPACK_PART

//----------------------------------------------------------------------------//
//  wpacks
//----------------------------------------------------------------------------//
#define WPACKS(DSTTYPE, SRCTYPE, NUM)                                          \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void wpacks(vector_##DSTTYPE &dst, wvector_##SRCTYPE src0,     \
                            uint16_t src1, T1 rs, vector_bool mask,            \
                            T2 mode = MODE_UNKNOWN) {                          \
    static_assert(std::is_same<T1, RSDisableType>::value ||                    \
                      std::is_same<T1, RSEnableType>::value,                   \
                  INVALID_VALUE_RS);                                           \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      vector_##DSTTYPE dstTmp;                                                 \
      if (rs.value == RS_DISABLE.value) {                                      \
        dstTmp = __builtin_cce_wpacks_rs0_##NUM##SRCTYPE##2##DSTTYPE##_x(      \
            src0, src1, (ULL)rs.value, mask);                                  \
      } else if (rs.value == RS_ENABLE.value) {                                \
        dstTmp = __builtin_cce_wpacks_rs1_##NUM##SRCTYPE##2##DSTTYPE##_x(      \
            src0, src1, (ULL)rs.value, mask);                                  \
      }                                                                        \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_vmov_##NUM##DSTTYPE##_m(dst, dstTmp, mask)     \
                : dstTmp;                                                      \
    } else {                                                                   \
      if (rs.value == RS_DISABLE.value) {                                      \
        dst = (mode.value == MODE_MERGING.value)                               \
                  ? __builtin_cce_wpacks_rs0_##NUM##SRCTYPE##2##DSTTYPE##_m(   \
                        dst, src0, src1, (ULL)rs.value, mask)                  \
                  : __builtin_cce_wpacks_rs0_##NUM##SRCTYPE##2##DSTTYPE##_x(   \
                        src0, src1, (ULL)rs.value, mask);                      \
      } else if (rs.value == RS_ENABLE.value) {                                \
        dst = (mode.value == MODE_MERGING.value)                               \
                  ? __builtin_cce_wpacks_rs1_##NUM##SRCTYPE##2##DSTTYPE##_m(   \
                        dst, src0, src1, (ULL)rs.value, mask)                  \
                  : __builtin_cce_wpacks_rs1_##NUM##SRCTYPE##2##DSTTYPE##_x(   \
                        src0, src1, (ULL)rs.value, mask);                      \
      }                                                                        \
    }                                                                          \
    return;                                                                    \
  }                                                                            \
                                                                               \
  template <class T>                                                           \
  CCE_INTRINSIC void wpacks(vector_##DSTTYPE &dst, wvector_##SRCTYPE src0,     \
                            uint16_t src1, T rs) {                             \
    static_assert(std::is_same<T, RSDisableType>::value ||                     \
                      std::is_same<T, RSEnableType>::value,                    \
                  INVALID_VALUE_RS);                                           \
    if (rs.value == RS_DISABLE.value) {                                        \
      dst = __builtin_cce_wpacks_rs0_##NUM##SRCTYPE##2##DSTTYPE##_u(           \
          src0, src1, (ULL)rs.value);                                          \
    } else if (rs.value == RS_ENABLE.value) {                                  \
      dst = __builtin_cce_wpacks_rs1_##NUM##SRCTYPE##2##DSTTYPE##_u(           \
          src0, src1, (ULL)rs.value);                                          \
    }                                                                          \
    return;                                                                    \
  }

WPACKS(s8, s24, v256)
WPACKS(u8, s24, v256)
WPACKS(s16, s48, v128)
WPACKS(u16, s48, v128)
#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
WPACKS(s32, s64, v64)
#endif
#undef WPACKS

#define WPACKS_PART(DSTTYPE, SRCTYPE, NUM)                                     \
  template <class T1, class T2>                                                \
  CCE_INTRINSIC void wpacks(vector_##DSTTYPE &dst, wvector_##SRCTYPE src0,     \
                            uint16_t src1, T1 rs, T2 part) {                   \
    static_assert(std::is_same<T1, RSDisableType>::value ||                    \
                      std::is_same<T1, RSEnableType>::value,                   \
                  INVALID_VALUE_RS);                                           \
    static_assert(std::is_same<T2, PartEvenType>::value ||                     \
                      std::is_same<T2, PartOddType>::value,                    \
                  INVALID_VALUE_PART);                                         \
    if (rs.value == RS_DISABLE.value) {                                        \
      dst = __builtin_cce_wpacks_rs0_##NUM##SRCTYPE##2##DSTTYPE##_u(           \
          src0, src1, (ULL)rs.value, (ULL)part.value);                         \
    } else if (rs.value == RS_ENABLE.value) {                                  \
      dst = __builtin_cce_wpacks_rs1_##NUM##SRCTYPE##2##DSTTYPE##_u(           \
          src0, src1, (ULL)rs.value, (ULL)part.value);                         \
    }                                                                          \
    return;                                                                    \
  }

WPACKS_PART(s16, s24, v256)
WPACKS_PART(u16, s24, v256)
WPACKS_PART(s32, s48, v128)
#undef WPACKS_PART

//----------------------------------------------------------------------------//
//  wcvt48
//----------------------------------------------------------------------------//
#define WCVT48(SRC0TYPE, SRC1TYPE, NUM)                                        \
  template <class T1, class T2 = Mode_Unknown_Type>                            \
  CCE_INTRINSIC void wcvt48(wvector_s48 &dst, vector_##SRC0TYPE src0,          \
                            vector_##SRC1TYPE src1, T1 part,                   \
                            T2 mode = MODE_UNKNOWN) {                          \
    static_assert(std::is_class<T1>::value, "the 4th argument is not valid");  \
    static_assert(std::is_class<T2>::value, "the last argument is not valid"); \
    static_assert(std::is_same<T1, PartEvenType>::value ||                     \
                      std::is_same<T1, PartOddType>::value,                    \
                  "The 4th argument of this wcvt48 can only be: "              \
                  "PART_EVEN, PART_ODD");                                      \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    static_assert(!(mode.value == MODE_ZEROING.value && isV210Target()),       \
                  INVALID_VALUE_V210_MODE);                                    \
    if (isSoftwareMergeMode()) {                                               \
      wvector_s48 dstTmp =                                                     \
          __builtin_cce_wcvt48_##NUM##s48_x(src0, src1, (ULL)part.value);      \
      if (mode.value == MODE_MERGING.value) {                                  \
        vector_bool mask = pge_b8(PAT_M4);                                     \
        if (part.value == Part::ODD) {                                         \
          vector_bool mask1 = pge_b8(PAT_ALL);                                 \
          mask = __builtin_cce_pnot_z(mask, mask1);                            \
        }                                                                      \
        dst = __builtin_cce_wmov_v128s48_m(dst, dstTmp, mask);                 \
      } else {                                                                 \
        dst = dstTmp;                                                          \
      }                                                                        \
    } else {                                                                   \
      dst = (mode.value == MODE_MERGING.value)                                 \
                ? __builtin_cce_wcvt48_##NUM##s48_m(dst, src0, src1,           \
                                                    (ULL)part.value)           \
                : __builtin_cce_wcvt48_##NUM##s48_x(src0, src1,                \
                                                    (ULL)part.value);          \
    }                                                                          \
    return;                                                                    \
  }
WCVT48(s32, s16, v128)
#undef WCVT48

//----------------------------------------------------------------------------//
//  wfifr2/wfifr2a/wfifr2s
// ISA : 8-bit and 16-bit data types have the same data parallel: VL_16.
// It means:
// 1) For u8/s8 the dst is s48 just like u16/s16.
// And for b16, VL_16(128) elements is in Vn and another 4 elements is in Vn+1,
// and for b8, VL_16 + 4 elements can be stored in one Vn, so do not need Vn+1,
// but Vn still must be even because hardware still will read Vn+1(but won't use
// the value). So in this we can give it the same argument of Vn.
//----------------------------------------------------------------------------//
#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
#define WFIFR2_B16(DSTTYPE, SRCTYPE, NUM)                                      \
  CCE_INTRINSIC void wfifr2(wvector_##DSTTYPE &dst, vector_##SRCTYPE src0,     \
                            vector_##SRCTYPE src1, uint32_t scalar) {          \
    dst = __builtin_cce_wfifr2_##NUM##SRCTYPE(src0, src1, scalar);             \
    return;                                                                    \
  }

WFIFR2_B16(s48, s16, v128)
WFIFR2_B16(s48, u16, v128)
#undef WFIFR2_B16

#define WFIFR2_B8(DSTTYPE, SRCTYPE, NUM)                                       \
  CCE_INTRINSIC void wfifr2(wvector_##DSTTYPE &dst, vector_##SRCTYPE src,      \
                            uint32_t scalar) {                                 \
    dst = __builtin_cce_wfifr2_##NUM##SRCTYPE(src, src, scalar);               \
    return;                                                                    \
  }

WFIFR2_B8(s48, s8, v256)
WFIFR2_B8(s48, u8, v256)
#undef WFIFR2_B8

#define WFIFR2A_B16(DSTTYPE, SRCTYPE, NUM)                                     \
  CCE_INTRINSIC void wfifr2a(wvector_##DSTTYPE &dst, vector_##SRCTYPE src0,    \
                             vector_##SRCTYPE src1, uint32_t scalar) {         \
    dst = __builtin_cce_wfifr2a_##NUM##SRCTYPE(dst, src0, src1, scalar);       \
    return;                                                                    \
  }

WFIFR2A_B16(s48, s16, v128)
WFIFR2A_B16(s48, u16, v128)
#undef WFIFR2A_B16

#define WFIFR2A_B8(DSTTYPE, SRCTYPE, NUM)                                      \
  CCE_INTRINSIC void wfifr2a(wvector_##DSTTYPE &dst, vector_##SRCTYPE src,     \
                             uint32_t scalar) {                                \
    dst = __builtin_cce_wfifr2a_##NUM##SRCTYPE(dst, src, src, scalar);         \
    return;                                                                    \
  }

WFIFR2A_B8(s48, s8, v256)
WFIFR2A_B8(s48, u8, v256)
#undef WFIFR2A_B8

#define WFIFR2S_B16(DSTTYPE, SRCTYPE, NUM)                                     \
  CCE_INTRINSIC void wfifr2s(wvector_##DSTTYPE &dst, vector_##SRCTYPE src0,    \
                             vector_##SRCTYPE src1, uint32_t scalar) {         \
    dst = __builtin_cce_wfifr2s_##NUM##SRCTYPE(dst, src0, src1, scalar);       \
    return;                                                                    \
  }

WFIFR2S_B16(s48, s16, v128)
WFIFR2S_B16(s48, u16, v128)
#undef WFIFR2S_B16

#define WFIFR2S_B8(DSTTYPE, SRCTYPE, NUM)                                      \
  CCE_INTRINSIC void wfifr2s(wvector_##DSTTYPE &dst, vector_##SRCTYPE src,     \
                             uint32_t scalar) {                                \
    dst = __builtin_cce_wfifr2s_##NUM##SRCTYPE(dst, src, src, scalar);         \
    return;                                                                    \
  }

WFIFR2S_B8(s48, s8, v256)
WFIFR2S_B8(s48, u8, v256)
#undef WFIFR2S_B8
#endif

//----------------------------------------------------------------------------//
//  vag, in unit of element
//----------------------------------------------------------------------------//
#if defined(__DAV_L210__) || defined(__DAV_M210_VEC__) || defined(__DAV_T210__)
CCE_INTRINSIC vector_address vag_b32(uint16_t in1, uint16_t in2 = 0,
                                     uint16_t in3 = 0, uint16_t in4 = 0) {
  // The size of uint32_t, int32_t, float is 4
  return __builtin_cce_vag_v210(in4 * 4, in3 * 4, in2 * 4, in1 * 4,
                                /* loop */ 0);
}

CCE_INTRINSIC vector_address vag_b16(uint16_t in1, uint16_t in2 = 0,
                                     uint16_t in3 = 0, uint16_t in4 = 0) {
  // The size of uint16_t, int16_t, half is 2
  return __builtin_cce_vag_v210(in4 * 2, in3 * 2, in2 * 2, in1 * 2,
                                /* loop */ 0);
}

CCE_INTRINSIC vector_address vag_b8(uint16_t in1, uint16_t in2 = 0,
                                    uint16_t in3 = 0, uint16_t in4 = 0) {
  return __builtin_cce_vag_v210(in4, in3, in2, in1, /* loop */ 0);
}
#endif

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
CCE_INTRINSIC vector_address vag_b32(uint16_t in1) {
  // The size of uint32_t, int32_t, float is 4
  return __builtin_cce_vag_1_layer_v300(in1 * 4);
}
CCE_INTRINSIC vector_address vag_b32(uint16_t in1, uint16_t in2) {
  // The size of uint32_t, int32_t, float is 4
  return __builtin_cce_vag_2_layer_v300(in1 * 4, in2 * 4);
}
CCE_INTRINSIC vector_address vag_b32(uint16_t in1, uint16_t in2, uint16_t in3) {
  // The size of uint32_t, int32_t, float is 4
  return __builtin_cce_vag_3_layer_v300(in1 * 4, in2 * 4, in3 * 4);
}
CCE_INTRINSIC vector_address vag_b32(uint16_t in1, uint16_t in2, uint16_t in3,
                                     uint16_t in4) {
  // The size of uint32_t, int32_t, float is 4
  return __builtin_cce_vag_4_layer_v300(in1 * 4, in2 * 4, in3 * 4, in4 * 4);
}

CCE_INTRINSIC vector_address vag_b16(uint16_t in1) {
  // The size of uint16_t, int16_t, half is 2
  return __builtin_cce_vag_1_layer_v300(in1 * 2);
}
CCE_INTRINSIC vector_address vag_b16(uint16_t in1, uint16_t in2) {
  // The size of uint16_t, int16_t, half is 2
  return __builtin_cce_vag_2_layer_v300(in1 * 2, in2 * 2);
}
CCE_INTRINSIC vector_address vag_b16(uint16_t in1, uint16_t in2, uint16_t in3) {
  // The size of uint16_t, int16_t, half is 2
  return __builtin_cce_vag_3_layer_v300(in1 * 2, in2 * 2, in3 * 2);
}
CCE_INTRINSIC vector_address vag_b16(uint16_t in1, uint16_t in2, uint16_t in3,
                                     uint16_t in4) {
  // The size of uint16_t, int16_t, half is 2
  return __builtin_cce_vag_4_layer_v300(in1 * 2, in2 * 2, in3 * 2, in4 * 2);
}

CCE_INTRINSIC vector_address vag_b8(uint16_t in1) {
  // The size of uint8_t, int8_t is 1
  return __builtin_cce_vag_1_layer_v300(in1);
}
CCE_INTRINSIC vector_address vag_b8(uint16_t in1, uint16_t in2) {
  // The size of uint8_t, int8_t is 1
  return __builtin_cce_vag_2_layer_v300(in1, in2);
}
CCE_INTRINSIC vector_address vag_b8(uint16_t in1, uint16_t in2, uint16_t in3) {
  // The size of uint8_t, int8_t is 1
  return __builtin_cce_vag_3_layer_v300(in1, in2, in3);
}
CCE_INTRINSIC vector_address vag_b8(uint16_t in1, uint16_t in2, uint16_t in3,
                                    uint16_t in4) {
  // The size of uint8_t, int8_t is 1
  return __builtin_cce_vag_4_layer_v300(in1, in2, in3, in4);
}

#endif

//----------------------------------------------------------------------------//
//  vbr
//----------------------------------------------------------------------------//
CCE_INTRINSIC void vbr(vector_u8 &dst, uint8_t val) {
  dst = vbr_u8(val, 0);
  return;
}

CCE_INTRINSIC void vbr(vector_s8 &dst, int8_t val) {
  dst = vbr_s8(val, 0);
  return;
}

CCE_INTRINSIC void vbr(vector_u16 &dst, uint16_t val) {
  dst = vbr_u16(val, 0);
  return;
}

CCE_INTRINSIC void vbr(vector_s16 &dst, int16_t val) {
  dst = vbr_s16(val, 0);
  return;
}

CCE_INTRINSIC void vbr(vector_s32 &dst, int32_t val) {
  dst = vbr_s32(val, 0);
  return;
}

CCE_INTRINSIC void vbr(vector_u32 &dst, uint32_t val) {
  dst = vbr_u32(val, 0);
  return;
}

CCE_INTRINSIC void vbr(vector_f16 &dst, half val) {
  dst = vbr_f16(val, 0);
  return;
}

CCE_INTRINSIC void vbr(vector_f32 &dst, float val) {
  dst = vbr_f32(val, 0);
  return;
}

//----------------------------------------------------------------------------//
//  vnop
//----------------------------------------------------------------------------//
CCE_INTRINSIC void vnop() { return __builtin_cce_vnop(); }

//----------------------------------------------------------------------------//
//  vnopxn (check constant argument range in func CheckHiIPUBuiltinFunctionCall)
//----------------------------------------------------------------------------//
#define vnopxn __builtin_cce_vnopxn

//----------------------------------------------------------------------------//
//  mem_bar
//----------------------------------------------------------------------------//
enum class MEM_TYPE {
  VV_ALL,
  VST_VLD,
  VLD_VST,
  VST_VST,
  VS_ALL,
  VST_LD,
  VLD_ST,
  VST_ST,
  SV_ALL,
  ST_VLD,
  LD_VST,
  ST_VST,
  SS_ALL,
  ST_LD,
  LD_ST,
  ST_ST
};

typedef std::integral_constant<MEM_TYPE, MEM_TYPE::VV_ALL> VV_ALL_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::VST_VLD> VST_VLD_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::VLD_VST> VLD_VST_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::VST_VST> VST_VST_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::VS_ALL> VS_ALL_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::VST_LD> VST_LD_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::VLD_ST> VLD_ST_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::VST_ST> VST_ST_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::SV_ALL> SV_ALL_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::ST_VLD> ST_VLD_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::LD_VST> LD_VST_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::ST_VST> ST_VST_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::SS_ALL> SS_ALL_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::ST_LD> ST_LD_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::LD_ST> LD_ST_Type;
typedef std::integral_constant<MEM_TYPE, MEM_TYPE::ST_ST> ST_ST_Type;
#define VV_ALL VV_ALL_Type()
#define VST_VLD VST_VLD_Type()
#define VLD_VST VLD_VST_Type()
#define VST_VST VST_VST_Type()
#define VS_ALL VS_ALL_Type()
#define VST_LD VST_LD_Type()
#define VLD_ST VLD_ST_Type()
#define VST_ST VST_ST_Type()

#define SV_ALL SV_ALL_Type()
#define ST_VLD ST_VLD_Type()
#define LD_VST LD_VST_Type()
#define ST_VST ST_VST_Type()
#define SS_ALL SS_ALL_Type()
#define ST_LD ST_LD_Type()
#define LD_ST LD_ST_Type()
#define ST_ST ST_ST_Type()

#define INVALID_VALUE_MEM_BAR                                                  \
  "Invalid Input Argument. It can only be VV_ALL, VST_VLD, etc."

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__) || defined(__DAV_L310__)
template <class T> CCE_INTRINSIC void mem_bar(T mem_type) {
  static_assert(std::is_same<T, VV_ALL_Type>::value ||
                    std::is_same<T, VST_VLD_Type>::value ||
                    std::is_same<T, VLD_VST_Type>::value ||
                    std::is_same<T, VST_VST_Type>::value ||
                    std::is_same<T, VS_ALL_Type>::value ||
                    std::is_same<T, VST_LD_Type>::value ||
                    std::is_same<T, VLD_ST_Type>::value ||
                    std::is_same<T, VST_ST_Type>::value ||
                    std::is_same<T, SV_ALL_Type>::value ||
                    std::is_same<T, ST_VLD_Type>::value ||
                    std::is_same<T, LD_VST_Type>::value ||
                    std::is_same<T, ST_VST_Type>::value ||
                    std::is_same<T, SS_ALL_Type>::value ||
                    std::is_same<T, ST_LD_Type>::value ||
                    std::is_same<T, LD_ST_Type>::value ||
                    std::is_same<T, ST_ST_Type>::value,
                INVALID_VALUE_MEM_BAR);
  if (std::is_same<T, VV_ALL_Type>::value) {
    return __builtin_cce_mem_bar_vv_all();
  } else if (std::is_same<T, VST_VLD_Type>::value) {
    return __builtin_cce_mem_bar_vst_vld();
  } else if (std::is_same<T, VLD_VST_Type>::value) {
    return __builtin_cce_mem_bar_vld_vst();
  } else if (std::is_same<T, VST_VST_Type>::value) {
    return __builtin_cce_mem_bar_vst_vst();
  } else if (std::is_same<T, VS_ALL_Type>::value) {
    return __builtin_cce_mem_bar_vs_all();
  } else if (std::is_same<T, VST_LD_Type>::value) {
    return __builtin_cce_mem_bar_vst_ld();
  } else if (std::is_same<T, VLD_ST_Type>::value) {
    return __builtin_cce_mem_bar_vld_st();
  } else if (std::is_same<T, VST_ST_Type>::value) {
    return __builtin_cce_mem_bar_vst_st();
  } else if (std::is_same<T, SV_ALL_Type>::value) {
    return __builtin_cce_mem_bar_sv_all();
  } else if (std::is_same<T, ST_VLD_Type>::value) {
    return __builtin_cce_mem_bar_st_vld();
  } else if (std::is_same<T, LD_VST_Type>::value) {
    return __builtin_cce_mem_bar_ld_vst();
  } else if (std::is_same<T, ST_VST_Type>::value) {
    return __builtin_cce_mem_bar_st_vst();
  } else if (std::is_same<T, SS_ALL_Type>::value) {
    return __builtin_cce_mem_bar_ss_all();
  } else if (std::is_same<T, ST_LD_Type>::value) {
    return __builtin_cce_mem_bar_st_ld();
  } else if (std::is_same<T, LD_ST_Type>::value) {
    return __builtin_cce_mem_bar_ld_st();
  } else if (std::is_same<T, ST_ST_Type>::value) {
    return __builtin_cce_mem_bar_st_st();
  }
}
#endif

//----------------------------------------------------------------------------//
//  vector_b64
//----------------------------------------------------------------------------//
#if defined(__DAV_C310__)
#define VLDS_B64(PT1, PT2, VT1, VT2)                                           \
  CCE_INTRINSIC void vlds(vector_2xvl_##VT1 &dst, __ubuf__ PT1 *base,          \
                          int32_t offset /* in unit of element */) {           \
    __ubuf__ PT2 *base1 = (__ubuf__ PT2 *)base;                                \
    __builtin_cce_vldsx2_v64##VT2(&dst, base1, offset * sizeof(PT1),           \
                                  (ULL)Dist::DIST_DINTLV_B32,                  \
                                  0 /* post update mode */);                   \
    return;                                                                    \
  }

VLDS_B64(uint64_t, uint32_t, u64, u32)
VLDS_B64(int64_t, int32_t, s64, s32)
#undef VLDS_B64

#define VGATHER2_B64(PT1, PT2, VT1, VT2)                                       \
  CCE_INTRINSIC void vgather2(vector_2xvl_##VT1 &dst, __ubuf__ PT1 *base,      \
                              vector_u32 indexOffset, vector_bool mask) {      \
    vector_u32 odd_idx, even_idx;                                              \
    odd_idx = __builtin_cce_vmuls_v64u32_x(indexOffset, 2, mask);              \
    even_idx = __builtin_cce_vadds_v64u32_x(odd_idx, 1, mask);                 \
    __ubuf__ PT2 *base1 = (__ubuf__ PT2 *)base;                                \
    dst.val[0] = __builtin_cce_vgather2_v300_v64##VT2(base1, odd_idx, mask);   \
    dst.val[1] = __builtin_cce_vgather2_v300_v64##VT2(base1, even_idx, mask);  \
    return;                                                                    \
  }

VGATHER2_B64(int64_t, int32_t, s64, s32)
VGATHER2_B64(uint64_t, uint32_t, u64, u32)
#undef VGATHER2_B64

#define VSTS_B64(PT1, PT2, VT1, VT2)                                           \
  CCE_INTRINSIC void vsts(vector_2xvl_##VT1 data, __ubuf__ PT1 *base,          \
                          int32_t offset /* in unit of element */,             \
                          vector_bool mask) {                                  \
    __ubuf__ PT2 *base1 = (__ubuf__ PT2 *)base;                                \
    return __builtin_cce_vstsx2_v64b32(data.val[0], data.val[1], base1,        \
                                       offset * sizeof(PT1),                   \
                                       (ULL)DistVST::DIST_INTLV_B32,           \
                                       0 /* post */, mask);                    \
  }

VSTS_B64(int64_t, int32_t, s64, s32)
VSTS_B64(uint64_t, uint32_t, u64, u32)
#undef VSTS_B64

#define VSCATTER_B64(PT1, PT2, VT1, VT2)                                       \
  CCE_INTRINSIC void vscatter(vector_2xvl_##VT1 data, __ubuf__ PT1 *base,      \
                              vector_u32 indexOffset, vector_bool mask) {      \
    vector_u32 odd_idx, even_idx;                                              \
    odd_idx = __builtin_cce_vmuls_v64u32_x(indexOffset, 2, mask);              \
    even_idx = __builtin_cce_vadds_v64u32_x(odd_idx, 1, mask);                 \
    __ubuf__ PT2 *base1 = (__ubuf__ PT2 *)base;                                \
    __builtin_cce_vscatter_v64##VT2##_v300(data.val[0], base1, odd_idx, mask); \
    __builtin_cce_vscatter_v64##VT2##_v300(data.val[1], base1, even_idx,       \
                                           mask);                              \
    return;                                                                    \
  }

VSCATTER_B64(int64_t, int32_t, s64, s32)
VSCATTER_B64(uint64_t, uint32_t, u64, u32)
#undef VSCATTER_B64

#define VBR_B64(ST1, ST2, VT1, VT2)                                            \
  CCE_INTRINSIC void vbr(vector_2xvl_##VT1 &dst, ST1 value) {                  \
    dst.val[0] = vbr_##VT2((ST2)value, 0);                                     \
    dst.val[1] = vbr_##VT2((ST2)(value >> 32), 0);                             \
    return;                                                                    \
  }

VBR_B64(int64_t, int32_t, s64, s32)
VBR_B64(uint64_t, uint32_t, u64, u32)
#undef VBR_B64

#define VDUP_B64(VT1, VT2)                                                     \
  template <class T1, class T2>                                                \
  CCE_INTRINSIC void vdup(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src,       \
                          vector_bool mask, T1 POS, T2 mode) {                 \
    static_assert(std::is_same<T1, Lowest_Type>::value ||                      \
                      std::is_same<T1, Highest_Type>::value,                   \
                  "The 4th argument of this vdup can only be: "                \
                  "POS_LOWEST, POS_HIGHEST");                                  \
    static_assert(mode.value == MODE_MERGING.value ||                          \
                      mode.value == MODE_ZEROING.value,                        \
                  "The 5th argument of this vdup can only be: "                \
                  "MODE_MERGING, MODE_ZEROING");                               \
    if (POS.value == POS_LOWEST.value) {                                       \
      if (mode.value == MODE_MERGING.value) {                                  \
        dst.val[0] =                                                           \
            __builtin_cce_vdup_v64##VT2(dst.val[0], src.val[0], mask, 0);      \
        dst.val[1] =                                                           \
            __builtin_cce_vdup_v64##VT2(dst.val[1], src.val[1], mask, 0);      \
      } else {                                                                 \
        dst.val[0] = __builtin_cce_vdup_v64##VT2##_z(src.val[0], mask, 1);     \
        dst.val[1] = __builtin_cce_vdup_v64##VT2##_z(src.val[1], mask, 1);     \
      }                                                                        \
    } else {                                                                   \
      if (mode.value == MODE_MERGING.value) {                                  \
        dst.val[0] =                                                           \
            __builtin_cce_vdupm_v64##VT2(dst.val[0], src.val[0], mask, 0);     \
        dst.val[1] =                                                           \
            __builtin_cce_vdupm_v64##VT2(dst.val[1], src.val[1], mask, 0);     \
      } else {                                                                 \
        dst.val[0] = __builtin_cce_vdupm_v64##VT2##_z(src.val[0], mask, 1);    \
        dst.val[1] = __builtin_cce_vdupm_v64##VT2##_z(src.val[1], mask, 1);    \
      }                                                                        \
    }                                                                          \
    return;                                                                    \
  }

VDUP_B64(u64, u32)
VDUP_B64(s64, s32)

#define VDUPS_B64(ST1, ST2, VT1, VT2)                                          \
  template <class T>                                                           \
  CCE_INTRINSIC void vdup(vector_2xvl_##VT1 &dst, ST1 value, vector_bool mask, \
                          T mode) {                                            \
    if (mode.value == MODE_MERGING.value) {                                    \
      dst.val[0] =                                                             \
          __builtin_cce_vdups_v64##VT2(dst.val[0], (ST2)value, mask, 0);       \
      dst.val[1] = __builtin_cce_vdups_v64##VT2(dst.val[1],                    \
                                                (ST2)(value >> 32), mask, 0);  \
    } else {                                                                   \
      dst.val[0] = __builtin_cce_vdups_v64##VT2##_z((ST2)value, mask, 1);      \
      dst.val[1] =                                                             \
          __builtin_cce_vdups_v64##VT2##_z((ST2)(value >> 32), mask, 1);       \
    }                                                                          \
    return;                                                                    \
  }

VDUPS_B64(int64_t, int32_t, s64, s32)
VDUPS_B64(uint64_t, uint32_t, u64, u32)
#undef VDUPS_B64

#define VMOV_B64_PG(VT1, VT2)                                                  \
  template <class Tm = std::integral_constant<Mode, Mode::MERGING_VALUE>>      \
  CCE_INTRINSIC void vmov(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src,       \
                          vector_bool mask, Tm mode = MODE_MERGING) {          \
    static_assert(mode.value == MODE_MERGING.value,                            \
                  "The last argument can only be 'MODE_MERGING' or empty.");   \
    dst.val[0] =                                                               \
        __builtin_cce_vmov_v64##VT2##_m(dst.val[0], src.val[0], mask);         \
    dst.val[1] =                                                               \
        __builtin_cce_vmov_v64##VT2##_m(dst.val[1], src.val[1], mask);         \
    return;                                                                    \
  }

VMOV_B64_PG(u64, u32)
VMOV_B64_PG(s64, s32)
#undef VMOV_B64_PG

#define VMOV_B64(VT1, VT2)                                                     \
  CCE_INTRINSIC void vmov(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src) {     \
    dst.val[0] = __builtin_cce_vmov_v64##VT2(src.val[0]);                      \
    dst.val[1] = __builtin_cce_vmov_v64##VT2(src.val[1]);                      \
    return;                                                                    \
  }

VMOV_B64(u64, u32)
VMOV_B64(s64, s32)
#undef VMOV_B64

#define VSEL_B64(VT1, VT2)                                                     \
  CCE_INTRINSIC void vsel(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,      \
                          vector_2xvl_##VT1 src1, vector_bool mask) {          \
    dst.val[0] = __builtin_cce_vsel_v64##VT2(src0.val[0], src1.val[0], mask);  \
    dst.val[1] = __builtin_cce_vsel_v64##VT2(src0.val[1], src1.val[1], mask);  \
    return;                                                                    \
  }

VSEL_B64(u64, u32)
VSEL_B64(s64, s32)
#undef VSEL_B64

#define VSELR_B64(VT1, VT2, VT3)                                               \
  CCE_INTRINSIC void vselr(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,     \
                           vector_##VT2 src1) {                                \
    dst.val[0] = __builtin_cce_vselr_v64##VT3(src0.val[0], src1);              \
    dst.val[1] = __builtin_cce_vselr_v64##VT3(src0.val[1], src1);              \
    return;                                                                    \
  }

VSELR_B64(u64, u32, u32)
VSELR_B64(s64, u32, s32)
#undef VSELR_B64

#define VINTLV_B64(OP_NAME, VT1, VT2)                                          \
  CCE_INTRINSIC void OP_NAME(vector_2xvl_##VT1 &dst0, vector_2xvl_##VT1 &dst1, \
                             vector_2xvl_##VT1 src0, vector_2xvl_##VT1 src1) { \
    vector_2xvl_##VT1 vTmp0, vTmp1;                                            \
    __builtin_cce_##OP_NAME##_v64##VT2(&vTmp0, src0.val[0], src1.val[0]);      \
    __builtin_cce_##OP_NAME##_v64##VT2(&vTmp1, src0.val[1], src1.val[1]);      \
    dst0.val[0] = vTmp0.val[0];                                                \
    dst0.val[1] = vTmp1.val[0];                                                \
    dst1.val[0] = vTmp0.val[1];                                                \
    dst1.val[1] = vTmp1.val[1];                                                \
    return;                                                                    \
  }

VINTLV_B64(vintlv, u64, u32)
VINTLV_B64(vintlv, s64, s32)
VINTLV_B64(vdintlv, u64, u32)
VINTLV_B64(vdintlv, s64, s32)
#undef VINTLV_B64

#define VSLIDE_B64(VT1, VT2)                                                   \
  CCE_INTRINSIC void vslide(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,    \
                            vector_2xvl_##VT1 src1, int16_t slideAmount) {     \
    dst.val[0] =                                                               \
        __builtin_cce_vslide_v64##VT2(src0.val[0], src1.val[0], slideAmount);  \
    dst.val[1] =                                                               \
        __builtin_cce_vslide_v64##VT2(src0.val[1], src1.val[1], slideAmount);  \
    return;                                                                    \
  }

VSLIDE_B64(u64, u32)
VSLIDE_B64(s64, s32)
#undef VSLIDE_B64

#define VLOGIC_B64(OP_NAME, VT1, VT2)                                          \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,   \
                             vector_2xvl_##VT1 src1, vector_bool mask,         \
                             T mode = MODE_UNKNOWN) {                          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_2xvl_##VT1 dstTmp;                                                  \
    dstTmp.val[0] = __builtin_cce_##OP_NAME##_v64##VT2##_x(src0.val[0],        \
                                                           src1.val[0], mask); \
    dstTmp.val[1] = __builtin_cce_##OP_NAME##_v64##VT2##_x(src0.val[1],        \
                                                           src1.val[1], mask); \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], dstTmp.val[0], mask) \
            : dstTmp.val[0];                                                   \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], dstTmp.val[1], mask) \
            : dstTmp.val[1];                                                   \
    return;                                                                    \
  }

VLOGIC_B64(vand, u64, u32)
VLOGIC_B64(vand, s64, s32)
VLOGIC_B64(vor, u64, u32)
VLOGIC_B64(vor, s64, s32)
VLOGIC_B64(vxor, u64, u32)
VLOGIC_B64(vxor, s64, s32)
#undef VLOGIC_B64

#define VNOT_B64(VT1, VT2)                                                     \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void vnot(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src,       \
                          vector_bool mask, T mode = MODE_UNKNOWN) {           \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_2xvl_##VT1 dstTmp;                                                  \
    dstTmp.val[0] = __builtin_cce_vnot_v64##VT2##_x(src.val[0], mask);         \
    dstTmp.val[1] = __builtin_cce_vnot_v64##VT2##_x(src.val[1], mask);         \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], dstTmp.val[0], mask) \
            : dstTmp.val[0];                                                   \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], dstTmp.val[1], mask) \
            : dstTmp.val[1];                                                   \
    return;                                                                    \
  }

VNOT_B64(u64, u32)
VNOT_B64(s64, s32)
#undef VNOT_B64

template <class T = IncOrderType>
CCE_INTRINSIC void vci(vector_2xvl_s64 &dst, int32_t index,
                       T order = INC_ORDER) {
  dst.val[0] = __builtin_cce_vci_v64s32((int32_t)index, (ULL)order.value);
  dst.val[1] = vbr_s32((int32_t)(index >> 32), 0);
  return;
}

#define VCP_B64(VT1, VT2)                                                      \
  template <class T> CCE_INTRINSIC void vcp(vector_2xvl_##VT1 &dst, T pat) {   \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VCP);                 \
    static_assert(std::is_same<T, PatCHN4TO8Type>::value ||                    \
                      std::is_same<T, PatCHN4TO16Type>::value ||               \
                      std::is_same<T, PatCHN4TO32Type>::value ||               \
                      std::is_same<T, PatCHN8TO16Type>::value ||               \
                      std::is_same<T, PatCHN8TO32Type>::value,                 \
                  ERROR_VALUE_VCP);                                            \
    dst.val[0] = __builtin_cce_vcp_v64##VT2((ULL)pat.value);                   \
    dst.val[1] = vbr_##VT2(0, 0);                                              \
    return;                                                                    \
  }

VCP_B64(s64, s32)
VCP_B64(u64, u32)
#undef VCP_B64

#define BINARY_OP_B64(OP_NAME, VT1, VT2)                                       \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,   \
                             vector_2xvl_##VT1 src1, vector_bool mask,         \
                             T mode = MODE_UNKNOWN) {                          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    struct {                                                                   \
      vector_##VT2 dst_;                                                       \
      vector_bool carryp_;                                                     \
    } ret0;                                                                    \
    __builtin_cce_##OP_NAME##c_v64##VT2(&ret0, src0.val[0], src1.val[0],       \
                                        mask);                                 \
    struct {                                                                   \
      vector_##VT2 dst_;                                                       \
      vector_bool carryp_;                                                     \
    } ret1;                                                                    \
    __builtin_cce_##OP_NAME##cs_v64##VT2(&ret1, src0.val[1], src1.val[1],      \
                                         ret0.carryp_, mask);                  \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], ret0.dst_, mask)     \
            : ret0.dst_;                                                       \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], ret1.dst_, mask)     \
            : ret1.dst_;                                                       \
    return;                                                                    \
  }

BINARY_OP_B64(vadd, s64, s32)
BINARY_OP_B64(vadd, u64, u32)
BINARY_OP_B64(vsub, s64, s32)
BINARY_OP_B64(vsub, u64, u32)
#undef BINARY_OP_B64

#define VMUL_B64(VT1, VT2)                                                     \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void vmul(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,      \
                          vector_2xvl_##VT1 src1, vector_bool mask,            \
                          T mode = MODE_UNKNOWN) {                             \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_u32x2_t vTmp;                                                       \
    __builtin_cce_vmull_v64u32(&vTmp, src0.val[0], src1.val[0], mask);         \
    vTmp.val[1] = __builtin_cce_vmula_v64##VT2##_m(vTmp.val[1], src0.val[0],   \
                                                   src1.val[1], mask);         \
    vTmp.val[1] = __builtin_cce_vmula_v64##VT2##_m(vTmp.val[1], src0.val[1],   \
                                                   src1.val[0], mask);         \
    dst.val[0] = (mode.value == MODE_MERGING.value)                            \
                     ? __builtin_cce_vmov_v64##VT2##_m(                        \
                           dst.val[0], (vector_##VT2)vTmp.val[0], mask)        \
                     : (vector_##VT2)vTmp.val[0];                              \
    dst.val[1] = (mode.value == MODE_MERGING.value)                            \
                     ? __builtin_cce_vmov_v64##VT2##_m(                        \
                           dst.val[1], (vector_##VT2)vTmp.val[1], mask)        \
                     : (vector_##VT2)vTmp.val[1];                              \
    return;                                                                    \
  }

VMUL_B64(s64, s32)
VMUL_B64(u64, u32)
#undef VMUL_B64

#define VCMP_B64(OP, VTI, VT2)                                                 \
  CCE_INTRINSIC void vcmp_##OP(vector_bool &dst, vector_2xvl_##VTI src1,       \
                               vector_2xvl_##VTI src2, vector_bool mask) {     \
    vector_bool pTmp0, pTmp1;                                                  \
    pTmp0 =                                                                    \
        __builtin_cce_vcmp_##OP##_##VT2##_z(src1.val[0], src2.val[0], mask);   \
    pTmp1 =                                                                    \
        __builtin_cce_vcmp_##OP##_##VT2##_z(src1.val[1], src2.val[1], mask);   \
    dst = __builtin_cce_pand_z(pTmp0, pTmp1, mask);                            \
    return;                                                                    \
  }

VCMP_B64(eq, s64, s32)
VCMP_B64(ne, s64, s32)
VCMP_B64(eq, u64, u32)
VCMP_B64(ne, u64, u32)
#undef VCMP_B64

#define VCMPS_B64(OP, VTI, VT2, ST1, ST2)                                      \
  CCE_INTRINSIC void vcmps_##OP(vector_bool &dst, vector_2xvl_##VTI src1,      \
                                ST1 src2, vector_bool mask) {                  \
    vector_bool pTmp0, pTmp1;                                                  \
    pTmp0 =                                                                    \
        __builtin_cce_vcmps_##OP##_##VT2##_z(src1.val[0], (ST2)src2, mask);    \
    pTmp1 = __builtin_cce_vcmps_##OP##_##VT2##_z(src1.val[1],                  \
                                                 (ST2)(src2 >> 32), mask);     \
    dst = __builtin_cce_pand_z(pTmp0, pTmp1, mask);                            \
    return;                                                                    \
  }

VCMPS_B64(eq, s64, s32, int64_t, int32_t)
VCMPS_B64(ne, s64, s32, int64_t, int32_t)
VCMPS_B64(eq, u64, u32, uint64_t, uint32_t)
VCMPS_B64(ne, u64, u32, uint64_t, uint32_t)
#undef VCMPS_B64

#define VCMP_B64(OP, VTI, VT2)                                                 \
  CCE_INTRINSIC void vcmp_##OP(vector_bool &dst, vector_2xvl_##VTI src1,       \
                               vector_2xvl_##VTI src2, vector_bool mask) {     \
    vector_bool pTmp0, pTmp1, pTmp2;                                           \
    pTmp0 =                                                                    \
        __builtin_cce_vcmp_##OP##_##VT2##_z(src1.val[0], src2.val[0], mask);   \
    pTmp1 =                                                                    \
        __builtin_cce_vcmp_##OP##_##VT2##_z(src1.val[1], src2.val[1], mask);   \
    pTmp2 = __builtin_cce_vcmp_eq##_##VT2##_z(src1.val[1], src2.val[1], mask); \
    dst = __builtin_cce_psel(pTmp0, pTmp1, pTmp2);                             \
    return;                                                                    \
  }

VCMP_B64(gt, s64, s32)
VCMP_B64(ge, s64, s32)
VCMP_B64(lt, s64, s32)
VCMP_B64(le, s64, s32)
VCMP_B64(gt, u64, u32)
VCMP_B64(ge, u64, u32)
VCMP_B64(lt, u64, u32)
VCMP_B64(le, u64, u32)
#undef VCMP_B64

#define VCMPS_B64(OP, VTI, VT2, ST1, ST2)                                      \
  CCE_INTRINSIC void vcmps_##OP(vector_bool &dst, vector_2xvl_##VTI src1,      \
                                ST1 src2, vector_bool mask) {                  \
    vector_bool pTmp0, pTmp1, pTmp2;                                           \
    pTmp0 =                                                                    \
        __builtin_cce_vcmps_##OP##_##VT2##_z(src1.val[0], (ST2)src2, mask);    \
    pTmp1 = __builtin_cce_vcmps_##OP##_##VT2##_z(src1.val[1],                  \
                                                 (ST2)(src2 >> 32), mask);     \
    pTmp2 = __builtin_cce_vcmps_eq##_##VT2##_z(src1.val[1], (ST2)(src2 >> 32), \
                                               mask);                          \
    dst = __builtin_cce_psel(pTmp0, pTmp1, pTmp2);                             \
    return;                                                                    \
  }

VCMPS_B64(gt, s64, s32, int64_t, int32_t)
VCMPS_B64(ge, s64, s32, int64_t, int32_t)
VCMPS_B64(lt, s64, s32, int64_t, int32_t)
VCMPS_B64(le, s64, s32, int64_t, int32_t)
VCMPS_B64(gt, u64, u32, uint64_t, uint32_t)
VCMPS_B64(ge, u64, u32, uint64_t, uint32_t)
VCMPS_B64(lt, u64, u32, uint64_t, uint32_t)
VCMPS_B64(le, u64, u32, uint64_t, uint32_t)
#undef VCMPS_B64

#define VNEG_B64(VT1, VT2)                                                     \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void vneg(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src,       \
                          vector_bool mask, T mode = MODE_UNKNOWN) {           \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_2xvl_##VT1 dstTmp;                                                  \
    vector_##VT1 vTmp;                                                         \
    vTmp = vbr_##VT2(0, 0);                                                    \
    struct {                                                                   \
      vector_##VT2 dst_;                                                       \
      vector_bool carryp_;                                                     \
    } ret0;                                                                    \
    __builtin_cce_vsubc_v64##VT2(&ret0, vTmp, src.val[0], mask);               \
    struct {                                                                   \
      vector_##VT2 dst_;                                                       \
      vector_bool carryp_;                                                     \
    } ret1;                                                                    \
    __builtin_cce_vsubcs_v64##VT2(&ret1, vTmp, src.val[1], ret0.carryp_,       \
                                  mask);                                       \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], ret0.dst_, mask)     \
            : ret0.dst_;                                                       \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], ret1.dst_, mask)     \
            : ret1.dst_;                                                       \
    return;                                                                    \
  }

VNEG_B64(u64, u32)
VNEG_B64(s64, s32)
#undef VNEG_B64

#define MIN_MAX_B64(OP_NAME, CMP, VT1, VT2)                                    \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,   \
                             vector_2xvl_##VT1 src1, vector_bool mask,         \
                             T mode = MODE_UNKNOWN) {                          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_bool pTmp;                                                          \
    vector_2xvl_##VT1 dstTmp;                                                  \
    vcmp_##CMP(pTmp, src0, src1, mask);                                        \
    vsel(dstTmp, src0, src1, pTmp);                                            \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], dstTmp.val[0], mask) \
            : dstTmp.val[0];                                                   \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], dstTmp.val[1], mask) \
            : dstTmp.val[1];                                                   \
    return;                                                                    \
  }

MIN_MAX_B64(vmax, gt, s64, s32)
MIN_MAX_B64(vmax, gt, u64, u32)
MIN_MAX_B64(vmin, lt, s64, s32)
MIN_MAX_B64(vmin, lt, u64, u32)
#undef MIN_MAX_B64

#define VEC_SCALAR_OP_B64(OP_NAME, VT, ST)                                     \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME##s(vector_2xvl_##VT &dst, vector_2xvl_##VT src0,  \
                                ST src1, vector_bool mask,                     \
                                T mode = MODE_UNKNOWN) {                       \
    vector_2xvl_##VT vTmp;                                                     \
    vbr(vTmp, src1);                                                           \
    OP_NAME(dst, src0, vTmp, mask, mode);                                      \
    return;                                                                    \
  }

VEC_SCALAR_OP_B64(vadd, s64, int64_t)
VEC_SCALAR_OP_B64(vadd, u64, uint64_t)

VEC_SCALAR_OP_B64(vmul, s64, int64_t)
VEC_SCALAR_OP_B64(vmul, u64, uint64_t)

VEC_SCALAR_OP_B64(vmax, s64, int64_t)
VEC_SCALAR_OP_B64(vmax, u64, uint64_t)

VEC_SCALAR_OP_B64(vmin, s64, int64_t)
VEC_SCALAR_OP_B64(vmin, u64, uint64_t)

#undef VEC_SCALAR_OP_B64

#define VABS_B64(VT1, VT2)                                                     \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void vabs(vector_2xvl_s64 &dst, vector_2xvl_s64 src,           \
                          vector_bool mask, T mode = MODE_UNKNOWN) {           \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_bool pTmp;                                                          \
    vector_2xvl_s64 dstTmp;                                                    \
    vector_s32 vTmp;                                                           \
    vTmp = vbr_s32(0, 0);                                                      \
    pTmp = __builtin_cce_vcmp_lt_s32_z(src.val[1], vTmp, mask);                \
    struct {                                                                   \
      vector_s32 dst_;                                                         \
      vector_bool carryp_;                                                     \
    } ret0;                                                                    \
    __builtin_cce_vsubc_v64s32(&ret0, vTmp, src.val[0], pTmp);                 \
    struct {                                                                   \
      vector_s32 dst_;                                                         \
      vector_bool carryp_;                                                     \
    } ret1;                                                                    \
    __builtin_cce_vsubcs_v64s32(&ret1, vTmp, src.val[1], ret0.carryp_, pTmp);  \
    src.val[0] = __builtin_cce_vmov_v64s32_m(src.val[0], ret0.dst_, pTmp);     \
    src.val[1] = __builtin_cce_vmov_v64s32_m(src.val[1], ret1.dst_, pTmp);     \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64s32_m(dst.val[0], src.val[0], mask)        \
            : src.val[0];                                                      \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64s32_m(dst.val[1], src.val[1], mask)        \
            : src.val[1];                                                      \
    return;                                                                    \
  }

VABS_B64(s64, s32)
#undef VABS_B64

// Indicates whether to perform logical shift and arithmetic shift based on
// shift Inst vector_type.
#define VSHIFT_B64(OP_NAME, VT1, VT2, VT3, VT4, VT5, VT6, ISSHR)               \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,   \
                             vector_s32 src1, vector_bool mask,                \
                             T mode = MODE_UNKNOWN) {                          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_2xvl_##VT1 dstTmp;                                                  \
    vector_s32 vTmp0, vTmp1, vTmp2;                                            \
    vector_##VT2 vTmp3, vTmp4;                                                 \
    vTmp0 = vbr_s32(32, 0);                                                    \
    vTmp1 = __builtin_cce_vsub_v64s32_x(vTmp0, src1, mask);                    \
    vTmp2 = __builtin_cce_vadds_v64s32_x(src1, 32, mask);                      \
    vTmp3 = __builtin_cce_##OP_NAME##_v64##VT3##_x((vector_##VT3)src0.val[0],  \
                                                   src1, mask);                \
    vTmp4 = ISSHR ? __builtin_cce_vshl##_v64##VT4##_x((vector_u32)src0.val[1], \
                                                      vTmp1, mask)             \
                  : __builtin_cce_vshl##_v64##VT4##_x((vector_u32)src0.val[1], \
                                                      vTmp2, mask);            \
    dstTmp.val[0] = __builtin_cce_vor_v64##VT2##_x(vTmp3, vTmp4, mask);        \
    vTmp3 = ISSHR                                                              \
                ? __builtin_cce_vshr##_v64##VT5##_x((vector_##VT4)src0.val[0], \
                                                    vTmp2, mask)               \
                : __builtin_cce_vshr##_v64##VT5##_x((vector_##VT4)src0.val[0], \
                                                    vTmp1, mask);              \
    vTmp4 = __builtin_cce_##OP_NAME##_v64##VT6##_x((vector_##VT5)src0.val[1],  \
                                                   src1, mask);                \
    dstTmp.val[1] = __builtin_cce_vor_v64##VT2##_x(vTmp3, vTmp4, mask);        \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], dstTmp.val[0], mask) \
            : dstTmp.val[0];                                                   \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], dstTmp.val[1], mask) \
            : dstTmp.val[1];                                                   \
    return;                                                                    \
  }

VSHIFT_B64(vshr, u64, u32, u32, u32, u32, u32, 1)
VSHIFT_B64(vshl, u64, u32, u32, u32, u32, u32, 0)
VSHIFT_B64(vshr, s64, s32, u32, s32, u32, s32, 1)
VSHIFT_B64(vshl, s64, s32, u32, u32, u32, s32, 0)
#undef VSHIFT_B64

#define VSHIFTS_B64(OP_NAME, VT1, VT2, VT3, VT4, VT5, VT6, ISSHR)              \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void OP_NAME(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,   \
                             int32_t src1, vector_bool mask,                   \
                             T mode = MODE_UNKNOWN) {                          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_2xvl_##VT1 dstTmp;                                                  \
    vector_##VT2 vTmp3, vTmp4;                                                 \
    vTmp3 = __builtin_cce_##OP_NAME##_v64##VT3##_x((vector_##VT3)src0.val[0],  \
                                                   (int32_t)src1, mask);       \
    vTmp4 = ISSHR ? __builtin_cce_vshls##_v64##VT4##_x(                        \
                        (vector_u32)src0.val[1], 32 - (int32_t)src1, mask)     \
                  : __builtin_cce_vshls##_v64##VT4##_x(                        \
                        (vector_u32)src0.val[1], 32 + (int32_t)src1, mask);    \
    dstTmp.val[0] = __builtin_cce_vor_v64##VT2##_x(vTmp3, vTmp4, mask);        \
    vTmp3 = ISSHR ? __builtin_cce_vshrs##_v64##VT5##_x(                        \
                        (vector_##VT4)src0.val[0], 32 + (int32_t)src1, mask)   \
                  : __builtin_cce_vshrs##_v64##VT5##_x(                        \
                        (vector_##VT4)src0.val[0], 32 - (int32_t)src1, mask);  \
    vTmp4 = __builtin_cce_##OP_NAME##_v64##VT6##_x((vector_##VT5)src0.val[1],  \
                                                   (int32_t)src1, mask);       \
    dstTmp.val[1] = __builtin_cce_vor_v64##VT2##_x(vTmp3, vTmp4, mask);        \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], dstTmp.val[0], mask) \
            : dstTmp.val[0];                                                   \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], dstTmp.val[1], mask) \
            : dstTmp.val[1];                                                   \
    return;                                                                    \
  }

VSHIFTS_B64(vshrs, u64, u32, u32, u32, u32, u32, 1)
VSHIFTS_B64(vshls, u64, u32, u32, u32, u32, u32, 0)
VSHIFTS_B64(vshrs, s64, s32, u32, s32, u32, s32, 1)
VSHIFTS_B64(vshls, s64, s32, u32, u32, u32, s32, 0)
#undef VSHIFTS_B64

#define VBCNT_B64(VT1, VT2)                                                    \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void vbcnt(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src,      \
                           vector_bool mask, T mode = MODE_UNKNOWN) {          \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_2xvl_##VT1 dstTmp;                                                  \
    vector_##VT1 vTmp0, vTmp1;                                                 \
    vTmp0 = __builtin_cce_vbcnt_v64##VT2##_x(src.val[0], mask);                \
    vTmp1 = __builtin_cce_vbcnt_v64##VT2##_x(src.val[1], mask);                \
    dstTmp.val[0] = __builtin_cce_vadd_v64##VT2##_x(vTmp0, vTmp1, mask);       \
    dstTmp.val[1] = vbr_##VT2(0, 0);                                           \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], dstTmp.val[0], mask) \
            : dstTmp.val[0];                                                   \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], dstTmp.val[1], mask) \
            : dstTmp.val[1];                                                   \
    return;                                                                    \
  }

VBCNT_B64(u64, u32)
VBCNT_B64(s64, s32)
#undef VBCNT_B64

#define VCLS_B64(VT1, VT2, SIGN)                                               \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void vcls(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src,       \
                          vector_bool mask, T mode = MODE_UNKNOWN) {           \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_2xvl_##VT1 dstTmp;                                                  \
    dstTmp.val[1] = vbr_##VT2(0, 0);                                           \
    vector_##VT2 vTmp0, vTmp1;                                                 \
    vector_bool pTmp;                                                          \
    vTmp0 = __builtin_cce_vcls_v64##VT2##_x(src.val[1], mask);                 \
    if (SIGN) {                                                                \
      vTmp1 = __builtin_cce_vshrs_v64s32_x(src.val[0], 31, mask);              \
    } else {                                                                   \
      vTmp1 = dstTmp.val[1];                                                   \
    }                                                                          \
    pTmp = __builtin_cce_vcmp_eq_s32_z(src.val[1], vTmp1, mask);               \
    vTmp1 = __builtin_cce_vcls_v64##VT2##_x(src.val[0], pTmp);                 \
    dstTmp.val[0] = __builtin_cce_vadd_v64##VT2##_x(vTmp0, vTmp1, mask);       \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], dstTmp.val[0], mask) \
            : dstTmp.val[0];                                                   \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], dstTmp.val[1], mask) \
            : dstTmp.val[1];                                                   \
    return;                                                                    \
  }

VCLS_B64(u64, u32, 0)
VCLS_B64(s64, s32, 1)
#undef VCLS_B64

#define VCVT_B64(TO, FROM)                                                     \
  CCE_INTRINSIC void vcvt(vector_##TO &dst, vector_2xvl_##FROM src) {          \
    dst = src.val[0];                                                          \
    return;                                                                    \
  }

VCVT_B64(u32, u64)
VCVT_B64(s32, s64)
#undef VCVT_B64

#define VCVT_B64(TO, FROM, TYPE)                                               \
  CCE_INTRINSIC void vcvt(vector_2xvl_##TO &dst, vector_2xvl_##FROM src) {     \
    dst.val[0] = (vector_##TYPE)src.val[0];                                    \
    dst.val[1] = (vector_##TYPE)src.val[1];                                    \
    return;                                                                    \
  }

VCVT_B64(s64, u64, s32)
VCVT_B64(u64, s64, u32)
#undef VCVT_B64

#define VCVT_B64(TO, FROM, SIGN)                                               \
  CCE_INTRINSIC void vcvt(vector_2xvl_##TO &dst, vector_##FROM src) {          \
    dst.val[0] = src;                                                          \
    vector_bool mask = pset_b32(PAT_ALL);                                      \
    if (SIGN) {                                                                \
      dst.val[1] = (vector_##FROM)__builtin_cce_vshrs_v64s32_x(src, 31, mask); \
    } else {                                                                   \
      dst.val[1] = (vector_##FROM)vbr_u32(0, 0);                               \
    }                                                                          \
    return;                                                                    \
  }

VCVT_B64(u64, u32, 0)
VCVT_B64(s64, s32, 1)
#undef VCVT_B64

// v64f32 to v64s64
template <class T1, class T2>
CCE_INTRINSIC void vcvt(vector_2xvl_s64 &dst, vector_f32 src, T1 rnd, T2 sat) {
  static_assert(std::is_class<T1>::value, "the 3th argument is invalid");
  static_assert(std::is_class<T2>::value, "the 4th argument is invalid");
  static_assert(std::is_same<T1, RoundRType>::value ||
                    std::is_same<T1, RoundAType>::value ||
                    std::is_same<T1, RoundFType>::value ||
                    std::is_same<T1, RoundCType>::value ||
                    std::is_same<T1, RoundZType>::value,
                "The 4th argument of this vcvt ( v64f32 2 v64s64 ) can "
                "only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");
  static_assert(std::is_same<T2, RSDisableType>::value ||
                    std::is_same<T2, RSEnableType>::value,
                "The 5th argument of this vcvt ( v64f32 2 v64s64 "
                ") can only be: RS_DISABLE, RS_ENABLE");
  vector_s64 vTmp0, vTmp1;
  vector_bool mask = pset_b32(PAT_ALL);
  vector_f32x2_t vTmp3;
  __builtin_cce_vintlv_v64s32(&vTmp3, src, src);
  vTmp0 = __builtin_cce_vcvtfi_f322s64_x(vTmp3.val[0], mask, (ULL)rnd.value,
                                         (ULL)sat.value, (ULL)Part::EVEN);
  vTmp1 = __builtin_cce_vcvtfi_f322s64_x(vTmp3.val[1], mask, (ULL)rnd.value,
                                         (ULL)sat.value, (ULL)Part::EVEN);
  __builtin_cce_vdintlv_v64s32(&dst, (vector_s32)vTmp0, (vector_s32)vTmp1);
  return;
}

// v64s64 to v64f32
template <class T>
CCE_INTRINSIC void vcvt(vector_f32 &dst, vector_2xvl_s64 src, T rnd) {
  static_assert(std::is_class<T>::value, "the 3th argument is invalid");
  static_assert(std::is_same<T, RoundRType>::value ||
                    std::is_same<T, RoundAType>::value ||
                    std::is_same<T, RoundFType>::value ||
                    std::is_same<T, RoundCType>::value ||
                    std::is_same<T, RoundZType>::value,
                "The 4th argument of this vcvt ( v64s64 2 v64f32 ) can "
                "only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z");
  vector_2xvl_s64 vTmp0;
  vector_f32 vTmp1, vTmp2;
  __builtin_cce_vintlv_v64s32(&vTmp0, src.val[0], src.val[1]);
  vector_bool mask = pset_b32(PAT_ALL);
  vTmp1 = __builtin_cce_vcvtif_s642f32_x((vector_s64)vTmp0.val[0], mask,
                                         (ULL)rnd.value, (ULL)Part::EVEN);
  vTmp2 = __builtin_cce_vcvtif_s642f32_x((vector_s64)vTmp0.val[1], mask,
                                         (ULL)rnd.value, (ULL)Part::EVEN);
  vector_f32x2_t vTmp3;
  __builtin_cce_vdintlv_v64f32(&vTmp3, vTmp1, vTmp2);
  dst = vTmp3.val[0];
  return;
}

// For details, see int64div.cpp in the design document.
#define VDIV_B64(VT1, VT2, SIGN)                                               \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void vdiv(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,      \
                          vector_2xvl_##VT1 src1, vector_bool mask,            \
                          T mode = MODE_UNKNOWN) {                             \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_2xvl_##VT1 vTmp0, vTmp1;                                            \
    vector_f32 vTmp2, vTmp3;                                                   \
    vector_u32 vTmp4;                                                          \
    vector_2xvl_s64 vTmp5, vAbsSrc0, vAbsSrc1;                                 \
    vAbsSrc0 = (vector_2xvl_s64)src0;                                          \
    vAbsSrc1 = (vector_2xvl_s64)src1;                                          \
    vector_2xvl_u64 vTmp6, vTmp7, vTmp8, vTmp9;                                \
    if (SIGN) {                                                                \
      vabs(vAbsSrc0, (vector_2xvl_s64)src0, mask);                             \
      vabs(vAbsSrc1, (vector_2xvl_s64)src1, mask);                             \
    }                                                                          \
    vcvt(vTmp2, vAbsSrc1, ROUND_R);                                            \
    vbr(vTmp3, (float)1);                                                      \
    vdiv(vTmp3, vTmp3, vTmp2, mask);                                           \
    vadds(vTmp4, (vector_u32)vTmp3, 0x1ffffffeU, mask);                        \
    vcvt(vTmp5, (vector_f32)vTmp4, ROUND_R, RS_DISABLE);                       \
    vmul(vTmp6, (vector_2xvl_u64)vAbsSrc1, (vector_2xvl_u64)vTmp5, mask);      \
    vnot(vTmp6, vTmp6, mask);                                                  \
    vadds(vTmp6, vTmp6, 1, mask);                                              \
    vbr(vTmp4, 0);                                                             \
    vector_2xvl_u64 vTmp128Mul0, vTmp128Mul1, vTmp128Mul2, vTmp128Mul3;        \
    struct vecP {                                                              \
      vector_u32 dstV_;                                                        \
      vector_bool carryP_;                                                     \
    };                                                                         \
    vecP vecPTmp0, vecPTmp1;                                                   \
    __builtin_cce_vmull_v64u32(&vTmp128Mul0, vTmp5.val[0], vTmp6.val[0],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul1, vTmp5.val[0], vTmp6.val[1],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul2, vTmp5.val[1], vTmp6.val[0],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul3, vTmp5.val[1], vTmp6.val[1],       \
                               mask);                                          \
    __builtin_cce_vaddc_v64u32(&vecPTmp0, vTmp128Mul0.val[1],                  \
                               vTmp128Mul1.val[0], mask);                      \
    __builtin_cce_vaddc_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[0],  \
                               mask);                                          \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp128Mul3.val[0],                 \
                                vTmp128Mul1.val[1], vecPTmp0.carryP_, mask);   \
    __builtin_cce_vaddcs_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[1], \
                                vecPTmp1.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vTmp128Mul3.val[1],          \
                                vecPTmp0.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vecPTmp0.dstV_,              \
                                vecPTmp1.carryP_, mask);                       \
    vTmp6.val[0] = vecPTmp1.dstV_;                                             \
    vTmp6.val[1] = vecPTmp0.dstV_;                                             \
    vadd(vTmp7, (vector_2xvl_u64)vTmp5, vTmp6, mask);                          \
    vmul(vTmp6, (vector_2xvl_u64)vAbsSrc1, (vector_2xvl_u64)vTmp7, mask);      \
    vnot(vTmp6, vTmp6, mask);                                                  \
    vadds(vTmp6, vTmp6, 1, mask);                                              \
    __builtin_cce_vmull_v64u32(&vTmp128Mul0, vTmp7.val[0], vTmp6.val[0],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul1, vTmp7.val[0], vTmp6.val[1],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul2, vTmp7.val[1], vTmp6.val[0],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul3, vTmp7.val[1], vTmp6.val[1],       \
                               mask);                                          \
    __builtin_cce_vaddc_v64u32(&vecPTmp0, vTmp128Mul0.val[1],                  \
                               vTmp128Mul1.val[0], mask);                      \
    __builtin_cce_vaddc_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[0],  \
                               mask);                                          \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp128Mul3.val[0],                 \
                                vTmp128Mul1.val[1], vecPTmp0.carryP_, mask);   \
    __builtin_cce_vaddcs_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[1], \
                                vecPTmp1.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vTmp128Mul3.val[1],          \
                                vecPTmp0.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vecPTmp0.dstV_,              \
                                vecPTmp1.carryP_, mask);                       \
    vTmp6.val[0] = vecPTmp1.dstV_;                                             \
    vTmp6.val[1] = vecPTmp0.dstV_;                                             \
    vadd(vTmp6, (vector_2xvl_u64)vTmp7, vTmp6, mask);                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul0, vAbsSrc0.val[0], vTmp6.val[0],    \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul1, vAbsSrc0.val[0], vTmp6.val[1],    \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul2, vAbsSrc0.val[1], vTmp6.val[0],    \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul3, vAbsSrc0.val[1], vTmp6.val[1],    \
                               mask);                                          \
    __builtin_cce_vaddc_v64u32(&vecPTmp0, vTmp128Mul0.val[1],                  \
                               vTmp128Mul1.val[0], mask);                      \
    __builtin_cce_vaddc_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[0],  \
                               mask);                                          \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp128Mul3.val[0],                 \
                                vTmp128Mul1.val[1], vecPTmp0.carryP_, mask);   \
    __builtin_cce_vaddcs_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[1], \
                                vecPTmp1.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vTmp128Mul3.val[1],          \
                                vecPTmp0.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vecPTmp0.dstV_,              \
                                vecPTmp1.carryP_, mask);                       \
    vTmp6.val[0] = vecPTmp1.dstV_;                                             \
    vTmp6.val[1] = vecPTmp0.dstV_;                                             \
    vmul(vTmp7, vTmp6, (vector_2xvl_u64)vAbsSrc1, mask);                       \
    vsub(vTmp7, (vector_2xvl_u64)vAbsSrc0, vTmp7, mask);                       \
    vector_bool ge0P;                                                          \
    vcmp_ge(ge0P, vTmp7, (vector_2xvl_u64)vAbsSrc1, mask);                     \
    vsub(vTmp8, vTmp7, (vector_2xvl_u64)vAbsSrc1, ge0P);                       \
    vadds(vTmp9, vTmp6, 1, ge0P);                                              \
    vsel(vTmp7, vTmp8, vTmp7, ge0P);                                           \
    vsel(vTmp6, vTmp9, vTmp6, ge0P);                                           \
    vcmp_ge(ge0P, vTmp7, (vector_2xvl_u64)vAbsSrc1, mask);                     \
    vsub(vTmp8, vTmp7, (vector_2xvl_u64)vAbsSrc1, ge0P);                       \
    vadds(vTmp9, vTmp6, 1, ge0P);                                              \
    vsel(vTmp7, vTmp8, vTmp7, ge0P);                                           \
    vsel(vTmp6, vTmp9, vTmp6, ge0P);                                           \
    vector_bool src0Ge0P, src1Ge0P, signResP;                                  \
    src0Ge0P = __builtin_cce_vcmp_ge_s32_z(src0.val[1], vTmp4, mask);          \
    src1Ge0P = __builtin_cce_vcmp_ge_s32_z(src1.val[1], vTmp4, mask);          \
    pxor(signResP, src0Ge0P, src1Ge0P, mask);                                  \
    pnot(signResP, signResP, mask);                                            \
    vector_bool pg1 = pset_b32(PAT_ALL);                                       \
    vneg(vTmp8, vTmp6, pg1);                                                   \
    vsel(vTmp6, vTmp6, vTmp8, signResP);                                       \
    ge0P = __builtin_cce_vcmp_ge_s32_z(src0.val[1], vTmp4, mask);              \
    vneg(vTmp8, vTmp7, pg1);                                                   \
    vsel(vTmp7, vTmp7, vTmp8, ge0P);                                           \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], vTmp6.val[0], mask)  \
            : vTmp6.val[0];                                                    \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], vTmp6.val[1], mask)  \
            : vTmp6.val[1];                                                    \
    return;                                                                    \
  }

VDIV_B64(s64, s32, 1)
VDIV_B64(u64, u32, 0)
#undef VDIV_B64

// For details, see int64div.cpp in the design document.
#define VMOD_B64(VT1, VT2, SIGN)                                               \
  template <class T = Mode_Unknown_Type>                                       \
  CCE_INTRINSIC void vmod(vector_2xvl_##VT1 &dst, vector_2xvl_##VT1 src0,      \
                          vector_2xvl_##VT1 src1, vector_bool mask,            \
                          T mode = MODE_UNKNOWN) {                             \
    static_assert(mode.value == MODE_ZEROING.value |                           \
                      mode.value == MODE_UNKNOWN.value |                       \
                      mode.value == MODE_MERGING.value,                        \
                  INVALID_VALUE_PREDICATE_MODE);                               \
    vector_2xvl_##VT1 vTmp0, vTmp1;                                            \
    vector_f32 vTmp2, vTmp3;                                                   \
    vector_u32 vTmp4;                                                          \
    vector_2xvl_s64 vTmp5, vAbsSrc0, vAbsSrc1;                                 \
    vAbsSrc0 = (vector_2xvl_s64)src0;                                          \
    vAbsSrc1 = (vector_2xvl_s64)src1;                                          \
    vector_2xvl_u64 vTmp6, vTmp7, vTmp8, vTmp9;                                \
    if (SIGN) {                                                                \
      vabs(vAbsSrc0, (vector_2xvl_s64)src0, mask);                             \
      vabs(vAbsSrc1, (vector_2xvl_s64)src1, mask);                             \
    }                                                                          \
    vcvt(vTmp2, vAbsSrc1, ROUND_R);                                            \
    vbr(vTmp3, (float)1);                                                      \
    vdiv(vTmp3, vTmp3, vTmp2, mask);                                           \
    vadds(vTmp4, (vector_u32)vTmp3, 0x1ffffffeU, mask);                        \
    vcvt(vTmp5, (vector_f32)vTmp4, ROUND_R, RS_DISABLE);                       \
    vmul(vTmp6, (vector_2xvl_u64)vAbsSrc1, (vector_2xvl_u64)vTmp5, mask);      \
    vnot(vTmp6, vTmp6, mask);                                                  \
    vadds(vTmp6, vTmp6, 1, mask);                                              \
    vbr(vTmp4, 0);                                                             \
    vector_2xvl_u64 vTmp128Mul0, vTmp128Mul1, vTmp128Mul2, vTmp128Mul3;        \
    struct vecP {                                                              \
      vector_u32 dstV_;                                                        \
      vector_bool carryP_;                                                     \
    };                                                                         \
    vecP vecPTmp0, vecPTmp1;                                                   \
    __builtin_cce_vmull_v64u32(&vTmp128Mul0, vTmp5.val[0], vTmp6.val[0],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul1, vTmp5.val[0], vTmp6.val[1],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul2, vTmp5.val[1], vTmp6.val[0],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul3, vTmp5.val[1], vTmp6.val[1],       \
                               mask);                                          \
    __builtin_cce_vaddc_v64u32(&vecPTmp0, vTmp128Mul0.val[1],                  \
                               vTmp128Mul1.val[0], mask);                      \
    __builtin_cce_vaddc_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[0],  \
                               mask);                                          \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp128Mul3.val[0],                 \
                                vTmp128Mul1.val[1], vecPTmp0.carryP_, mask);   \
    __builtin_cce_vaddcs_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[1], \
                                vecPTmp1.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vTmp128Mul3.val[1],          \
                                vecPTmp0.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vecPTmp0.dstV_,              \
                                vecPTmp1.carryP_, mask);                       \
    vTmp6.val[0] = vecPTmp1.dstV_;                                             \
    vTmp6.val[1] = vecPTmp0.dstV_;                                             \
    vadd(vTmp7, (vector_2xvl_u64)vTmp5, vTmp6, mask);                          \
    vmul(vTmp6, (vector_2xvl_u64)vAbsSrc1, (vector_2xvl_u64)vTmp7, mask);      \
    vnot(vTmp6, vTmp6, mask);                                                  \
    vadds(vTmp6, vTmp6, 1, mask);                                              \
    __builtin_cce_vmull_v64u32(&vTmp128Mul0, vTmp7.val[0], vTmp6.val[0],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul1, vTmp7.val[0], vTmp6.val[1],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul2, vTmp7.val[1], vTmp6.val[0],       \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul3, vTmp7.val[1], vTmp6.val[1],       \
                               mask);                                          \
    __builtin_cce_vaddc_v64u32(&vecPTmp0, vTmp128Mul0.val[1],                  \
                               vTmp128Mul1.val[0], mask);                      \
    __builtin_cce_vaddc_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[0],  \
                               mask);                                          \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp128Mul3.val[0],                 \
                                vTmp128Mul1.val[1], vecPTmp0.carryP_, mask);   \
    __builtin_cce_vaddcs_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[1], \
                                vecPTmp1.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vTmp128Mul3.val[1],          \
                                vecPTmp0.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vecPTmp0.dstV_,              \
                                vecPTmp1.carryP_, mask);                       \
    vTmp6.val[0] = vecPTmp1.dstV_;                                             \
    vTmp6.val[1] = vecPTmp0.dstV_;                                             \
    vadd(vTmp6, (vector_2xvl_u64)vTmp7, vTmp6, mask);                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul0, vAbsSrc0.val[0], vTmp6.val[0],    \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul1, vAbsSrc0.val[0], vTmp6.val[1],    \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul2, vAbsSrc0.val[1], vTmp6.val[0],    \
                               mask);                                          \
    __builtin_cce_vmull_v64u32(&vTmp128Mul3, vAbsSrc0.val[1], vTmp6.val[1],    \
                               mask);                                          \
    __builtin_cce_vaddc_v64u32(&vecPTmp0, vTmp128Mul0.val[1],                  \
                               vTmp128Mul1.val[0], mask);                      \
    __builtin_cce_vaddc_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[0],  \
                               mask);                                          \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp128Mul3.val[0],                 \
                                vTmp128Mul1.val[1], vecPTmp0.carryP_, mask);   \
    __builtin_cce_vaddcs_v64u32(&vecPTmp1, vecPTmp0.dstV_, vTmp128Mul2.val[1], \
                                vecPTmp1.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vTmp128Mul3.val[1],          \
                                vecPTmp0.carryP_, mask);                       \
    __builtin_cce_vaddcs_v64u32(&vecPTmp0, vTmp4, vecPTmp0.dstV_,              \
                                vecPTmp1.carryP_, mask);                       \
    vTmp6.val[0] = vecPTmp1.dstV_;                                             \
    vTmp6.val[1] = vecPTmp0.dstV_;                                             \
    vmul(vTmp7, vTmp6, (vector_2xvl_u64)vAbsSrc1, mask);                       \
    vsub(vTmp7, (vector_2xvl_u64)vAbsSrc0, vTmp7, mask);                       \
    vector_bool ge0P;                                                          \
    vcmp_ge(ge0P, vTmp7, (vector_2xvl_u64)vAbsSrc1, mask);                     \
    vsub(vTmp8, vTmp7, (vector_2xvl_u64)vAbsSrc1, ge0P);                       \
    vsel(vTmp7, vTmp8, vTmp7, ge0P);                                           \
    vcmp_ge(ge0P, vTmp7, (vector_2xvl_u64)vAbsSrc1, mask);                     \
    vsub(vTmp8, vTmp7, (vector_2xvl_u64)vAbsSrc1, ge0P);                       \
    vsel(vTmp7, vTmp8, vTmp7, ge0P);                                           \
    vector_bool pg1 = pset_b32(PAT_ALL);                                       \
    ge0P = __builtin_cce_vcmp_ge_s32_z(src0.val[1], vTmp4, mask);              \
    vneg(vTmp8, vTmp7, pg1);                                                   \
    vsel(vTmp7, vTmp7, vTmp8, ge0P);                                           \
    dst.val[0] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[0], vTmp7.val[0], mask)  \
            : vTmp7.val[0];                                                    \
    dst.val[1] =                                                               \
        (mode.value == MODE_MERGING.value)                                     \
            ? __builtin_cce_vmov_v64##VT2##_m(dst.val[1], vTmp7.val[1], mask)  \
            : vTmp7.val[1];                                                    \
    return;                                                                    \
  }

VMOD_B64(s64, s32, 1)
VMOD_B64(u64, u32, 0)
#undef VMOD_B64

#endif //__DAV_C310__

#endif //__CLANG_CCE_VECTOR_INTRINSICS_H
