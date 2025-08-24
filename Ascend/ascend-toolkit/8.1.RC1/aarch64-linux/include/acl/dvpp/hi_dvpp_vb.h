/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 and
 * only version 2 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * Description:
 * Author: huawei
 * Create: 2020-4-1
 */

#ifndef HI_DVPP_VB_H_
#define HI_DVPP_VB_H_

#include "hi_dvpp_common.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#define HI_MAX_MMZ_NAME_LEN 32
#define HI_VB_INVALID_POOLID -1
#define HI_VB_INVALID_HANDLE -1
typedef hi_u32 hi_vb_pool;
typedef hi_u32  hi_vb_blk;
typedef hi_u64 hi_phys_addr_t;

typedef enum hiVB_REMAP_MODE_E {
    VB_REMAP_MODE_NONE = 0,
    VB_REMAP_MODE_NOCACHE = 1,
    VB_REMAP_MODE_CACHED = 2,
} hi_vb_remap_mode;

typedef struct {
    hi_u64  blk_size;
    hi_u32  blk_cnt;
    hi_vb_remap_mode remap_mode;
    hi_char mmz_name[HI_MAX_MMZ_NAME_LEN];
} hi_vb_pool_config;

typedef enum  {
    HI_VB_SRC_COMMON  = 0,
    HI_VB_SRC_MOD  = 1,
    HI_VB_SRC_PRIVATE = 2,
    HI_VB_SRC_USER    = 3,
    HI_VB_SRC_BUTT
} hi_vb_src;

hi_vb_pool hi_mpi_vb_create_pool(const hi_vb_pool_config *vb_pool_cfg);
hi_s32 hi_mpi_vb_destroy_pool(hi_vb_pool pool);
hi_vb_blk hi_mpi_vb_get_block(hi_vb_pool pool, hi_u64 blk_size, const hi_char *mmz_name);
hi_s32 hi_mpi_vb_release_block(hi_vb_blk vb_blk);
hi_void* hi_mpi_vb_handle_to_addr(hi_vb_blk vb_blk);
hi_s32 hi_mpi_vb_block_copyref(hi_vb_blk vb_blk);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif // #ifndef HI_DVPP_VB_H_
