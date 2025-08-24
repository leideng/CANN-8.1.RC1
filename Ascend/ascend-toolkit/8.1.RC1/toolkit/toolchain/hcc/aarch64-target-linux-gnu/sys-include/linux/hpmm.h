/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * Copyright @ Huawei Technologies Co., Ltd. 2022-2022. ALL rights reserved.
 * Description: Header file for hpmm module.
 */

#ifndef _LINUX_HPMM_MOD_H
#define _LINUX_HPMM_MOD_H

#include <linux/ioctl.h>
#include <linux/types.h>

enum hpmm_mem_type {
	HPMM_PERSISTENT_MEMORY		= 0x000001UL,
	HPMM_VOLATILE_MEMORY		= 0x000002UL,
	HPMM_HIGH_RELIABLE_MEMORY	= 0x000004UL,
	HPMM_LOW_RELIABLE_MEMORY	= 0x000008UL,
	HPMM_MEMORY_TYPE_MASK		= 0x00000fUL,
};

/* Clear the pagetables and memory in persistent device specified by pid */
#define HPMM_CLEAR_PID			(1 << 0)
/* Clear all the memory in persistent device */
#define HPMM_CLEAR_ALL			(1 << 1)
/* Clear the meta data of the bbu memory */
#define HPMM_CLEAR_METADATA		(1 << 2)
#define HPMM_CLEAR_VALID		(HPMM_CLEAR_PID | HPMM_CLEAR_METADATA | HPMM_CLEAR_ALL)

/* Recover the bbu memory data specified by pid */
#define HPMM_RECOVER_CUR		(1 << 3)
/* Recover the bbu memory through linear mapping process */
#define HPMM_RECOVER_LINEAR_MAPPING	(1 << 4)
#define HPMM_RECOVER_VALID		(HPMM_RECOVER_CUR | HPMM_RECOVER_LINEAR_MAPPING)

struct linear_mapping {
	unsigned long start_addr;	/* the virtual start address to be mapped */
	unsigned long length;		/* the virtual addr length to be mapped */
	unsigned long offset;		/* Offset from the physical start address to be mapped */
};

struct hpmm_mem_info {
	/* the type of hpmm node */
	int type;
	/* unique id of process */
	unsigned long uuid;
	/* process pid need to be op on */
	int	pid;
	/* numa node */
	int node;
	/* For clear memory in persistent device
	 * HPMM_CLEAR_MEMORY:
	 * HPMM_CLEAR_PID | HPMM_CLEAR_ALL | HPMM_CLEAR_METADATA
	 * For recover memory in persistent device
	 * HPMM_RECOVER_MEMORY:
	 * HPMM_RECOVER_PID | HPMM_RECOVER_CUR| HPMM_RECOVER_LINEAR_MAPPING
	 */
	int flags;

	/* the virtual addr and length to be mapped */
	struct linear_mapping mapping_addr;
};

struct uce_record {
	/* pfn correspond to the UCE page */
	unsigned long pfn;
	/* UUID of the process where the UCE occurs */
	unsigned long uuid;
	/* virtual addr mapped to the uce page */
	unsigned long va;
	/* UCE type */
	unsigned long type;
};

struct hpmm_uce_records {
	/* UCE type */
	int type;
	/* count of uce num */
	int count;
	/* UUID of the process where the UCE occurs */
	unsigned long uuid;
	/* uce record area */
	struct uce_record records[128];
};

/* HPMM ioctl parameters */
#define HPMM_MAGIC 0xaa

enum _HPMM_IOCTL_CMD {
	_HPMM_SET_MEMORY_TYPE,
	_HPMM_CLEAR_MEMORY,
	_HPMM_RECOVER_MEMORY,
	_HPMM_THP_MEMORY_MAPPING,
	_HPMM_QUERY_UCE_RECORD,
	_HPMM_CLEAR_UCE_RECORD,
	_HPMM_IOCTL_MAX_NR,
};

#define HPMM_SET_MEMORY_TYPE		\
	_IOW(HPMM_MAGIC, _HPMM_SET_MEMORY_TYPE, struct hpmm_mem_info)
#define HPMM_CLEAR_MEMORY		\
	_IOW(HPMM_MAGIC, _HPMM_CLEAR_MEMORY, struct hpmm_mem_info)
#define HPMM_RECOVER_MEMORY		\
	_IOW(HPMM_MAGIC, _HPMM_RECOVER_MEMORY, struct hpmm_mem_info)
#define HPMM_THP_MEMORY_MAPPING	\
	_IOW(HPMM_MAGIC, _HPMM_THP_MEMORY_MAPPING, struct hpmm_mem_info)
#define HPMM_QUERY_MEMORY_INFO		\
	_IOW(HPMM_MAGIC, _HPMM_QUERY_MEMORY_INFO, struct hpmm_mem_info)
#define HPMM_QUERY_UCE_RECORD	\
	_IOW(HPMM_MAGIC, _HPMM_QUERY_UCE_RECORD, struct hpmm_uce_records)
#define HPMM_CLEAR_UCE_RECORD	\
	_IOW(HPMM_MAGIC, _HPMM_CLEAR_UCE_RECORD, struct hpmm_uce_records)
#endif /* _LINUX_HPMM_MOD_H */

