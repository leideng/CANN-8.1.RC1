/* SPDX-License-Identifier: ((GPL-2.0 WITH Linux-syscall-note) */

#ifndef IB_USER_UHW_H
#define IB_USER_UHW_H

#include <linux/types.h>

#define IB_UHW_MAX_QP_NUM 9

struct ib_uhw_abi_hdr {
	__u32 cmd;
	__u16 in;
	__u16 out;
};

struct ib_uhw_send {
	__aligned_u64 data;
	__u32 hostid;
	__u32 private_data_len;
};

struct ib_uhw_get {
	__aligned_u64 response;
};

struct ib_uhw_get_resp {
	__u32 event;
	__u32 data_len;
	__u8 data[0];
};

struct ib_uhw_set_conn_sts {
	__u32 num;
	__u32 data[IB_UHW_MAX_QP_NUM];
};

struct ib_uhw_disconnect {
	__be64 tid;
	__be32 local_comm_id;
	__be32 remote_comm_id;
	__u32 local_qpn;
	__u32 remote_qpn;
};

struct ib_uhw_query_qplist {
	__aligned_u64 response;
	__u64 data;
	__u32 data_len;
};

struct ib_uhw_query_qplist_resp {
	__u32 start;
	__u32 num;
};

struct ib_uhw_query_master_qpc {
	__aligned_u64 response;
	__u32 qpn;
};

struct ib_uhw_query_master_qpc_resp {
	__u8 host;
	__u32 admin_qp;
	__s32 qp_state;
};

struct ib_uhw_query_hostid {
	__aligned_u64 response;
};

struct ib_uhw_query_hostid_resp {
	__u32 hostid;
};

struct ib_uhw_dd_cfg {
	__u8 dd_id;
	__u8 default_ctrl;
	__u16 op;
	__u32 data_len;
	__u64 data;
};

struct ib_uhw_shard_cfg {
	__u16 op;
	__u32 entry_data_len;
	__u64 entry_data;
};

enum {
	IB_UHW_CMD_SEND_MSG,
	IB_UHW_CMD_GET_EVENT,
	IB_UHW_CMD_SET_CONN_STS,
	IB_UHW_CMD_DISCONNECT,
	IB_UHW_CMD_QUERY_QPLIST,
	IB_UHW_CMD_QUERY_MASTER_QPC,
	IB_UHW_CMD_QUERY_HOSTID,
	IB_UHW_CMD_READY_TO_USE,
	IB_UHW_CMD_SET_DD_CFG,
	IB_UHW_CMD_SET_SHARD_CFG,
	IB_UHW_CMD_SWITCH_IO,
	IB_UHW_CMD_RSVD1,
};
#endif /* IB_USER_UHW_H */
