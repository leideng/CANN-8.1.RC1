# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.

class VllmHelper:
    vllm_req_map = {}

    @classmethod
    def int_req(cls, rid):
        if cls.vllm_req_map.get(rid) is None:
            cls.vllm_req_map[rid] = {}
            cls.vllm_req_map[rid]['batch_iter'] = 0
            cls.vllm_req_map[rid]['receiveToken'] = 0
        return cls.vllm_req_map

    @classmethod
    def add_req_batch_iter(cls, rid, iter_size):
        if cls.vllm_req_map.get(rid) is not None and cls.vllm_req_map[rid]['receiveToken'] == 0:
            cls.vllm_req_map[rid]['receiveToken'] = iter_size
        elif cls.vllm_req_map.get(rid) is not None:
            cls.vllm_req_map[rid]['batch_iter'] += iter_size
        else:
            VllmHelper.int_req(rid)
            cls.vllm_req_map[rid]['receiveToken'] = iter_size
        return cls.vllm_req_map[rid]['batch_iter']
