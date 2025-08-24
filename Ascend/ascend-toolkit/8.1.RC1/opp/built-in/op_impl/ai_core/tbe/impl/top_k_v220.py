#!/usr/bin/env python
# coding: utf-8
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
top_k_v220
"""
from tbe import tik


# 'pylint: disable=too-few-public-methods
class Setting:
    """
    define some setting values
    """
    # a sort struct contains 4 fp16
    num_fp16_struc = 4
    # a sort struct contains 2 fp32
    num_fp32_struc = 2
    # part size befor first exhausted merge
    part_8192 = 8192
    # the last remained part size, 10w%8192 = 1696
    part_1696 = 1696


# 'pylint: disable=too-many-arguments,line-too-long,too-many-locals
def _sort_8192(tik_instance, data_gm, index_gm, out_gm, out_gm2):
    """
    sort 10w data to 13 part sorted list, 1~12 part has 8192 element, and last has 1696 elements
    every elements has 4 fp16 or 2 fp32
    do not need exhausted mode merge
    """
    with tik_instance.new_stmt_scope():
        data_ub = tik_instance.Tensor("float16", (Setting.part_8192, ), name="data_ub", scope=tik.scope_ubuf)
        index_ub = tik_instance.Tensor("uint32", (Setting.part_8192, ), name="index_ub", scope=tik.scope_ubuf)
        sorted_ub = tik_instance.Tensor(
            "float16", (Setting.part_8192 * Setting.num_fp16_struc, ), name="sorted_ub_1", scope=tik.scope_ubuf)
        sorted_ub_2 = tik_instance.Tensor(
            "float16", (Setting.part_8192 * Setting.num_fp16_struc, ), name="sorted_ub_2", scope=tik.scope_ubuf)
        valid_assist_len = 2048
        assist_ub_fp16 = tik_instance.Tensor(
            "float16", (valid_assist_len, ), name="assist_ub_fp16", scope=tik.scope_ubuf)
        assist_ub_int32 = tik_instance.Tensor(
            "int32", (valid_assist_len, ), name="assist_ub_int32", scope=tik.scope_ubuf)

        # `10W = 8192*12+1696`
        tik_instance.data_move(assist_ub_fp16, index_gm, 0, 1, valid_assist_len * 2 // 32, 0, 0)
        tik_instance.vconv(64, "round", assist_ub_int32, assist_ub_fp16, valid_assist_len // 64, 1, 1, 8, 4)
        burst_len = Setting.part_8192 * 2 // 32
        part_32 = 32
        part_128 = 128
        part_512 = 512
        part_2048 = 2048
        with tik_instance.for_range(0, 12) as i:
            tik_instance.data_move(data_ub, data_gm[i * Setting.part_8192], 0, 1, burst_len, 0, 0)
            with tik_instance.for_range(0, 4) as j:
                tik_instance.vadds(64, index_ub[j * valid_assist_len].reinterpret_cast_to("int32"), assist_ub_int32,
                                   i * Setting.part_8192 + j * valid_assist_len, valid_assist_len // 64, 1, 1, 8, 8)

            # `sort len = 32, repeat = 8192 // 32 = 256`
            repeat = 255
            tik_instance.vsort32(sorted_ub, data_ub, index_ub, repeat)
            # repeat can not larger than 255, so split repat 256 into 255 + 1
            repeat = 1
            tik_instance.vsort32(sorted_ub[255 * part_32 * Setting.num_fp16_struc], data_ub[255 * part_32],
                                 index_ub[255 * part_32], repeat)

            # `merge_len = 32`
            # `repeat 64 (4*32*64=8192)`
            repeat = 64
            tik_instance.vmrgsort(
                sorted_ub_2,
                (sorted_ub, sorted_ub[part_32 * Setting.num_fp16_struc],
                 sorted_ub[part_32 * 2 * Setting.num_fp16_struc], sorted_ub[part_32 * 3 * Setting.num_fp16_struc]),
                part_32, False, repeat)

            # `merge len = 32*4 =128, repeat 16`
            repeat = 16
            tik_instance.vmrgsort(sorted_ub, (sorted_ub_2, sorted_ub_2[part_128 * Setting.num_fp16_struc],
                                              sorted_ub_2[part_128 * 2 * Setting.num_fp16_struc],
                                              sorted_ub_2[part_128 * 3 * Setting.num_fp16_struc]), part_128, False,
                                  repeat)

            # `merge len = 128*4 = 512, repeat 4`
            repeat = 4
            tik_instance.vmrgsort(
                sorted_ub_2,
                (sorted_ub, sorted_ub[part_512 * Setting.num_fp16_struc],
                 sorted_ub[part_512 * 2 * Setting.num_fp16_struc], sorted_ub[part_512 * 3 * Setting.num_fp16_struc]),
                part_512, False, repeat)

            # `merge len = 512*4 = 2048, repeat 1`
            repeat = 1
            tik_instance.vmrgsort(sorted_ub, (sorted_ub_2, sorted_ub_2[part_2048 * Setting.num_fp16_struc],
                                              sorted_ub_2[part_2048 * 2 * Setting.num_fp16_struc],
                                              sorted_ub_2[part_2048 * 3 * Setting.num_fp16_struc]), part_2048, False,
                                  repeat)

            burst_len = Setting.part_8192 * Setting.num_fp16_struc * 2 // 32
            tik_instance.data_move(out_gm[i * Setting.part_8192 * Setting.num_fp16_struc], sorted_ub, 0, 1, burst_len,
                                   0, 0)

        # sort 1696
        tik_instance.data_move(data_ub, data_gm[12 * Setting.part_8192], 0, 1, Setting.part_1696 * 2 // 32, 0, 0)
        # for fp32 vadds, to process 1696 element repeat is ceil_div(1696/64) = 27
        tik_instance.vadds(64, index_ub.reinterpret_cast_to("int32"), assist_ub_int32, 12 * Setting.part_8192, 27, 1, 1,
                           8, 8)
        # for vsort32, to process 1696 element repeat is ceil_div(1696/32) = 53
        tik_instance.vsort32(sorted_ub, data_ub, index_ub, 53)

        # `first merge 32*4*13+32 = 1696, repeat 13`
        repeat = 13
        tik_instance.vmrgsort(
            sorted_ub_2,
            (sorted_ub, sorted_ub[part_32 * Setting.num_fp16_struc], sorted_ub[part_32 * 2 * Setting.num_fp16_struc],
             sorted_ub[part_32 * 3 * Setting.num_fp16_struc]), part_32, False, repeat)
        burst_len = part_32 * 2 * Setting.num_fp16_struc // 32
        tik_instance.data_move(sorted_ub_2[part_32 * 4 * 13 * Setting.num_fp16_struc],
                               sorted_ub[part_32 * 4 * 13 * Setting.num_fp16_struc], 0, 1, burst_len, 0, 0)

        # `second merge 128*4*3 + 128 + 32 = 1696, repeat 3`
        repeat = 3
        tik_instance.vmrgsort(
            sorted_ub,
            (sorted_ub_2, sorted_ub_2[part_128 * Setting.num_fp16_struc],
             sorted_ub_2[part_128 * 2 * Setting.num_fp16_struc], sorted_ub_2[part_128 * 3 * Setting.num_fp16_struc]),
            part_128, False, repeat)
        tik_instance.vmrgsort(
            sorted_ub[part_128 * 4 * 3 * Setting.num_fp16_struc],
            (sorted_ub_2[part_128 * 4 * 3 * Setting.num_fp16_struc],
             sorted_ub_2[part_128 * 4 * 3 * Setting.num_fp16_struc + part_128 * Setting.num_fp16_struc]),
            (part_128, part_32), False, 1)

        # third merge, 512+512+512+160=1696, last part is 128+32=160
        tik_instance.vmrgsort(
            sorted_ub_2,
            (sorted_ub, sorted_ub[part_512 * Setting.num_fp16_struc], sorted_ub[part_512 * 2 * Setting.num_fp16_struc],
             sorted_ub[part_512 * 3 * Setting.num_fp16_struc]), (part_512, part_512, part_512, 160), False, 1)

        burst_len = Setting.part_1696 * Setting.num_fp16_struc * 2 // 32
        tik_instance.data_move(out_gm2[12 * Setting.part_8192 * Setting.num_fp16_struc], sorted_ub_2, 0, 1, burst_len,
                               0, 0)


def _move_last_part(tik_instance, workspace_src, workspace_dst, part_size):
    """
    move sorted struc form workspace_src to workspace_dst
    """
    with tik_instance.new_stmt_scope():
        sorted_ub = tik_instance.Tensor(
            "float16", (part_size * Setting.num_fp16_struc, ), name="sorted_ub_last_part", scope=tik.scope_ubuf)
        tik_instance.data_move(sorted_ub, workspace_src, 0, 1, part_size // Setting.num_fp16_struc, 0, 0)
        tik_instance.data_move(workspace_dst, sorted_ub, 0, 1, part_size // Setting.num_fp16_struc, 0, 0)


# 'pylint: disable=too-many-locals,too-many-arguments,line-too-long,too-many-statements,unused-variable
def _merge_exhausted_mode(tik_instance, workspace_src, workspace_dst, merge_list, part_list):
    """
    use exhausted mode vmrg to merge 4 sorted list into one sorted list
    """
    part_n1, part_n2, part_n3, part_n4 = part_list
    len_n1, len_n2, len_n3, len_n4 = merge_list
    with tik_instance.new_stmt_scope():

        def update_scalar(total_merged, to_merge, merged, load_data, part, total_to_merge):
            """
            update scalar func
            """
            total_merged.set_as(total_merged + merged)
            with tik_instance.if_scope(to_merge == merged):
                with tik_instance.if_scope(total_merged < total_to_merge):
                    to_merge.set_as(part)
                    load_data.set_as(1)
                with tik_instance.else_scope():
                    to_merge.set_as(0)
                    load_data.set_as(0)
            with tik_instance.else_scope():
                to_merge.set_as(to_merge - merged)
                load_data.set_as(0)

        sorted_ub_1 = tik_instance.Tensor(
            "float16", (part_n1 * Setting.num_fp16_struc, ), name="sorted_ub_1", scope=tik.scope_ubuf)
        sorted_ub_2 = tik_instance.Tensor(
            "float16", (part_n2 * Setting.num_fp16_struc, ), name="sorted_ub_2", scope=tik.scope_ubuf)
        sorted_ub_3 = tik_instance.Tensor(
            "float16", (part_n3 * Setting.num_fp16_struc, ), name="sorted_ub_3", scope=tik.scope_ubuf)
        sorted_ub_4 = tik_instance.Tensor(
            "float16", (part_n4 * Setting.num_fp16_struc, ), name="sorted_ub_4", scope=tik.scope_ubuf)
        sorted_ub_dst = tik_instance.Tensor(
            "float16", ((part_n1 + part_n2 + part_n3 + part_n4) * Setting.num_fp16_struc, ),
            name="sorted_ub_dst",
            scope=tik.scope_ubuf)
        # one block, 16 fp16
        align_ub = tik_instance.Tensor("float16", (16, ), name="align_ub", scope=tik.scope_ubuf)

        total_merged_n1 = tik_instance.Scalar(dtype="int32", name="total_merged_n1", init_value=0)
        total_merged_n2 = tik_instance.Scalar(dtype="int32", name="total_merged_n2", init_value=0)
        total_merged_n3 = tik_instance.Scalar(dtype="int32", name="total_merged_n3", init_value=0)
        total_merged_n4 = tik_instance.Scalar(dtype="int32", name="total_merged_n4", init_value=0)
        merged_n1 = tik_instance.Scalar(dtype="int32", name="merged_n1")
        merged_n2 = tik_instance.Scalar(dtype="int32", name="merged_n2")
        merged_n3 = tik_instance.Scalar(dtype="int32", name="merged_n3")
        merged_n4 = tik_instance.Scalar(dtype="int32", name="merged_n4")
        to_merge_n1 = tik_instance.Scalar(dtype="int32", name="to_merge_n1", init_value=part_n1)
        to_merge_n2 = tik_instance.Scalar(dtype="int32", name="to_merge_n2", init_value=part_n2)
        to_merge_n3 = tik_instance.Scalar(dtype="int32", name="to_merge_n3", init_value=part_n3)
        to_merge_n4 = tik_instance.Scalar(dtype="int32", name="to_merge_n4", init_value=part_n4)
        # 50 is an empirical value, this value will not be reached, usually early break
        loop_cnt = tik_instance.Scalar(dtype="int32", name="loop_cnt", init_value=50)
        load_data_n1 = tik_instance.Scalar(dtype="int32", name="load_data_n1", init_value=1)
        load_data_n2 = tik_instance.Scalar(dtype="int32", name="load_data_n2", init_value=1)
        load_data_n3 = tik_instance.Scalar(dtype="int32", name="load_data_n3", init_value=1)
        load_data_n4 = tik_instance.Scalar(dtype="int32", name="load_data_n4", init_value=1)

        with tik_instance.for_range(0, loop_cnt) as i:
            n1_offset = total_merged_n1 * Setting.num_fp16_struc
            n2_offset = len_n1 * Setting.num_fp16_struc + total_merged_n2 * Setting.num_fp16_struc
            n3_offset = (len_n1 + len_n2) * Setting.num_fp16_struc + total_merged_n3 * Setting.num_fp16_struc
            n4_offset = (len_n1 + len_n2 + len_n3) * Setting.num_fp16_struc + total_merged_n4 * Setting.num_fp16_struc
            dst_offset = (
                total_merged_n1 + total_merged_n2 + total_merged_n3 + total_merged_n4) * Setting.num_fp16_struc
            ub_n1_offset = (part_n1 - to_merge_n1) * Setting.num_fp16_struc
            ub_n2_offset = (part_n2 - to_merge_n2) * Setting.num_fp16_struc
            ub_n3_offset = (part_n3 - to_merge_n3) * Setting.num_fp16_struc
            ub_n4_offset = (part_n4 - to_merge_n4) * Setting.num_fp16_struc

            with tik_instance.if_scope(load_data_n1):
                tik_instance.data_move(sorted_ub_1, workspace_src[n1_offset], 0, 1, part_n1 // 4, 0, 0)
            with tik_instance.if_scope(load_data_n2):
                tik_instance.data_move(sorted_ub_2, workspace_src[n2_offset], 0, 1, part_n2 // 4, 0, 0)
            with tik_instance.if_scope(load_data_n3):
                tik_instance.data_move(sorted_ub_3, workspace_src[n3_offset], 0, 1, part_n3 // 4, 0, 0)
            with tik_instance.if_scope(load_data_n4):
                tik_instance.data_move(sorted_ub_4, workspace_src[n4_offset], 0, 1, part_n4 // 4, 0, 0)

            with tik_instance.if_scope(
                    tik.all(total_merged_n1 < len_n1, total_merged_n2 < len_n2, total_merged_n3 < len_n3,
                            total_merged_n4 < len_n4)):
                # four route merge
                src_list = (sorted_ub_1[ub_n1_offset], sorted_ub_2[ub_n2_offset], sorted_ub_3[ub_n3_offset],
                            sorted_ub_4[ub_n4_offset])
                to_merge_list = (to_merge_n1, to_merge_n2, to_merge_n3, to_merge_n4)
                merged_list = (merged_n1, merged_n2, merged_n3, merged_n4)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1,
                                       (merged_n1 + merged_n2 + merged_n3 + merged_n4 + 3) // 4, 0, 0)
                update_scalar(total_merged_n1, to_merge_n1, merged_n1, load_data_n1, part_n1, len_n1)
                update_scalar(total_merged_n2, to_merge_n2, merged_n2, load_data_n2, part_n2, len_n2)
                update_scalar(total_merged_n3, to_merge_n3, merged_n3, load_data_n3, part_n3, len_n3)
                update_scalar(total_merged_n4, to_merge_n4, merged_n4, load_data_n4, part_n4, len_n4)
            with tik_instance.elif_scope(
                    tik.all(total_merged_n1 < len_n1, total_merged_n2 < len_n2, total_merged_n3 < len_n3)):
                # (n1, n2, n3) merge
                src_list = (sorted_ub_1[ub_n1_offset], sorted_ub_2[ub_n2_offset], sorted_ub_3[ub_n3_offset])
                to_merge_list = (to_merge_n1, to_merge_n2, to_merge_n3)
                merged_list = (merged_n1, merged_n2, merged_n3)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1,
                                       (merged_n1 + merged_n2 + merged_n3 + 3) // 4, 0, 0)
                update_scalar(total_merged_n1, to_merge_n1, merged_n1, load_data_n1, part_n1, len_n1)
                update_scalar(total_merged_n2, to_merge_n2, merged_n2, load_data_n2, part_n2, len_n2)
                update_scalar(total_merged_n3, to_merge_n3, merged_n3, load_data_n3, part_n3, len_n3)
            with tik_instance.elif_scope(
                    tik.all(total_merged_n1 < len_n1, total_merged_n2 < len_n2, total_merged_n4 < len_n4)):
                # (n1, n2, n4) merge
                src_list = (sorted_ub_1[ub_n1_offset], sorted_ub_2[ub_n2_offset], sorted_ub_4[ub_n4_offset])
                to_merge_list = (to_merge_n1, to_merge_n2, to_merge_n4)
                merged_list = (merged_n1, merged_n2, merged_n4)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1,
                                       (merged_n1 + merged_n2 + merged_n4 + 3) // 4, 0, 0)
                update_scalar(total_merged_n1, to_merge_n1, merged_n1, load_data_n1, part_n1, len_n1)
                update_scalar(total_merged_n2, to_merge_n2, merged_n2, load_data_n2, part_n2, len_n2)
                update_scalar(total_merged_n4, to_merge_n4, merged_n4, load_data_n4, part_n4, len_n4)
            with tik_instance.elif_scope(
                    tik.all(total_merged_n1 < len_n1, total_merged_n3 < len_n3, total_merged_n4 < len_n4)):
                # (n1, n3, n4) merge
                src_list = (sorted_ub_1[ub_n1_offset], sorted_ub_3[ub_n3_offset], sorted_ub_4[ub_n4_offset])
                to_merge_list = (to_merge_n1, to_merge_n3, to_merge_n4)
                merged_list = (merged_n1, merged_n3, merged_n4)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1,
                                       (merged_n1 + merged_n3 + merged_n4 + 3) // 4, 0, 0)
                update_scalar(total_merged_n1, to_merge_n1, merged_n1, load_data_n1, part_n1, len_n1)
                update_scalar(total_merged_n3, to_merge_n3, merged_n3, load_data_n3, part_n3, len_n3)
                update_scalar(total_merged_n4, to_merge_n4, merged_n4, load_data_n4, part_n4, len_n4)
            with tik_instance.elif_scope(
                    tik.all(total_merged_n2 < len_n2, total_merged_n3 < len_n3, total_merged_n4 < len_n4)):
                # (n2 , n3, n4) merge
                src_list = (sorted_ub_2[ub_n2_offset], sorted_ub_3[ub_n3_offset], sorted_ub_4[ub_n4_offset])
                to_merge_list = (to_merge_n2, to_merge_n3, to_merge_n4)
                merged_list = (merged_n2, merged_n3, merged_n4)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1,
                                       (merged_n2 + merged_n3 + merged_n4 + 3) // 4, 0, 0)
                update_scalar(total_merged_n2, to_merge_n2, merged_n2, load_data_n2, part_n2, len_n2)
                update_scalar(total_merged_n3, to_merge_n3, merged_n3, load_data_n3, part_n3, len_n3)
                update_scalar(total_merged_n4, to_merge_n4, merged_n4, load_data_n4, part_n4, len_n4)
            with tik_instance.elif_scope(tik.all(total_merged_n1 < len_n1, total_merged_n2 < len_n2)):
                # (n1, n2) merge
                src_list = (sorted_ub_1[ub_n1_offset], sorted_ub_2[ub_n2_offset])
                to_merge_list = (to_merge_n1, to_merge_n2)
                merged_list = (merged_n1, merged_n2)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1, (merged_n1 + merged_n2 + 3) // 4,
                                       0, 0)
                update_scalar(total_merged_n1, to_merge_n1, merged_n1, load_data_n1, part_n1, len_n1)
                update_scalar(total_merged_n2, to_merge_n2, merged_n2, load_data_n2, part_n2, len_n2)
            with tik_instance.elif_scope(tik.all(total_merged_n1 < len_n1, total_merged_n3 < len_n3)):
                # (n1, n3) merge
                src_list = (sorted_ub_1[ub_n1_offset], sorted_ub_3[ub_n3_offset])
                to_merge_list = (to_merge_n1, to_merge_n3)
                merged_list = (merged_n1, merged_n3)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1, (merged_n1 + merged_n3 + 3) // 4,
                                       0, 0)
                update_scalar(total_merged_n1, to_merge_n1, merged_n1, load_data_n1, part_n1, len_n1)
                update_scalar(total_merged_n3, to_merge_n3, merged_n3, load_data_n3, part_n3, len_n3)
            with tik_instance.elif_scope(tik.all(total_merged_n1 < len_n1, total_merged_n4 < len_n4)):
                # (n1, n4) merge
                src_list = (sorted_ub_1[ub_n1_offset], sorted_ub_4[ub_n4_offset])
                to_merge_list = (to_merge_n1, to_merge_n4)
                merged_list = (merged_n1, merged_n4)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1, (merged_n1 + merged_n4 + 3) // 4,
                                       0, 0)
                update_scalar(total_merged_n1, to_merge_n1, merged_n1, load_data_n1, part_n1, len_n1)
                update_scalar(total_merged_n4, to_merge_n4, merged_n4, load_data_n4, part_n4, len_n4)
            with tik_instance.elif_scope(tik.all(total_merged_n2 < len_n2, total_merged_n3 < len_n3)):
                # (n2, n3) merge
                src_list = (sorted_ub_2[ub_n2_offset], sorted_ub_3[ub_n3_offset])
                to_merge_list = (to_merge_n2, to_merge_n3)
                merged_list = (merged_n2, merged_n3)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1, (merged_n2 + merged_n3 + 3) // 4,
                                       0, 0)
                update_scalar(total_merged_n2, to_merge_n2, merged_n2, load_data_n2, part_n2, len_n2)
                update_scalar(total_merged_n3, to_merge_n3, merged_n3, load_data_n3, part_n3, len_n3)
            with tik_instance.elif_scope(tik.all(total_merged_n2 < len_n2, total_merged_n4 < len_n4)):
                # (n2, n4) merge
                src_list = (sorted_ub_2[ub_n2_offset], sorted_ub_4[ub_n4_offset])
                to_merge_list = (to_merge_n2, to_merge_n4)
                merged_list = (merged_n2, merged_n4)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1, (merged_n2 + merged_n4 + 3) // 4,
                                       0, 0)
                update_scalar(total_merged_n2, to_merge_n2, merged_n2, load_data_n2, part_n2, len_n2)
                update_scalar(total_merged_n4, to_merge_n4, merged_n4, load_data_n4, part_n4, len_n4)
            with tik_instance.elif_scope(tik.all(total_merged_n3 < len_n3, total_merged_n4 < len_n4)):
                # (n3, n4) merge
                src_list = (sorted_ub_3[ub_n3_offset], sorted_ub_4[ub_n4_offset])
                to_merge_list = (to_merge_n3, to_merge_n4)
                merged_list = (merged_n3, merged_n4)
                tik_instance.vmrgsort(sorted_ub_dst, src_list, to_merge_list, True, 1, merged_list)
                tik_instance.data_move(workspace_dst[dst_offset], sorted_ub_dst, 0, 1, (merged_n3 + merged_n4 + 3) // 4,
                                       0, 0)
                update_scalar(total_merged_n3, to_merge_n3, merged_n3, load_data_n3, part_n3, len_n3)
                update_scalar(total_merged_n4, to_merge_n4, merged_n4, load_data_n4, part_n4, len_n4)
            with tik_instance.else_scope():
                # may left last one list
                with tik_instance.if_scope(total_merged_n1 < len_n1):
                    with tik_instance.if_scope(to_merge_n1 % 4 != 0):
                        with tik_instance.for_range(0, (to_merge_n1 % 4) * Setting.num_fp16_struc) as j:
                            align_ub[j] = sorted_ub_1[ub_n1_offset + j]
                        tik_instance.data_move(workspace_dst[dst_offset], align_ub, 0, 1, 1, 0, 0)
                    tik_instance.data_move(workspace_dst[dst_offset + (to_merge_n1 % 4) * Setting.num_fp16_struc],
                                           sorted_ub_1[ub_n1_offset + (to_merge_n1 % 4) * Setting.num_fp16_struc], 0, 1,
                                           to_merge_n1 // 4, 0, 0)
                    merged_n1.set_as(to_merge_n1)
                    update_scalar(total_merged_n1, to_merge_n1, merged_n1, load_data_n1, part_n1, len_n1)
                with tik_instance.elif_scope(total_merged_n2 < len_n2):
                    with tik_instance.if_scope(to_merge_n2 % 4 != 0):
                        with tik_instance.for_range(0, (to_merge_n2 % 4) * Setting.num_fp16_struc) as j:
                            align_ub[j] = sorted_ub_2[ub_n2_offset + j]
                        tik_instance.data_move(workspace_dst[dst_offset], align_ub, 0, 1, 1, 0, 0)
                    tik_instance.data_move(workspace_dst[dst_offset + (to_merge_n2 % 4) * Setting.num_fp16_struc],
                                           sorted_ub_2[ub_n2_offset + (to_merge_n2 % 4) * Setting.num_fp16_struc], 0, 1,
                                           to_merge_n2 // 4, 0, 0)
                    merged_n2.set_as(to_merge_n2)
                    update_scalar(total_merged_n2, to_merge_n2, merged_n2, load_data_n2, part_n2, len_n2)
                with tik_instance.elif_scope(total_merged_n3 < len_n3):
                    with tik_instance.if_scope(to_merge_n3 % 4 != 0):
                        with tik_instance.for_range(0, (to_merge_n3 % 4) * Setting.num_fp16_struc) as j:
                            align_ub[j] = sorted_ub_3[ub_n3_offset + j]
                        tik_instance.data_move(workspace_dst[dst_offset], align_ub, 0, 1, 1, 0, 0)
                    tik_instance.data_move(workspace_dst[dst_offset + (to_merge_n3 % 4) * Setting.num_fp16_struc],
                                           sorted_ub_3[ub_n3_offset + (to_merge_n3 % 4) * Setting.num_fp16_struc], 0, 1,
                                           to_merge_n3 // 4, 0, 0)
                    merged_n3.set_as(to_merge_n3)
                    update_scalar(total_merged_n3, to_merge_n3, merged_n3, load_data_n3, part_n3, len_n3)
                with tik_instance.elif_scope(total_merged_n4 < len_n4):
                    with tik_instance.if_scope(to_merge_n4 % 4 != 0):
                        with tik_instance.for_range(0, (to_merge_n4 % 4) * Setting.num_fp16_struc) as j:
                            align_ub[j] = sorted_ub_4[ub_n4_offset + j]
                        tik_instance.data_move(workspace_dst[dst_offset], align_ub, 0, 1, 1, 0, 0)
                    tik_instance.data_move(workspace_dst[dst_offset + (to_merge_n4 % 4) * Setting.num_fp16_struc],
                                           sorted_ub_4[ub_n4_offset + (to_merge_n4 % 4) * Setting.num_fp16_struc], 0, 1,
                                           to_merge_n4 // 4, 0, 0)
                    merged_n4.set_as(to_merge_n4)
                    update_scalar(total_merged_n4, to_merge_n4, merged_n4, load_data_n4, part_n4, len_n4)
                with tik_instance.else_scope():
                    # all data is sorted, modify the loop limitation, break the loop
                    loop_cnt.set_as(0)


def _extract_10w(tik_instance, sorted_gm, out_data_gm, out_index_gm):
    """
    extract value and index from sorted struct.
    """
    with tik_instance.new_stmt_scope():
        sorted_ub = tik_instance.Tensor(
            "float16", (Setting.part_8192 * Setting.num_fp16_struc, ), name="sorted_ub", scope=tik.scope_ubuf)
        data_ub = tik_instance.Tensor("float16", (Setting.part_8192, ), name="data_ub", scope=tik.scope_ubuf)
        index_ub = tik_instance.Tensor("uint32", (Setting.part_8192, ), name="index_ub", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, 12) as i:
            tik_instance.data_move(sorted_ub, sorted_gm[i * Setting.part_8192 * Setting.num_fp16_struc], 0, 1,
                                   Setting.part_8192 * Setting.num_fp16_struc * 2 // 32, 0, 0)
            tik_instance.vreduce(
                Setting.part_8192 * Setting.num_fp16_struc,
                data_ub,
                sorted_ub,
                3, # src1_pattern
                1, # repeat_times, count mode do not use repeat
                1, # src0_blk_stride
                8, # src0_rep_stride
                0, # src1_rep_stride
                0, # stride_unit
                None, # rsvd_scalar
                "counter") # mask_mode
            tik_instance.vreduce(
                Setting.part_8192 * Setting.num_fp32_struc,
                index_ub,
                sorted_ub.reinterpret_cast_to("uint32"),
                2, # src1_pattern
                1, # repeat_times, count mode do not use repeat
                1, # src0_blk_stride
                8, # src0_rep_stride
                0, # src1_rep_stride
                0, # stride_unit
                None, # rsvd_scalar
                "counter") # mask_mode
            tik_instance.data_move(out_data_gm[i * Setting.part_8192], data_ub, 0, 1, Setting.part_8192 // 16, 0, 0)
            tik_instance.data_move(out_index_gm[i * Setting.part_8192], index_ub, 0, 1, Setting.part_8192 // 8, 0, 0)

        tik_instance.data_move(sorted_ub, sorted_gm[12 * Setting.part_8192 * Setting.num_fp16_struc], 0, 1,
                               Setting.part_1696 * Setting.num_fp16_struc * 2 // 32, 0, 0)
        tik_instance.vreduce(
            Setting.part_1696 * Setting.num_fp16_struc,
            data_ub,
            sorted_ub,
            3, # src1_pattern
            1, # repeat_times, count mode do not use repeat
            1, # src0_blk_stride
            8, # src0_rep_stride
            0, # src1_rep_stride
            0, # stride_unit
            None, # rsvd_scalar
            "counter") # mask_mode
        tik_instance.vreduce(
            Setting.part_1696 * Setting.num_fp32_struc,
            index_ub,
            sorted_ub.reinterpret_cast_to("uint32"),
            2, # src1_pattern
            1, # repeat_times, count mode do not use repeat
            1, # src0_blk_stride
            8, # src0_rep_stride
            0, # src1_rep_stride
            0, # stride_unit
            None, # rsvd_scalar
            "counter") # mask_mode
        tik_instance.data_move(out_data_gm[12 * Setting.part_8192], data_ub, 0, 1, Setting.part_1696 // 16, 0, 0)
        tik_instance.data_move(out_index_gm[12 * Setting.part_8192], index_ub, 0, 1, Setting.part_1696 // 8, 0, 0)


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,redefined-builtin
def build_topk_10w_v220(input_tensor,
                        indices_tensor,
                        out_tensor,
                        out_indices_tensor,
                        k,
                        sorted=True,
                        dim=-1,
                        largest=True,
                        kernel_name='top_k'):
    """
    build topk 10w data  sorted for v220, 10w = 8192 * 12 +1696
    first use vsort and none exhaust vmrg to get 12 part sorted list with 8192 elements
    and 1 part sorted list with 1696 elements
    every elements contains 4 fp16 or 2 fp32
    then use exhausted mode vmrg to get 3 part sorted list with 32768 elements
    finaly used exhausted mode vmrg go merge 3 part 32768 list and 1 part 1696 list to a sorted list of size 10w
    at the end , extract value and index from sorted  struct.
    """
    tik_instance = tik.Tik()
    column = 100000
    part_size = 8192
    last_part_size = column % part_size
    ub_part_size = 2048
    assist_len = 8192
    data_gm = tik_instance.Tensor("float16", (column, ), name="data_gm", scope=tik.scope_gm)
    indices_gm = tik_instance.Tensor("float16", (assist_len, ), name="indices_gm", scope=tik.scope_gm)
    workspace_1 = tik_instance.Tensor(
        "float16", (column * Setting.num_fp16_struc, ), name="workspace1", scope=tik.scope_gm, is_workspace=True)
    workspace_2 = tik_instance.Tensor(
        "float16", (column * Setting.num_fp16_struc, ), name="workspace2", scope=tik.scope_gm, is_workspace=True)
    out_data_gm = tik_instance.Tensor("float16", (column, ), name="out_data_gm", scope=tik.scope_gm)
    out_indices_gm = tik_instance.Tensor("int32", (column, ), name="out_indices_ub", scope=tik.scope_gm)

    # `10W = 8192*12 + 1696`
    _sort_8192(tik_instance, data_gm, indices_gm, workspace_1, workspace_1)
    # every part size if 8192
    count_list = (part_size, part_size, part_size, part_size)
    # every ub size if 2048
    ub_count_list = (ub_part_size, ub_part_size, ub_part_size, ub_part_size)
    _merge_exhausted_mode(tik_instance, workspace_1, workspace_2, count_list, ub_count_list)
    # 8192 is part size, first 4 is num_fp16_struc, second 4 is four route merge
    element_offset = 8192 * 4 * 4
    _merge_exhausted_mode(tik_instance, workspace_1[element_offset], workspace_2[element_offset], count_list,
                          ub_count_list)
    element_offset = 8192 * 4 * 4 * 2
    _merge_exhausted_mode(tik_instance, workspace_1[element_offset], workspace_2[element_offset], count_list,
                          ub_count_list)
    element_offset = 8192 * 4 * 4 * 3
    _move_last_part(tik_instance, workspace_1[element_offset], workspace_2[element_offset], last_part_size)
    count_list = (part_size * 4, part_size * 4, part_size * 4, last_part_size)
    ub_count_list = (ub_part_size, ub_part_size, ub_part_size, last_part_size)
    _merge_exhausted_mode(tik_instance, workspace_2, workspace_1, count_list, ub_count_list)
    _extract_10w(tik_instance, workspace_1, out_data_gm, out_indices_gm)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[data_gm, indices_gm], outputs=[out_data_gm, out_indices_gm])
