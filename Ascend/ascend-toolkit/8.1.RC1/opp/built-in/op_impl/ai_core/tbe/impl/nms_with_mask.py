#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Huawei Technologies Co., Ltd
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
nms_with_mask
"""

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.dynamic.nms_with_mask_large_n import NMSLargeN


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """

    # shape's dim of input must be 2
    INPUT_DIM = 2
    # scaling factor
    DOWN_FACTOR = 0.054395
    # vector unit can compute 256 bytes in one cycle
    BYTES_ONE_CYCLE_VECTOR = 256
    # process 128 proposals at a time for fp16
    BURST_PROPOSAL_NUM = 128
    # RPN compute 16 proposals per iteration
    RPN_PROPOSAL_NUM = 16
    # the coordinate column contains x1,y1,x2,y2
    COORD_COLUMN_NUM = 4
    # valid proposal column contains x1,y1,x2,y2,score
    VALID_COLUMN_NUM = 5
    # each region proposal contains eight elements
    ELEMENT_NUM = 8
    # data align size, also size of one block
    CONFIG_DATA_ALIGN = 32
    REPEAT_TIMES_MAX = 255
    # next_nonzero_idx shape0 is 16 for 32B aligned, 16 is enough
    SHAPE_NEXT_NONZERO = 16
    # mask used for vcmax in update_next_nonzero, 256//2=128, fixed fp16 here but enough for input_dtype
    MASK_VCMAX_FP16 = 128
    # size of some data types
    INT8_SIZE = tbe_platform.get_bit_len('int8') // 8
    UINT8_SIZE = tbe_platform.get_bit_len('uint8') // 8
    UINT16_SIZE = tbe_platform.get_bit_len('uint16') // 8
    FP16_SIZE = tbe_platform.get_bit_len('float16') // 8
    INT32_SIZE = tbe_platform.get_bit_len('int32') // 8
    UINT32_SIZE = tbe_platform.get_bit_len('uint32') // 8
    FP32_SIZE = tbe_platform.get_bit_len('fp32') // 8


def _get_soc_version():
    return tbe_platform.get_soc_spec("SHORT_SOC_VERSION")


def _ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


def _ceiling(value, factor):
    """
    Compute the smallest integer value that is greater than or equal to value and can divide factor
    """
    result = (value + (factor - 1)) // factor * factor
    return result


# 'pylint: disable=invalid-name
def _get_src_tensor(ib):
    """
    Produce two tensors with all zero or all one

    Parameters
    ----------
    ib: TIK API

    Returns
    -------
    src0_ub: the tensor with zero
    src1_ub: the tensor with one
    """
    one_scalar = ib.Scalar(dtype="float16", name="one_scalar", init_value=1.0)
    zero_scalar = ib.Scalar(dtype="float16", name="zero_scalar", init_value=0.0)
    src0_ub = ib.Tensor("float16", (Constant.BURST_PROPOSAL_NUM,), name="src0_ub", scope=tik.scope_ubuf)
    src1_ub = ib.Tensor("float16", (Constant.BURST_PROPOSAL_NUM,), name="src1_ub", scope=tik.scope_ubuf)
    ib.vector_dup(128, src0_ub, zero_scalar, 1, 1, 8)
    ib.vector_dup(128, src1_ub, one_scalar, 1, 1, 8)

    return src0_ub, src1_ub


# 'pylint: disable=invalid-name
def _get_reduced_proposal(ib, out_proposal, output_proposals_final, in_proposal, coord_addr):
    """
    Reduce input proposal when input boxes out of range.

    Parameters
    ----------
    ib: TIK API

    out_proposal: output proposal after reduce

    output_proposals_final: output proposal with boxes and scores, support [128,5]

    in_proposal: input proposal with boxes and scores, support [128,8]

    coord_addr: intermediate proposal after reshape

    Returns
    -------
    None
    """
    # extract original coordinates
    if tbe_platform.api_check_support("tik.vreduce", "float16") and tbe_platform.api_check_support(
            "tik.v4dtrans", "float16"):
        with ib.for_range(0, Constant.VALID_COLUMN_NUM) as i:
            ib.vextract(coord_addr[Constant.BURST_PROPOSAL_NUM * i],
                        in_proposal, Constant.BURST_PROPOSAL_NUM // Constant.RPN_PROPOSAL_NUM, i)
        # transpose 5*burst_proposal_num to burst_proposal_num*5, output boxes and scores
        ib.v4dtrans(True, output_proposals_final, coord_addr, Constant.BURST_PROPOSAL_NUM, Constant.VALID_COLUMN_NUM)
    else:
        with ib.for_range(0, Constant.COORD_COLUMN_NUM) as i:
            ib.vextract(coord_addr[Constant.BURST_PROPOSAL_NUM * i],
                        in_proposal, Constant.BURST_PROPOSAL_NUM // Constant.RPN_PROPOSAL_NUM, i)

    # coordinate multiplied by down_factor to prevent out of range
    ib.vmuls(128, coord_addr, coord_addr, Constant.DOWN_FACTOR, 4, 1, 1, 8, 8)

    if _get_soc_version() in ("Ascend310", "Ascend310B"):
        # add 1 for x1 and y1 to resist rpn_offset in viou/vrpac
        ib.vadds(128, coord_addr[0], coord_addr[0], 1.0, 1, 1, 1, 8, 8)
        ib.vadds(128, coord_addr[Constant.BURST_PROPOSAL_NUM * 1],
                 coord_addr[Constant.BURST_PROPOSAL_NUM * 1], 1.0, 1, 1, 1, 8, 8)

    # compose new proposals
    with ib.for_range(0, Constant.COORD_COLUMN_NUM) as i:
        ib.vconcat(out_proposal, coord_addr[Constant.BURST_PROPOSAL_NUM * i],
                   Constant.BURST_PROPOSAL_NUM // Constant.RPN_PROPOSAL_NUM, i)


# 'pylint: disable=too-many-locals,too-many-arguments
def _tik_func_nms_single_core_multithread(input_shape, thresh, total_output_proposal_num, kernel_name_var):
    """
    Compute output boxes after non-maximum suppression.

    Parameters
    ----------
    input_shape: dict
        shape of input boxes, including proposal boxes and corresponding confidence scores

    thresh: float
        iou threshold

    total_output_proposal_num: int
        the number of output proposal boxes

    kernel_name_var: str
        cce kernel name

    Returns
    -------
    tik_instance: TIK API
    """
    tik_instance = tik.Tik()
    if _get_soc_version() not in ("Ascend310", "Ascend310B"):
        tik_instance.set_rpn_offset(0.0)
    total_input_proposal_num, _ = input_shape
    proposals = tik_instance.Tensor("float16", (total_input_proposal_num, Constant.ELEMENT_NUM),
                                    name="in_proposals",
                                    scope=tik.scope_gm)
    support_vreduce = tbe_platform.api_check_support("tik.vreduce", "float16")
    support_v4dtrans = tbe_platform.api_check_support("tik.v4dtrans", "float16")
    # output shape is [N,5]
    ret = tik_instance.Tensor("float16", (total_output_proposal_num, Constant.VALID_COLUMN_NUM), name="out_proposals",
                              scope=tik.scope_gm)
    out_index = tik_instance.Tensor("int32", (total_output_proposal_num,), name="out_index", scope=tik.scope_gm)
    out_mask = tik_instance.Tensor("uint8", (total_output_proposal_num,), name="out_mask", scope=tik.scope_gm)
    # address is 32B aligned
    output_index_ub = tik_instance.Tensor("int32", (Constant.BURST_PROPOSAL_NUM,),
                                          name="output_index_ub", scope=tik.scope_ubuf)
    output_mask_ub = tik_instance.Tensor("uint8", (Constant.BURST_PROPOSAL_NUM,),
                                         name="output_mask_ub", scope=tik.scope_ubuf)
    output_proposals_ub = tik_instance.Tensor("float16", (Constant.BURST_PROPOSAL_NUM, Constant.VALID_COLUMN_NUM),
                                              name="output_proposals_ub",
                                              scope=tik.scope_ubuf)

    # init tensor every 128 proposals
    fresh_proposals_ub = tik_instance.Tensor("float16", (Constant.BURST_PROPOSAL_NUM, Constant.ELEMENT_NUM),
                                             name="fresh_proposals_ub",
                                             scope=tik.scope_ubuf)
    temp_reduced_proposals_ub = tik_instance.Tensor("float16", (Constant.BURST_PROPOSAL_NUM, Constant.ELEMENT_NUM),
                                                    name="temp_reduced_proposals_ub",
                                                    scope=tik.scope_ubuf)
    tik_instance.vector_dup(128, temp_reduced_proposals_ub[0], 0, 8, 1, 8)

    # init middle selected proposals
    selected_reduced_proposals_ub = tik_instance.Tensor(
        "float16", (_ceiling(total_output_proposal_num, Constant.RPN_PROPOSAL_NUM), Constant.ELEMENT_NUM),
        name="selected_reduced_proposals_ub",
        scope=tik.scope_ubuf)
    # init middle selected area
    selected_area_ub = tik_instance.Tensor("float16", (_ceiling(total_output_proposal_num,
                                                                Constant.RPN_PROPOSAL_NUM),),
                                           name="selected_area_ub",
                                           scope=tik.scope_ubuf)
    # init middle sup_vec
    sup_vec_ub = tik_instance.Tensor("uint16", (_ceiling(total_output_proposal_num,
                                                         Constant.RPN_PROPOSAL_NUM),),
                                     name="sup_vec_ub",
                                     scope=tik.scope_ubuf)
    if total_output_proposal_num >= 128:
        tik_instance.vector_dup(128, sup_vec_ub, 1, total_output_proposal_num // 128, 1, 8)
    if total_output_proposal_num % 128 > 0:
        tik_instance.vector_dup(total_output_proposal_num % 128,
                                sup_vec_ub[total_output_proposal_num // 128 * 128], 1, 1, 1, 8)

    # init nms tensor
    temp_area_ub = tik_instance.Tensor("float16", (Constant.BURST_PROPOSAL_NUM,),
                                       name="temp_area_ub", scope=tik.scope_ubuf)
    temp_iou_ub = tik_instance.Tensor("float16",
                                      (_ceiling(total_output_proposal_num,
                                                Constant.RPN_PROPOSAL_NUM), Constant.RPN_PROPOSAL_NUM),
                                      name="temp_iou_ub",
                                      scope=tik.scope_ubuf)
    temp_join_ub = tik_instance.Tensor("float16",
                                       (_ceiling(total_output_proposal_num,
                                                 Constant.RPN_PROPOSAL_NUM), Constant.RPN_PROPOSAL_NUM),
                                       name="temp_join_ub",
                                       scope=tik.scope_ubuf)
    temp_sup_matrix_ub = tik_instance.Tensor("uint16", (_ceiling(total_output_proposal_num,
                                                                 Constant.RPN_PROPOSAL_NUM),),
                                             name="temp_sup_matrix_ub",
                                             scope=tik.scope_ubuf)
    temp_sup_vec_ub = tik_instance.Tensor("uint16", (Constant.BURST_PROPOSAL_NUM,),
                                          name="temp_sup_vec_ub",
                                          scope=tik.scope_ubuf)

    if support_vreduce and support_v4dtrans:
        output_mask_f16 = tik_instance.Tensor("float16", (Constant.BURST_PROPOSAL_NUM,),
                                              name="output_mask_f16",
                                              scope=tik.scope_ubuf)
        data_zero, data_one = _get_src_tensor(tik_instance)

        middle_reduced_proposals = tik_instance.Tensor("float16", (Constant.BURST_PROPOSAL_NUM,
                                                                   Constant.ELEMENT_NUM),
                                                       name="middle_reduced_proposals",
                                                       scope=tik.scope_ubuf)

        # init v200 reduce param
        nms_tensor_pattern = tik_instance.Tensor(dtype="uint16",
                                                 shape=(Constant.ELEMENT_NUM,),
                                                 name="nms_tensor_pattern",
                                                 scope=tik.scope_ubuf)
        # init ori coord
        coord_addr = tik_instance.Tensor("float16", (Constant.VALID_COLUMN_NUM,
                                                     Constant.BURST_PROPOSAL_NUM),
                                         name="coord_addr",
                                         scope=tik.scope_ubuf)
        # init reduce zoom coord
        zoom_coord_reduce = tik_instance.Tensor("float16", (Constant.COORD_COLUMN_NUM,
                                                            Constant.BURST_PROPOSAL_NUM),
                                                name="zoom_coord_reduce",
                                                scope=tik.scope_ubuf)
        # init reduce num
        num_nms = tik_instance.Scalar(dtype="uint32")
    else:
        # init ori coord
        coord_addr = tik_instance.Tensor("float16", (Constant.COORD_COLUMN_NUM,
                                                     Constant.BURST_PROPOSAL_NUM),
                                         name="coord_addr",
                                         scope=tik.scope_ubuf)
        mask = tik_instance.Scalar(dtype="uint8")

    # variables
    selected_proposals_cnt = tik_instance.Scalar(dtype="uint16")
    selected_proposals_cnt.set_as(0)
    handling_proposals_cnt = tik_instance.Scalar(dtype="uint16")
    handling_proposals_cnt.set_as(0)
    left_proposal_cnt = tik_instance.Scalar(dtype="uint16")
    left_proposal_cnt.set_as(total_input_proposal_num)
    scalar_zero = tik_instance.Scalar(dtype="uint16")
    scalar_zero.set_as(0)
    sup_vec_ub[0].set_as(scalar_zero)

    # handle 128 proposals every time
    with tik_instance.for_range(0, _ceil_div(total_input_proposal_num, Constant.BURST_PROPOSAL_NUM),
                                thread_num=1) as burst_index:
        # update counter
        with tik_instance.if_scope(left_proposal_cnt < Constant.BURST_PROPOSAL_NUM):
            handling_proposals_cnt.set_as(left_proposal_cnt)
        with tik_instance.else_scope():
            handling_proposals_cnt.set_as(Constant.BURST_PROPOSAL_NUM)

        tik_instance.data_move(fresh_proposals_ub[0],
                               proposals[burst_index * Constant.BURST_PROPOSAL_NUM * Constant.ELEMENT_NUM], 0, 1,
                               _ceil_div(handling_proposals_cnt *
                                         Constant.RPN_PROPOSAL_NUM, Constant.CONFIG_DATA_ALIGN), 0, 0, 0)
        # reduce fresh proposal
        _get_reduced_proposal(tik_instance, temp_reduced_proposals_ub, output_proposals_ub, fresh_proposals_ub,
                              coord_addr)
        # calculate the area of reduced-proposal
        tik_instance.vrpac(temp_area_ub[0], temp_reduced_proposals_ub[0],
                           _ceil_div(handling_proposals_cnt, Constant.RPN_PROPOSAL_NUM))
        # start to update iou and or area from the first 16 proposal and get suppression vector 16 by 16 proposal
        length = tik_instance.Scalar(dtype="uint16")
        length.set_as(_ceiling(selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM))
        # clear temp_sup_vec_ub
        tik_instance.vector_dup(128, temp_sup_vec_ub[0], 1,
                                temp_sup_vec_ub.shape[0] // Constant.BURST_PROPOSAL_NUM, 1, 8)

        with tik_instance.for_range(0, _ceil_div(handling_proposals_cnt, Constant.RPN_PROPOSAL_NUM)) as i:
            length.set_as(length + Constant.RPN_PROPOSAL_NUM)
            # calculate intersection of tempReducedProposals and selReducedProposals
            tik_instance.viou(temp_iou_ub, selected_reduced_proposals_ub,
                              temp_reduced_proposals_ub[i * Constant.RPN_PROPOSAL_NUM, 0],
                              _ceil_div(selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM))
            # calculate intersection of tempReducedProposals and tempReducedProposals(include itself)
            tik_instance.viou(temp_iou_ub[_ceiling(selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM), 0],
                              temp_reduced_proposals_ub,
                              temp_reduced_proposals_ub[i * Constant.RPN_PROPOSAL_NUM, 0], i + 1)
            # calculate join of tempReducedProposals and selReducedProposals
            tik_instance.vaadd(temp_join_ub, selected_area_ub, temp_area_ub[i * Constant.RPN_PROPOSAL_NUM],
                               _ceil_div(selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM))
            # calculate intersection of tempReducedProposals and tempReducedProposals(include itself)
            tik_instance.vaadd(temp_join_ub[_ceiling(selected_proposals_cnt,
                                                     Constant.RPN_PROPOSAL_NUM), 0], temp_area_ub,
                               temp_area_ub[i * Constant.RPN_PROPOSAL_NUM], i + 1)
            # calculate join*(thresh/(1+thresh))
            tik_instance.vmuls(128, temp_join_ub, temp_join_ub, thresh,
                               _ceil_div(length, Constant.ELEMENT_NUM), 1, 1, 8, 8)
            # compare and generate suppression matrix
            tik_instance.vcmpv_gt(temp_sup_matrix_ub, temp_iou_ub,
                                  temp_join_ub, _ceil_div(length, Constant.ELEMENT_NUM), 1, 1,
                                  8, 8)
            # generate suppression vector
            rpn_cor_ir = tik_instance.set_rpn_cor_ir(0)
            # non-diagonal
            rpn_cor_ir = tik_instance.rpn_cor(temp_sup_matrix_ub[0], sup_vec_ub[0], 1, 1,
                                              _ceil_div(selected_proposals_cnt, Constant.RPN_PROPOSAL_NUM))
            with tik_instance.if_scope(i > 0):
                rpn_cor_ir = tik_instance.rpn_cor(
                    temp_sup_matrix_ub[_ceiling(selected_proposals_cnt,
                                                Constant.RPN_PROPOSAL_NUM)], temp_sup_vec_ub, 1, 1, i)
            # diagonal
            tik_instance.rpn_cor_diag(temp_sup_vec_ub[i * Constant.RPN_PROPOSAL_NUM],
                                      temp_sup_matrix_ub[length - Constant.RPN_PROPOSAL_NUM], rpn_cor_ir)

        if support_vreduce and support_v4dtrans:
            with tik_instance.for_range(0, handling_proposals_cnt) as i:
                output_index_ub[i].set_as(i + burst_index * Constant.BURST_PROPOSAL_NUM)

            # get the mask tensor of temp_sup_vec_ub
            temp_tensor = temp_sup_vec_ub.reinterpret_cast_to("float16")
            cmpmask = tik_instance.vcmp_eq(128, temp_tensor, data_zero, 1, 1)
            tik_instance.mov_cmpmask_to_tensor(nms_tensor_pattern.reinterpret_cast_to("uint16"), cmpmask)

            # save the area corresponding to these filtered proposals for the next nms use
            tik_instance.vreduce(128, selected_area_ub[selected_proposals_cnt],
                                 temp_area_ub, nms_tensor_pattern, 1, 1,
                                 8, 0, 0, num_nms, "counter")
            # sup_vec_ub set as 0
            with tik_instance.if_scope(num_nms > 0):
                tik_instance.vector_dup(num_nms, sup_vec_ub[selected_proposals_cnt], 0, 1, 1, 1)

            # save the filtered proposal for next nms use
            tik_instance.vector_dup(128, zoom_coord_reduce, 0, 4, 1, 8)
            tik_instance.vector_dup(128, middle_reduced_proposals, 0, 8, 1, 8)
            with tik_instance.for_range(0, Constant.COORD_COLUMN_NUM) as i:
                tik_instance.vreduce(128, zoom_coord_reduce[i, 0],
                                     coord_addr[i, 0], nms_tensor_pattern, 1, 1, 8, 0, 0,
                                     None, "counter")
            with tik_instance.for_range(0, Constant.COORD_COLUMN_NUM) as i:
                tik_instance.vconcat(middle_reduced_proposals, zoom_coord_reduce[i, 0],
                                     _ceil_div(num_nms, Constant.RPN_PROPOSAL_NUM), i)
            tik_instance.data_move(selected_reduced_proposals_ub[selected_proposals_cnt, 0],
                                   middle_reduced_proposals,
                                   0, 1, _ceil_div(num_nms * Constant.ELEMENT_NUM, Constant.RPN_PROPOSAL_NUM), 0, 0)

            selected_proposals_cnt.set_as(selected_proposals_cnt + num_nms)

            # convert the output mask from binary to decimal
            tik_instance.vsel(128, 0, output_mask_f16, cmpmask, data_one, data_zero, 1, 1, 1, 1, 8, 8, 8)
            tik_instance.vec_conv(128, "none", output_mask_ub, output_mask_f16, 1, 8, 8)
        else:
            with tik_instance.for_range(0, handling_proposals_cnt) as i:
                with tik_instance.for_range(0, Constant.VALID_COLUMN_NUM) as j:
                    # update selOriginalProposals_ub
                    output_proposals_ub[i, j].set_as(fresh_proposals_ub[i, j])
                output_index_ub[i].set_as(i + burst_index * Constant.BURST_PROPOSAL_NUM)
                with tik_instance.if_scope(temp_sup_vec_ub[i] == 0):
                    with tik_instance.for_range(0, Constant.ELEMENT_NUM) as j:
                        # update selected_reduced_proposals_ub
                        selected_reduced_proposals_ub[selected_proposals_cnt,
                                                      j].set_as(temp_reduced_proposals_ub[i, j])
                    # update selected_area_ub
                    selected_area_ub[selected_proposals_cnt].set_as(temp_area_ub[i])
                    # update sup_vec_ub
                    sup_vec_ub[selected_proposals_cnt].set_as(scalar_zero)
                    mask.set_as(1)
                    output_mask_ub[i].set_as(mask)
                    # update counter
                    selected_proposals_cnt.set_as(selected_proposals_cnt + 1)
                with tik_instance.else_scope():
                    mask.set_as(0)
                    output_mask_ub[i].set_as(mask)

        left_proposal_cnt.set_as(left_proposal_cnt - handling_proposals_cnt)
        # mov target proposals to out - mte3
        tik_instance.data_move(ret[burst_index * Constant.BURST_PROPOSAL_NUM, 0], output_proposals_ub, 0, 1,
                               _ceil_div(handling_proposals_cnt *
                                         Constant.VALID_COLUMN_NUM, Constant.RPN_PROPOSAL_NUM), 0, 0, 0)
        tik_instance.data_move(out_index[burst_index * Constant.BURST_PROPOSAL_NUM], output_index_ub, 0, 1,
                               _ceil_div(handling_proposals_cnt, Constant.ELEMENT_NUM), 0, 0, 0)
        tik_instance.data_move(out_mask[burst_index * Constant.BURST_PROPOSAL_NUM], output_mask_ub, 0, 1,
                               _ceil_div(handling_proposals_cnt, Constant.CONFIG_DATA_ALIGN), 0, 0, 0)
    if _get_soc_version() not in ("Ascend310", "Ascend310B"):
        tik_instance.set_rpn_offset(1.0)
    tik_instance.BuildCCE(kernel_name=kernel_name_var,
                          inputs=[proposals],
                          outputs=[ret, out_index, out_mask],
                          output_files_path=None,
                          enable_l2=False)
    return tik_instance


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def nms_with_mask(box_scores, selected_boxes, selected_idx, selected_mask, iou_thr=0.5, kernel_name="nms_with_mask"):
    """
    algorithm: nms_with_mask

    find the best target bounding box and eliminate redundant bounding boxes

    Parameters
    ----------
    box_scores: dict
        2-D shape and dtype of input tensor, only support [N, 8]
        including proposal boxes and corresponding confidence scores

    selected_boxes: dict
        2-D shape and dtype of output boxes tensor, only support [N,5]
        including proposal boxes and corresponding confidence scores

    selected_idx: dict
        the index of input proposal boxes

    selected_mask: dict
        the symbol judging whether the output proposal boxes is valid

    iou_thr: float
        iou threshold

    kernel_name: str
        cce kernel name, default value is "nms_with_mask"

    Returns
    -------
    None
    """
    # check shape
    input_shape = box_scores.get("shape")
    para_check.check_shape(input_shape, min_dim=1,
                           min_rank=Constant.INPUT_DIM, max_rank=Constant.INPUT_DIM, param_name="box_scores")

    # new soc branch
    if tbe_platform.api_check_support("tik.vreduce",
                                      "float16") and not tbe_platform.api_check_support("tik.vaadd", "float16"):
        return _nms_with_mask_basic_api(box_scores, selected_boxes, selected_idx, selected_mask, iou_thr, kernel_name)

    input_dtype = box_scores.get("dtype").lower()

    # check dtype
    check_list = ("float16")
    para_check.check_dtype(input_dtype, check_list, param_name="box_scores")

    support_vreduce = tbe_platform.api_check_support("tik.vreduce", "float16")
    support_v4dtrans = tbe_platform.api_check_support("tik.v4dtrans", "float16")

    # Considering the memory space of Unified_Buffer
    fp16_size = tbe_platform.get_bit_len("float16") // 8
    int32_size = tbe_platform.get_bit_len("int32") // 8
    uint8_size = tbe_platform.get_bit_len("uint8") // 8
    uint16_size = tbe_platform.get_bit_len("uint16") // 8
    ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # output shape is [N,5], including x1,y1,x2,y2,scores
    burst_size = Constant.BURST_PROPOSAL_NUM * int32_size + Constant.BURST_PROPOSAL_NUM * uint8_size + \
                 Constant.BURST_PROPOSAL_NUM * Constant.VALID_COLUMN_NUM * fp16_size
    # compute shape is [N,8]
    selected_size = _ceiling(input_shape[0],
                             Constant.RPN_PROPOSAL_NUM) * Constant.ELEMENT_NUM * fp16_size + _ceiling(
        input_shape[0], Constant.RPN_PROPOSAL_NUM) *\
                    fp16_size + _ceiling(input_shape[0], Constant.RPN_PROPOSAL_NUM) * uint16_size
    # intermediate calculation results
    temp_iou_size = _ceiling(input_shape[0], Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM * fp16_size
    temp_join_size = _ceiling(input_shape[0], Constant.RPN_PROPOSAL_NUM) * Constant.RPN_PROPOSAL_NUM * fp16_size
    temp_sup_matrix_size = _ceiling(input_shape[0], Constant.RPN_PROPOSAL_NUM) * uint16_size
    temp_sup_vec_size = Constant.BURST_PROPOSAL_NUM * uint16_size
    temp_area_size = Constant.BURST_PROPOSAL_NUM * fp16_size
    temp_reduced_proposals_size = Constant.BURST_PROPOSAL_NUM * Constant.ELEMENT_NUM * fp16_size
    temp_size = temp_iou_size + temp_join_size + temp_sup_matrix_size + temp_sup_vec_size + \
                temp_area_size + temp_reduced_proposals_size
    # input shape is [N,8]
    fresh_size = Constant.BURST_PROPOSAL_NUM * Constant.ELEMENT_NUM * fp16_size
    if support_vreduce and support_v4dtrans:
        coord_size = Constant.BURST_PROPOSAL_NUM * Constant.VALID_COLUMN_NUM * fp16_size
        middle_reduced_proposals_size = Constant.BURST_PROPOSAL_NUM * Constant.ELEMENT_NUM * fp16_size
        src_tensor_size = Constant.BURST_PROPOSAL_NUM * fp16_size + Constant.BURST_PROPOSAL_NUM * fp16_size
        output_mask_f16_size = Constant.BURST_PROPOSAL_NUM * fp16_size
        nms_tensor_pattern_size = Constant.ELEMENT_NUM * uint16_size
        zoom_coord_reduce = Constant.BURST_PROPOSAL_NUM * Constant.COORD_COLUMN_NUM * fp16_size
        v200_size = output_mask_f16_size + src_tensor_size + middle_reduced_proposals_size + \
                    nms_tensor_pattern_size + zoom_coord_reduce
        used_size = burst_size + selected_size + temp_size + fresh_size + coord_size + v200_size
    else:
        coord_size = Constant.BURST_PROPOSAL_NUM * Constant.COORD_COLUMN_NUM * fp16_size
        used_size = burst_size + selected_size + temp_size + fresh_size + coord_size

    if used_size > ub_size_bytes:
        error_manager_vector.raise_err_check_params_rules(
            kernel_name, "the number of input boxes out of range(%d B)" % ub_size_bytes, "used size", used_size)

    if input_shape[1] != Constant.ELEMENT_NUM:
        error_manager_vector.raise_err_check_params_rules(kernel_name, "the 2nd-dim of input boxes must be equal to 8",
                                                          "box_scores.shape", input_shape)

    output_size, _ = input_shape
    iou_thr = iou_thr / (1 + iou_thr)
    return _tik_func_nms_single_core_multithread(input_shape, iou_thr, output_size, kernel_name)


class _NMSHelper():
    """
    handle all input proposals, e.g. N may > 128

    idea:
                        sn's mask: sn+1     andMask(means: which idx still exists in dst), vand or vmul
        init:           [1 1 1 1 1 1 1 1]   [1 1 1 1 1 1 1 1]  init state, from 0.elem, now 0.elem
        s0's result:    [0 0 1 0 1 1 0 1]   [0 0 1 0 1 1 0 1]  after one loop, get s1 is result of s0, now 2.elem
        s2's result:    [0 0 0 1 1 0 1 0]   [0 0 0 0 1 0 0 0]  now 4.elem
        s4's result:    [0 0 0 0 0 0 0 0]   [0 0 0 0 0 0 0 0]  end

        dst: 0.2.4. elem, so [1 0 1 0 1 0 0 0]
        so far, get output_mask_ub

    note:
        output mask: uint8
        output index: int32
        output proposals: float16 or float32
    """

    def __init__(self, tik_instance, all_inp_proposals_gm_1980, input_shape, input_dtype, iou_thres):
        """
        Parameters:
        ----------
        tik_instance: tik instance
        all_inp_proposals_gm_1980: size is N*8
        input_shape: corresponds to all_inp_proposals_ub_1980
        input_dtype: new soc supports: float16 and float32
        iou_thres: iou threshold, one box is valid if its iou is lower than the threshold

        Returns
        -------
        None
        """
        self.tik_instance = tik_instance
        self.input_dtype = input_dtype

        if input_dtype == 'float16':
            self.input_bytes_each_elem = Constant.FP16_SIZE
            self.input_vector_mask_max = Constant.BURST_PROPOSAL_NUM
        elif input_dtype == 'float32':
            self.input_bytes_each_elem = Constant.FP32_SIZE
            self.input_vector_mask_max = Constant.BURST_PROPOSAL_NUM // 2

        self.data_type = 'float32'
        self.bytes_each_elem = 4
        self.vector_mask_max = 64
        self.all_inp_proposals_gm_1980 = all_inp_proposals_gm_1980

        self.N, _ = input_shape
        self.ceil_n = _ceiling(self.N, self.vector_mask_max)
        # note: N canbe used in size, but not for def tensor, should use ceil_n
        self.input_size = self.N * Constant.ELEMENT_NUM
        self.iou_thres_factor = iou_thres / (iou_thres + 1)

        # cache frequently used
        self.negone_int8_scalar = tik_instance.Scalar('int8', 'negone_int8_scalar', init_value=-1)
        self.zero_int8_scalar = tik_instance.Scalar('int8', 'zero_int8_scalar', init_value=0)
        self.zero_int16_scalar = tik_instance.Scalar('int16', 'zero_int16_scalar', init_value=0)
        self.one_uint8_scalar = tik_instance.Scalar('uint8', 'one_uint8_scalar', init_value=1)
        self.one_int16_scalar = tik_instance.Scalar('int16', 'one_int16_scalar', init_value=1)

        # scalar: zero of dtype, one
        self.zero_datatype_scalar = tik_instance.Scalar(self.data_type, 'zero_dtype_scalar', init_value=0.)
        self.one_datatype_scalar = tik_instance.Scalar(self.data_type, 'one_dtype_scalar', init_value=1.)

        # note: defed size need to 32b aligned
        self.x1_ub = tik_instance.Tensor(shape=(self.ceil_n,),
                                         dtype=self.data_type, name='x1_ub', scope=tik.scope_ubuf)
        self.x2_ub = tik_instance.Tensor(shape=(self.ceil_n,),
                                         dtype=self.data_type, name='x2_ub', scope=tik.scope_ubuf)
        self.y1_ub = tik_instance.Tensor(shape=(self.ceil_n,),
                                         dtype=self.data_type, name='y1_ub', scope=tik.scope_ubuf)
        self.y2_ub = tik_instance.Tensor(shape=(self.ceil_n,),
                                         dtype=self.data_type, name='y2_ub', scope=tik.scope_ubuf)

        # 1980's input => new soc's output_mask_ub
        self.all_inp_proposals_ub_1980_fp32 = tik_instance.Tensor('float32', (self.ceil_n, Constant.ELEMENT_NUM),
                                                                  name="all_inp_proposals_ub_1980_fp32",
                                                                  scope=tik.scope_ubuf)
        # def tmp ub tensor
        self.tmp_tensor_ub_fp16 = tik_instance.Tensor('float16', (self.ceil_n,), tik.scope_ubuf, 'tmp_tensor_ub_fp16')
        self.tmp_tensor_ub_fp16_burst = tik_instance.Tensor('float16', (Constant.BURST_PROPOSAL_NUM,), tik.scope_ubuf,
                                                            'tmp_tensor_ub_fp16_burst')

        self._input_trans()

        # cache area, calc once is enough
        self.total_areas_ub = None

        # [0] stores next nonzero idx, shape[0]=16 same as idx_fp16_ub.shape in order to conv
        self.next_nonzero_int32_idx = tik_instance.Tensor('int32', (Constant.SHAPE_NEXT_NONZERO,), tik.scope_ubuf,
                                                          'next_nonzero_int32_idx')

        # init for valid mask
        self._init_for_valid_mask()

        # selected_idx_ub generate
        self.selected_idx_ub = self._selected_idx_gen()

        # init for vcmax
        self._init_for_vcmax()

        # for inter
        self._init_for_inter()

        self.area_cur = self.tik_instance.Scalar(self.data_type, 'area_cur_scalar')

        # output mask, dtype is int8 fixed
        self.output_mask_int8_ub = self.tik_instance.Tensor('int8', (self.valid_mask_size_int8,), tik.scope_ubuf,
                                                            "output_mask_int_ub")

        self._init_for_cmpmask2bitmask()

        # scaling
        self._scaling()

    def _init_for_inter(self):
        """
        init tensors for inter

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.xx1 = self.tik_instance.Tensor(self.data_type, (self.ceil_n,), tik.scope_ubuf, "xx1_ub")
        self.yy1 = self.tik_instance.Tensor(self.data_type, (self.ceil_n,), tik.scope_ubuf, "yy1_ub")
        # xx2 is reused several times
        self.xx2 = self.tik_instance.Tensor(self.data_type, (self.ceil_n,), tik.scope_ubuf, "xx2_ub")
        self.x1i = self.tik_instance.Scalar(self.data_type, name='x1i_scalar')
        self.y1i = self.tik_instance.Scalar(self.data_type, name='y1i_scalar')
        self.x2i = self.tik_instance.Scalar(self.data_type, name='x2i_scalar')
        self.y2i = self.tik_instance.Scalar(self.data_type, name='y2i_scalar')

    def _init_for_cmpmask2bitmask(self):
        """
        for cmpmask2bitmask, fp16 fixed is OK, this is used in one repeat, so Constant.BURST_PROPOSAL_NUM below is OK

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.output_mask_f16 = self.tik_instance.Tensor('float16',
                                                        (Constant.BURST_PROPOSAL_NUM,), name="output_mask_f16",
                                                        scope=tik.scope_ubuf)
        zero_fp16_scalar = self.tik_instance.Scalar(dtype="float16", name="zero_scalar", init_value=0.0)
        one_fp16_scalar = self.tik_instance.Scalar(dtype="float16", name="one_scalar", init_value=1.0)
        self.data_fp16_zero = self.tik_instance.Tensor("float16", (Constant.BURST_PROPOSAL_NUM,), name="data_zero",
                                                       scope=tik.scope_ubuf)
        self.data_fp16_one = self.tik_instance.Tensor("float16", (Constant.BURST_PROPOSAL_NUM,), name="data_one",
                                                      scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(Constant.BURST_PROPOSAL_NUM, self.data_fp16_zero, zero_fp16_scalar, 1, 1, 8)
        self.tik_instance.vector_dup(Constant.BURST_PROPOSAL_NUM, self.data_fp16_one, one_fp16_scalar, 1, 1, 8)

    def _init_for_valid_mask(self):
        """
        note:
            for update_valid_mask, valid_mask uses int16, which is for using vand,
            vand supports fp162int16 (use round ...)

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        # ceiling vector_mask_max for handling tailing
        self.valid_mask_size_int8 = _ceiling(self.N, self.vector_mask_max)
        self.valid_mask_int8_ub = tik_instance.Tensor('int8', (self.valid_mask_size_int8,), tik.scope_ubuf,
                                                      'valid_mask_int8_ub')
        self.valid_mask_fp16_ub = self.tmp_tensor_ub_fp16

        scalar_i = tik_instance.Scalar('float16', init_value=1)
        self._tailing_handle_vector_dup(self.valid_mask_fp16_ub,
                                        scalar_i, self.valid_mask_size_int8, Constant.FP16_SIZE)
        self._tailing_handle_vec_conv(self.valid_mask_int8_ub, self.valid_mask_fp16_ub, self.valid_mask_size_int8,
                                      Constant.INT8_SIZE, Constant.FP16_SIZE, 'round')

        # update valid mask, here float16 fixed, ensure 32b aligned. note: size `below = valid_mask_size_int8`
        self.tmp_valid_mask_float16 = self.tik_instance.Tensor('float16', (self.valid_mask_size_int8,), tik.scope_ubuf,
                                                               'tmp_valid_mask_float16')
        self.tmp_mask_float16 = self.tmp_tensor_ub_fp16

    def _init_for_vcmax(self):
        """
        init for vcmax, which is used in _update_next_nonzero_idx()

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        tik_instance = self.tik_instance
        # dscend sorted list in ub, fixed dtype is fp16. use selected_idx_ub to generate dsorts_ub
        dsorts_size = _ceiling(self.N, self.vector_mask_max)
        self.dsorts_ub = tik_instance.Tensor('float16', (dsorts_size,), tik.scope_ubuf, 'dsorts_ub')
        scalar_dsorts_size = tik_instance.Scalar('float16', init_value=dsorts_size)
        self._tailing_handle_vector_dup(self.dsorts_ub, scalar_dsorts_size, dsorts_size, Constant.FP16_SIZE)
        selected_idx_ub_fp16 = self.tmp_tensor_ub_fp16
        self._tailing_handle_vector_dup(selected_idx_ub_fp16, scalar_dsorts_size, dsorts_size, Constant.FP16_SIZE)
        self._tailing_handle_vec_conv(selected_idx_ub_fp16,
                                      self.selected_idx_ub, self.ceil_n, Constant.FP16_SIZE, Constant.INT32_SIZE,
                                      '', 1.)
        self._tailing_handle_vsub(self.dsorts_ub, self.dsorts_ub, selected_idx_ub_fp16, dsorts_size, Constant.FP16_SIZE,
                                  Constant.FP16_SIZE, Constant.FP16_SIZE)

        self.vcmax_ub = tik_instance.Tensor('float16', (Constant.MASK_VCMAX_FP16,), tik.scope_ubuf, 'vcmax_ub')
        self.middle_max_val = tik_instance.Tensor('float16',
                                                  (Constant.MASK_VCMAX_FP16,), tik.scope_ubuf, 'middle_max_val')
        self.dst_max_val_ub = tik_instance.Tensor('float16', (Constant.SHAPE_NEXT_NONZERO,), tik.scope_ubuf,
                                                  'dst_max_val_ub')

        # idx_fp16_ub stores next nonzero idx, dtype needs conv to int8
        self.idx_fp16_ub = tik_instance.Tensor('float16',
                                               (Constant.SHAPE_NEXT_NONZERO,), tik.scope_ubuf, 'idx_fp16_ub')

        # practically ceil_n is less than Constant.MASK_VCMAX_FP16 * REPEAT_TIMES_MAX
        self.repeat_vmul_vcmax = self.ceil_n % (Constant.MASK_VCMAX_FP16 *
                                                Constant.REPEAT_TIMES_MAX) // Constant.MASK_VCMAX_FP16
        self.last_num_vmul_vcmax = self.ceil_n % Constant.MASK_VCMAX_FP16
        self.vcmax_mask = self.repeat_vmul_vcmax + (1 if self.last_num_vmul_vcmax > 0 else 0)

    def _input_trans(self):
        """
        1980's inputs trans to new soc's
        Note: should use vreduce, not vgather
        all_inp_proposals_ub_1980:
            1980:
                shape is (N, 8), only one addr_base
                [
                [x1, y1, x2, y2, score, /, /, /]
                [x1, y1, x2, y2, score, /, /, /]
                [x1, y1, x2, y2, score, /, /, /]
                [x1, y1, x2, y2, score, /, /, /]
                ...
                ]

            new:
                5 addr_bases
                x1[] with N elems
                x2[]
                y1[]
                y2[]
                score[]

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # 2 ** 0 + 2 ** 8 + 2 ** 16 + 2 ** 24
        pattern_value_fp32_x1 = 16843009
        # 2 ** 1 + 2 ** 9 + 2 ** 17 + 2 ** 25
        pattern_value_fp32_y1 = 33686018
        # 2 ** 2 + 2 ** 10 + 2 ** 18 + 2 ** 26
        pattern_value_fp32_x2 = 67372036
        # 2 ** 3 + 2 ** 11 + 2 ** 19 + 2 ** 27
        pattern_value_fp32_y2 = 134744072
        if self.input_dtype == 'float16':
            # Constant.BURST_PROPOSAL_NUM is shape0 of tmp_tensor_ub_fp16_burst
            repeat = _ceil_div(self.N * Constant.ELEMENT_NUM, Constant.BURST_PROPOSAL_NUM)
            with self.tik_instance.for_range(0, repeat) as i:
                offset = i * Constant.BURST_PROPOSAL_NUM
                self.tik_instance.data_move(self.tmp_tensor_ub_fp16_burst,
                                            self.all_inp_proposals_gm_1980[offset], 0, 1,
                                            Constant.BURST_PROPOSAL_NUM *
                                            self.input_bytes_each_elem // Constant.CONFIG_DATA_ALIGN,
                                            src_stride=0, dst_stride=0)
                self._tailing_handle_vec_conv(self.all_inp_proposals_ub_1980_fp32[offset],
                                              self.tmp_tensor_ub_fp16_burst,
                                              Constant.BURST_PROPOSAL_NUM, Constant.FP32_SIZE, Constant.FP16_SIZE)
        else:
            # `info: max. burst is 65535, so max. bytes is 65535*32b, support max. N is 65535*32/2/8=131070 for fp16
            self.tik_instance.data_move(self.all_inp_proposals_ub_1980_fp32, self.all_inp_proposals_gm_1980, 0, 1,
                                        self.ceil_n * Constant.ELEMENT_NUM *
                                        self.input_bytes_each_elem // Constant.CONFIG_DATA_ALIGN,
                                        src_stride=0, dst_stride=0)

        # fp32. uint32 covers 32 elems, so shape[0] is 256/32=8
        pattern_x1 = self.tik_instance.Tensor('uint32', (8,), tik.scope_ubuf, name='pattern_x1_ub')
        pattern_y1 = self.tik_instance.Tensor('uint32', (8,), tik.scope_ubuf, name='pattern_y1_ub')
        pattern_x2 = self.tik_instance.Tensor('uint32', (8,), tik.scope_ubuf, name='pattern_x2_ub')
        pattern_y2 = self.tik_instance.Tensor('uint32', (8,), tik.scope_ubuf, name='pattern_y2_ub')

        self.tik_instance.vector_dup(8, pattern_x1,
                                     self.tik_instance.Scalar('uint32', init_value=pattern_value_fp32_x1),
                                     1, 1, 1)
        self.tik_instance.vector_dup(8, pattern_y1,
                                     self.tik_instance.Scalar('uint32', init_value=pattern_value_fp32_y1),
                                     1, 1, 1)
        self.tik_instance.vector_dup(8, pattern_x2,
                                     self.tik_instance.Scalar('uint32', init_value=pattern_value_fp32_x2),
                                     1, 1, 1)
        self.tik_instance.vector_dup(8, pattern_y2,
                                     self.tik_instance.Scalar('uint32', init_value=pattern_value_fp32_y2),
                                     1, 1, 1)

        self._tailing_handle_vreduce_input(self.x1_ub, self.all_inp_proposals_ub_1980_fp32, pattern_x1)
        self._tailing_handle_vreduce_input(self.y1_ub, self.all_inp_proposals_ub_1980_fp32, pattern_y1)
        self._tailing_handle_vreduce_input(self.x2_ub, self.all_inp_proposals_ub_1980_fp32, pattern_x2)
        self._tailing_handle_vreduce_input(self.y2_ub, self.all_inp_proposals_ub_1980_fp32, pattern_y2)

    def _tailing_handle_vreduce_input(self, dst_ub, src0_ub, src1_pattern_ub):
        """
        tailing handle: means handle all inputs, especially the tail is special and need to deal with
        3 steps to handle tailing.
        for nms: step2 and step3 is enough, if all uses UB at the same time

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0 in ub
        src1_pattern_ub: pattern for src1

        Returns
        -------
        None
        """
        # 16 for fp16, 8 for fp32
        vector_proposals_max = self.vector_mask_max // 8
        offset = 0

        # step1: max repeat
        # only this tailing need the step1, other tailings don't need it, as ceil_n may > vector_proposals_max * 255
        loop_num = self.ceil_n // (vector_proposals_max * Constant.REPEAT_TIMES_MAX)
        for i in range(0, loop_num):
            self.tik_instance.vreduce(mask=self.vector_mask_max,
                                      dst=dst_ub[offset],
                                      src0=src0_ub[offset * 8],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=Constant.REPEAT_TIMES_MAX,
                                      src0_blk_stride=1,
                                      src0_rep_stride=self.vector_mask_max *
                                                      self.bytes_each_elem // Constant.CONFIG_DATA_ALIGN,
                                      # here 0 means: pattern is reused in each repeat
                                      src1_rep_stride=0)
            offset = (i + 1) * vector_proposals_max * Constant.REPEAT_TIMES_MAX

        # step2: repeat num
        repeat = self.ceil_n % (vector_proposals_max * Constant.REPEAT_TIMES_MAX) // vector_proposals_max
        if repeat > 0:
            self.tik_instance.vreduce(mask=self.vector_mask_max,
                                      dst=dst_ub[offset],
                                      src0=src0_ub[offset * 8],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=repeat,
                                      src0_blk_stride=1,
                                      src0_rep_stride=self.vector_mask_max * self.bytes_each_elem //
                                                      Constant.CONFIG_DATA_ALIGN,
                                      # here 0 means: pattern is reused in each repeat
                                      src1_rep_stride=0)

        # step3: last num
        last_num = self.ceil_n % vector_proposals_max
        if last_num > 0:
            offset += repeat * vector_proposals_max
            self.tik_instance.vreduce(mask=8 * last_num,
                                      dst=dst_ub[offset],
                                      src0=src0_ub[offset * 8],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=1,
                                      src0_blk_stride=1,
                                      # no need to repeat, so 0
                                      src0_rep_stride=0,
                                      # here 0 means: pattern is reused in each repeat
                                      src1_rep_stride=0)

    def _tailing_handle_vreduce_output(self, dst_ub, src0_ub, src1_pattern_ub):
        """
        [N, 8] => [N, 5]

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0 in ub
        src1_pattern_ub: pattern for src1

        Returns
        -------
        None
        """
        # `info: =16 for fp16, =8 for fp32. here 8 is ncols
        vector_proposals_max = self.input_vector_mask_max // 8
        offset = 0

        # step2: repeat num
        repeat = self.ceil_n % (vector_proposals_max * Constant.REPEAT_TIMES_MAX) // vector_proposals_max
        if repeat > 0:
            self.tik_instance.vreduce(mask=self.input_vector_mask_max,
                                      dst=dst_ub[offset * 5],
                                      src0=src0_ub[offset * 8],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=repeat,
                                      src0_blk_stride=1,
                                      src0_rep_stride=self.vector_mask_max * self.input_bytes_each_elem \
                                                      // Constant.CONFIG_DATA_ALIGN,
                                      src1_rep_stride=0)

        # step3: last num
        last_num = self.ceil_n % vector_proposals_max
        if last_num > 0:
            offset += repeat * vector_proposals_max
            self.tik_instance.vreduce(mask=8 * last_num,
                                      dst=dst_ub[offset * 5],
                                      src0=src0_ub[offset * 8],
                                      src1_pattern=src1_pattern_ub,
                                      repeat_times=1,
                                      src0_blk_stride=1, src0_rep_stride=0,
                                      src1_rep_stride=0)

    def _tailing_handle_vmuls(self, dst_ub, src_ub, scalar, size):
        """
        handle tailing of vmuls

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src ub
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat num
        repeat = size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vmuls(mask=self.vector_mask_max,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=repeat,
                                    dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=8, src_rep_stride=8)

        # step3: last num
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vmuls(mask=last_num,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=1,
                                    dst_blk_stride=1, src_blk_stride=1, dst_rep_stride=8, src_rep_stride=8)

    def _tailing_handle_vsub(self, dst_ub, src0_ub, src1_ub, size, dst_bytes, src0_bytes, src1_bytes):
        """
        handle tailing of vsub

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0
        src1_ub: src1
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat num
        repeat = size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vsub(mask=self.vector_mask_max,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=repeat,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=self.vector_mask_max * dst_bytes // Constant.CONFIG_DATA_ALIGN,
                                   src0_rep_stride=self.vector_mask_max * src0_bytes // Constant.CONFIG_DATA_ALIGN,
                                   src1_rep_stride=self.vector_mask_max * src1_bytes // Constant.CONFIG_DATA_ALIGN)

        # step3: last num
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vsub(mask=last_num,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=1,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8)

    def _tailing_handle_vmul(self, dst_ub, src0_ub, src1_ub, size, mask_max=None,
                             dst_bytes=None, src0_bytes=None, src1_bytes=None):
        """
        handle tailing of vmul

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src0_ub: src0
        src1_ub: src1
        size: totol size of elems
        mask_max: max. mask
        dst_bytes: dst bytes
        src0_bytes: src0 bytes
        src1_bytes: src1 bytes

        Returns
        -------
        None
        """
        if mask_max is None:
            mask_max = self.vector_mask_max

        offset = 0

        # step2: repeat num
        repeat = size % (mask_max * Constant.REPEAT_TIMES_MAX) // mask_max
        if repeat > 0:
            self.tik_instance.vmul(mask=mask_max,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=repeat,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=mask_max * dst_bytes // Constant.CONFIG_DATA_ALIGN,
                                   src0_rep_stride=mask_max * src0_bytes // Constant.CONFIG_DATA_ALIGN,
                                   src1_rep_stride=mask_max * src1_bytes // Constant.CONFIG_DATA_ALIGN)

        # step3: last num
        last_num = size % mask_max
        if last_num > 0:
            offset += repeat * mask_max
            self.tik_instance.vmul(mask=last_num,
                                   dst=dst_ub[offset],
                                   src0=src0_ub[offset],
                                   src1=src1_ub[offset],
                                   repeat_times=1,
                                   dst_blk_stride=1, src0_blk_stride=1, src1_blk_stride=1,
                                   dst_rep_stride=8, src0_rep_stride=8, src1_rep_stride=8)

    def _tailing_handle_vadds(self, dst_ub, src_ub, scalar, size):
        """
        handle tailing of vadds

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat num
        repeat = size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vadds(mask=self.vector_mask_max,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=repeat,
                                    dst_blk_stride=1, src_blk_stride=1,
                                    dst_rep_stride=8, src_rep_stride=8)

        # step3: last num
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vadds(mask=last_num,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=1,
                                    dst_blk_stride=1, src_blk_stride=1,
                                    dst_rep_stride=8, src_rep_stride=8)

    def _tailing_handle_vmaxs(self, dst_ub, src_ub, scalar, size):
        """
        handle tailing of vmaxs

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat num
        repeat = size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vmaxs(mask=self.vector_mask_max,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=repeat,
                                    dst_blk_stride=1, src_blk_stride=1,
                                    dst_rep_stride=8, src_rep_stride=8)

        # step3: last num
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vmaxs(mask=last_num,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=1,
                                    dst_blk_stride=1, src_blk_stride=1,
                                    dst_rep_stride=8, src_rep_stride=8)

    def _tailing_handle_vmins(self, dst_ub, src_ub, scalar, size):
        """
        handle tailing of vmins

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src
        scalar: scalar
        size: totol size of elems

        Returns
        -------
        None
        """
        offset = 0

        # step2: repeat num
        repeat = size % (self.vector_mask_max * Constant.REPEAT_TIMES_MAX) // self.vector_mask_max
        if repeat > 0:
            self.tik_instance.vmins(mask=self.vector_mask_max,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=repeat,
                                    dst_blk_stride=1, src_blk_stride=1,
                                    dst_rep_stride=8, src_rep_stride=8)

        # step3: last num
        last_num = size % self.vector_mask_max
        if last_num > 0:
            offset += repeat * self.vector_mask_max
            self.tik_instance.vmins(mask=last_num,
                                    dst=dst_ub[offset],
                                    src=src_ub[offset],
                                    scalar=scalar,
                                    repeat_times=1,
                                    dst_blk_stride=1, src_blk_stride=1,
                                    dst_rep_stride=8, src_rep_stride=8)

    def _tailing_handle_vec_conv(self, dst_ub, src_ub, size, dst_bytes, src_bytes, mode="none", deqscale=None):
        """
        handle tailing of vec_conv

        Parameters
        ----------
        dst_ub: dst tensor in ub
        src_ub: src ub
        size: totol size of elems
        dst_bytes: bytes of each elem of dst
        src_bytes: bytes of each elem of src

        Returns
        -------
        None
        """
        # max. is vector_mask_max. src_bytes can be 1
        mask_max = min(Constant.BYTES_ONE_CYCLE_VECTOR // src_bytes, self.vector_mask_max)
        if dst_bytes == Constant.INT32_SIZE:
            mask_max = Constant.BURST_PROPOSAL_NUM // 2

        offset = 0

        # step2: repeat num
        repeat = size % (mask_max * Constant.REPEAT_TIMES_MAX) // mask_max
        if repeat > 0:
            self.tik_instance.vec_conv(mask_max,
                                       mode,
                                       dst=dst_ub[offset],
                                       src=src_ub[offset],
                                       repeat_times=repeat,
                                       dst_rep_stride=mask_max * dst_bytes // Constant.CONFIG_DATA_ALIGN,
                                       src_rep_stride=mask_max * src_bytes // Constant.CONFIG_DATA_ALIGN,
                                       deqscale=deqscale)

        # step3: last num
        last_num = size % mask_max
        if last_num > 0:
            offset += repeat * mask_max
            self.tik_instance.vec_conv(last_num,
                                       mode,
                                       dst=dst_ub[offset],
                                       src=src_ub[offset],
                                       repeat_times=1,
                                       dst_rep_stride=0, src_rep_stride=0,
                                       deqscale=deqscale)

    def _tailing_handle_vector_dup(self, dst_ub, scalar, size, src_bytes):
        """
        handle tailing of vector dup

        Parameters
        ----------
        dst_ub: dst tensor in ub
        scalar: scalar used to dup
        size: totol size of elems
        src_bytes: bytes of each elem of src

        Returns
        -------
        None
        """
        # max. is vector_mask_max. src_bytes can be 1
        mask_max = min(Constant.BYTES_ONE_CYCLE_VECTOR // src_bytes, self.vector_mask_max)
        offset = 0

        # step2: repeat num
        repeat = size % (mask_max * Constant.REPEAT_TIMES_MAX) // mask_max
        if repeat > 0:
            self.tik_instance.vector_dup(mask=mask_max,
                                         dst=dst_ub[offset],
                                         scalar=scalar,
                                         repeat_times=repeat,
                                         dst_blk_stride=1,
                                         dst_rep_stride=mask_max * src_bytes // Constant.CONFIG_DATA_ALIGN)

        # step3: last num
        last_num = size % mask_max
        if last_num > 0:
            offset += repeat * mask_max
            self.tik_instance.vector_dup(mask=last_num,
                                         dst=dst_ub[offset],
                                         scalar=scalar,
                                         repeat_times=1,
                                         dst_blk_stride=0, dst_rep_stride=0)

    def selected_boxes_gen(self):
        """
        selected_boxes generate from proposals_ub_1980

        original box_scores: [N, 8]
        selected_boxes:      [N, 5]

        Parameters
        ----------
        None

        Returns
        -------
        selected_boxes_ub:
        """
        # 7967 is [1 1 1 1 1 0 0 0 1 1 1 1 1 0 0 0] for 16 inputs
        pattern_value_7967 = 7967
        # [1 1 1 1 1 0 0 0   1 1 1 1 1 0 0 0   1 1 1 1 1 0 0 0
        # 1 1 1 1 1 0 0 0] one uint32 can handle selection of 32 elems
        pattern_value_522133279 = 522133279
        # def selected_boxes_ub
        selected_boxes_ub = self.tik_instance.Tensor(self.input_dtype,
                                                     (self.ceil_n, Constant.VALID_COLUMN_NUM), tik.scope_ubuf,
                                                     'selected_boxes_ub')

        # create pattern, shape is 16 or 8, which is enough and it'll be reused in vreduce, and vreduce output
        if self.input_dtype == 'float16':
            pattern = self.tik_instance.Tensor('uint16', (16,), tik.scope_ubuf, 'pattern_ub')
            # init pattern
            self.tik_instance.vector_dup(16, pattern,
                                         self.tik_instance.Scalar('uint16', 'pattern_s', init_value=pattern_value_7967),
                                         1, 1, 1)
            repeat = _ceil_div(self.N * Constant.ELEMENT_NUM, Constant.BURST_PROPOSAL_NUM)
            with self.tik_instance.for_range(0, repeat) as i:
                offset = i * Constant.BURST_PROPOSAL_NUM
                self.tik_instance.data_move(self.tmp_tensor_ub_fp16_burst,
                                            self.all_inp_proposals_gm_1980[offset], 0, 1,
                                            Constant.BURST_PROPOSAL_NUM *
                                            self.input_bytes_each_elem // Constant.CONFIG_DATA_ALIGN, 0, 0)
                self.tik_instance.vreduce(mask=Constant.BURST_PROPOSAL_NUM,
                                          dst=selected_boxes_ub[offset *
                                                                Constant.VALID_COLUMN_NUM // Constant.ELEMENT_NUM],
                                          src0=self.tmp_tensor_ub_fp16_burst,
                                          src1_pattern=pattern,
                                          repeat_times=1,
                                          src0_blk_stride=1,
                                          src0_rep_stride=0,
                                          src1_rep_stride=0)
        else:
            pattern = self.tik_instance.Tensor('uint32', (8,), tik.scope_ubuf,
                                               'pattern_ub')
            self.tik_instance.vector_dup(8, pattern,
                                         self.tik_instance.Scalar('uint32', 'pattern_s',
                                                                  init_value=pattern_value_522133279), 1, 1, 1)
            self._tailing_handle_vreduce_output(selected_boxes_ub, self.all_inp_proposals_ub_1980_fp32, pattern)

        return selected_boxes_ub

    def _selected_idx_gen(self):
        """
        selected_idx generate

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # int32 is fixed for output index
        selected_idx_ub = self.tik_instance.Tensor('int32', (self.ceil_n,), tik.scope_ubuf, 'selected_idx_ub')
        with self.tik_instance.for_range(0, self.ceil_n) as i:
            selected_idx_ub[i].set_as(i)

        return selected_idx_ub

    def _scaling(self):
        """
        scaling of input, scaling factor is Constant.DOWN_FACTOR

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._tailing_handle_vmuls(self.x1_ub, self.x1_ub, Constant.DOWN_FACTOR, self.ceil_n)
        self._tailing_handle_vmuls(self.x2_ub, self.x2_ub, Constant.DOWN_FACTOR, self.ceil_n)
        self._tailing_handle_vmuls(self.y1_ub, self.y1_ub, Constant.DOWN_FACTOR, self.ceil_n)
        self._tailing_handle_vmuls(self.y2_ub, self.y2_ub, Constant.DOWN_FACTOR, self.ceil_n)

    def _area(self):
        """
        area = (x2-x1) * (y2-y1), this is vector computing
        area can be reused in loops

        Parameters
        ----------
        None


        Returns
        -------
        None
        """
        if self.total_areas_ub is not None:
            return self.total_areas_ub

        tik_instance = self.tik_instance
        self.total_areas_ub = tik_instance.Tensor(self.data_type, (self.ceil_n,), name="total_areas_ub",
                                                  scope=tik.scope_ubuf)

        # reuse tmp tensor xx2 for y2suby1
        y2suby1 = self.xx2

        self._tailing_handle_vsub(self.total_areas_ub, self.x2_ub, self.x1_ub, self.ceil_n, self.bytes_each_elem,
                                  self.bytes_each_elem, self.bytes_each_elem)
        self._tailing_handle_vsub(y2suby1, self.y2_ub, self.y1_ub, self.ceil_n, self.bytes_each_elem,
                                  self.bytes_each_elem, self.bytes_each_elem)
        self._tailing_handle_vmul(self.total_areas_ub, self.total_areas_ub, y2suby1, self.ceil_n, None,
                                  self.bytes_each_elem, self.bytes_each_elem, self.bytes_each_elem)

        return self.total_areas_ub

    def _intersection(self, cur):
        """
        intersection calculation

        Parameters
        ----------
        cur: intersection of cur proposal and the others

        Returns
        -------
        None
        """
        self.x1i.set_as(self.x1_ub[cur])
        self.y1i.set_as(self.y1_ub[cur])
        self.x2i.set_as(self.x2_ub[cur])
        self.y2i.set_as(self.y2_ub[cur])

        # `xx1 = max(x1[i], x1[1:]),  yy1 = max(y1[i], y1[1:]), xx2=min(x2[i], x2[1:]),  yy2=min(y2[i], y2[1:])`
        self._tailing_handle_vmaxs(self.xx1, self.x1_ub, self.x1i, self.ceil_n)
        self._tailing_handle_vmins(self.xx2, self.x2_ub, self.x2i, self.ceil_n)

        # `w = max(0, xx2-xx1+offset), h = max(0, yy2-yy1+offset), offset=0 here`
        self._tailing_handle_vsub(self.xx1, self.xx2, self.xx1, self.ceil_n, self.bytes_each_elem,
                                  self.bytes_each_elem, self.bytes_each_elem)
        # w stores in xx1
        self._tailing_handle_vmaxs(self.xx1, self.xx1, self.zero_datatype_scalar, self.ceil_n)

        # reuse tmp tensor
        # 'pylint: disable=attribute-defined-outside-init
        self.yy2 = self.xx2
        self._tailing_handle_vmaxs(self.yy1, self.y1_ub, self.y1i, self.ceil_n)
        self._tailing_handle_vmins(self.yy2, self.y2_ub, self.y2i, self.ceil_n)
        self._tailing_handle_vsub(self.yy1, self.yy2, self.yy1, self.ceil_n, self.bytes_each_elem,
                                  self.bytes_each_elem, self.bytes_each_elem)
        # h stores in yy1
        self._tailing_handle_vmaxs(self.yy1, self.yy1, self.zero_datatype_scalar, self.ceil_n)
        # inter stores in xx1
        self._tailing_handle_vmul(self.xx1, self.xx1, self.yy1, self.ceil_n, None,
                                  self.bytes_each_elem, self.bytes_each_elem, self.bytes_each_elem)

        return self.xx1

    def _cmpmask2bitmask(self, dst_ub, cmpmask, handle_dst_size):
        """
        in one repeat, handle max. 128 elems. so tensor defed below has 128 shape
        bitmask is like [1 0 1 1 0 0 0 1]

        Parameters
        ----------
        cur: compute cur proposal and the others

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        tik_instance.vsel(Constant.BURST_PROPOSAL_NUM, 0,
                          self.output_mask_f16, cmpmask, self.data_fp16_one, self.data_fp16_zero,
                          1, 1, 1, 1, 8, 8, 8)

        tik_instance.vec_conv(handle_dst_size, "none", dst_ub, self.output_mask_f16, 1, 8, 8)

    def _update_next_nonzero_idx(self, valid_mask_int8_ub):
        """
        update next nonzero idx
        note that using fp16 for dsorts_ub,valid_mask_fp16_ub,vcmax_ub... may cause precision problem if N > 2048

        Parameters
        ----------
        valid_mask: is [0 0 1 1 ]

        Returns
        -------
        tensor[0] contains the next nonzero idx
        """
        # int8 conv to fp16
        self._tailing_handle_vec_conv(self.valid_mask_fp16_ub, valid_mask_int8_ub, size=self.valid_mask_size_int8,
                                      dst_bytes=2, src_bytes=1)

        # already compute repeat and last_num in _init_for_vcmax()
        repeat = self.repeat_vmul_vcmax
        last_num = self.last_num_vmul_vcmax

        # vmul
        if repeat > 0:
            self.tik_instance.vmul(Constant.MASK_VCMAX_FP16,
                                   self.valid_mask_fp16_ub, self.valid_mask_fp16_ub, self.dsorts_ub,
                                   repeat, 1, 1, 1, 8, 8, 8)

        if last_num > 0:
            vmul_offset = repeat * Constant.MASK_VCMAX_FP16
            self.tik_instance.vmul(last_num, self.valid_mask_fp16_ub[vmul_offset], self.valid_mask_fp16_ub[vmul_offset],
                                   self.dsorts_ub[vmul_offset], 1, 1, 1, 1, 8, 8, 8)

        # vcmax
        if repeat > 0:
            self.tik_instance.vcmax(Constant.MASK_VCMAX_FP16, self.vcmax_ub, self.valid_mask_fp16_ub, repeat, 1, 1, 8)

        if last_num > 0:
            offset = repeat * Constant.MASK_VCMAX_FP16
            self.tik_instance.vcmax(last_num, self.vcmax_ub[repeat * 2], self.valid_mask_fp16_ub[offset], 1, 1, 1, 8)

        self.tik_instance.pipe_barrier("PIPE_V")
        # pattern here means 101010..., vreduce once is enough
        self.tik_instance.vreduce(Constant.MASK_VCMAX_FP16,
                                  self.middle_max_val, self.vcmax_ub, src1_pattern=1, repeat_times=1,
                                  src0_blk_stride=1, src0_rep_stride=0, src1_rep_stride=0)

        # below: dst_max_val_ub[0], idx_fp16_ub[0], next_nonzero_int32_idx[0] stores meaningful val
        self.tik_instance.vcmax(self.vcmax_mask, self.dst_max_val_ub, self.middle_max_val, 1, 0, 1, 0)

        # dst idx, note: idx maybe valid_mask_size
        self.tik_instance.vsub(Constant.SHAPE_NEXT_NONZERO,
                               self.idx_fp16_ub, self.dsorts_ub, self.dst_max_val_ub, 1, 1, 1,
                               1, 8, 8, 8)

        # conv to int32
        self._tailing_handle_vec_conv(self.next_nonzero_int32_idx,
                                      self.idx_fp16_ub, Constant.SHAPE_NEXT_NONZERO, Constant.INT32_SIZE,
                                      Constant.FP16_SIZE, mode='round')

    def _one_loop(self, cur):
        """
        in one loop: iou, generate bitmask and return output_mask_int8_ub

        logic of nms, new way:
            output mask = 1 if intersection < (area_i + area_j) * iou_thres / (iou_thres + 1)

        Parameters
        ----------
        cur: compute cur proposal and the others

        Returns
        -------
        output_mask_int8_ub
        """
        inter = self._intersection(cur)

        areas = self._area()
        self.area_cur.set_as(areas[cur])
        aadds = self.xx2
        self._tailing_handle_vadds(aadds, areas, self.area_cur, self.ceil_n)
        self._tailing_handle_vmuls(aadds, aadds, self.iou_thres_factor, self.ceil_n)

        # cmpmask 2 bitmask
        output_mask_int8_ub = self._tailing_handle_cmp_le_and_2bitmask(inter, aadds, self.ceil_n)

        # set output_mask[cur] = 0, because will be added into DST, and deleted from SRC proposal list
        output_mask_int8_ub[cur].set_as(self.zero_int8_scalar)
        return output_mask_int8_ub

    def _tailing_handle_cmp_le_and_2bitmask(self, src0_ub, src1_ub, size):
        """
        combine vcmp_le() and cmpmask2bitmask()
        vcmp handle max. 128 mask, repeat = 1

        size: total size of proposals

        Parameters
        ----------
        src0_ub: src0 in ub
        src1_ub: src1 in ub
        cur: compute cur proposal and the others

        Returns
        -------
        output_mask_int8_ub
        """
        loops = size // (self.vector_mask_max * Constant.INT8_SIZE)
        offset = 0

        # step1: max. mask * max. repeat  * loops times
        if loops > 0:
            for loop_index in range(0, loops):
                # vcmp only run once, so repeat = 1
                cmpmask = self.tik_instance.vcmp_le(mask=self.vector_mask_max,
                                                    src0=src0_ub[offset],
                                                    src1=src1_ub[offset],
                                                    # 1 is fixed
                                                    src0_stride=1, src1_stride=1)
                self._cmpmask2bitmask(dst_ub=self.output_mask_int8_ub[offset],
                                      cmpmask=cmpmask, handle_dst_size=self.vector_mask_max)

                offset = (loop_index + 1) * self.vector_mask_max * Constant.INT8_SIZE

        # step2: not used
        # step3: last num
        last_num = size % self.vector_mask_max
        if last_num > 0:
            cmpmask = self.tik_instance.vcmp_le(mask=last_num,
                                                src0=src0_ub[offset],
                                                src1=src1_ub[offset],
                                                src0_stride=1, src1_stride=1)
            self._cmpmask2bitmask(dst_ub=self.output_mask_int8_ub[offset],
                                  cmpmask=cmpmask, handle_dst_size=last_num)

        return self.output_mask_int8_ub

    def _update_valid_mask(self, mask_ub_int8_ub):
        """
        update valid mask
        note: use vand instead of vmul, but vand only compute uint16/int16,
            so use int16 for out_mask, support f162s16 using round mode in cmpmask2bitmask()

        Parameters
        ----------
        mask_ub_int8_ub: which will be used to update valid_mask_ub

        Returns
        -------
        None
        """
        self._tailing_handle_vec_conv(self.tmp_valid_mask_float16, self.valid_mask_int8_ub, self.valid_mask_size_int8,
                                      dst_bytes=2, src_bytes=1)
        self._tailing_handle_vec_conv(self.tmp_mask_float16, mask_ub_int8_ub, self.valid_mask_size_int8, dst_bytes=2,
                                      src_bytes=1)

        # [0 0 1 1] * [1 0 1 0] = [0 0 1 0]
        self._tailing_handle_vmul(self.tmp_valid_mask_float16, self.tmp_valid_mask_float16, self.tmp_mask_float16,
                                  self.valid_mask_size_int8,
                                  self.vector_mask_max, Constant.FP16_SIZE, Constant.FP16_SIZE, Constant.FP16_SIZE)

        # float16 to int8
        self._tailing_handle_vec_conv(self.valid_mask_int8_ub, self.tmp_valid_mask_float16, self.valid_mask_size_int8,
                                      dst_bytes=1, src_bytes=2)

    def loops(self):
        """
        run loops

        Parameters
        ----------
        None

        Returns
        -------
        selected_mask_ub
        """
        # def and init selected_mask_ub
        selected_mask_ub = self.tik_instance.Tensor('uint8', (self.ceil_n,), name="selected_mask_ub",
                                                    scope=tik.scope_ubuf)
        selected_mask_ub_tmp = self.tmp_tensor_ub_fp16
        scalar_i = self.tik_instance.Scalar('float16', init_value=0)
        self._tailing_handle_vector_dup(selected_mask_ub_tmp, scalar_i, size=self.ceil_n, src_bytes=Constant.FP16_SIZE)
        self._tailing_handle_vec_conv(selected_mask_ub,
                                      selected_mask_ub_tmp, self.ceil_n, Constant.UINT8_SIZE, Constant.FP16_SIZE,
                                      'round')

        cur = self.tik_instance.Scalar(dtype='int32', name='cur_scalar', init_value=0)
        with self.tik_instance.for_range(0, self.N):
            with self.tik_instance.if_scope(cur < self.N):
                # set 1, means valid
                selected_mask_ub[cur] = self.one_uint8_scalar
                mask_ub = self._one_loop(cur)
                self._update_valid_mask(mask_ub)
                self._update_next_nonzero_idx(self.valid_mask_int8_ub)

                cur.set_as(self.next_nonzero_int32_idx[0])

        return selected_mask_ub


# 'pylint: disable=too-many-locals,too-many-arguments
def _tik_func_nms_multi_core_basic_api(input_shape, input_dtype, thresh, total_output_proposal_num, kernel_name_var):
    """
    Compute output boxes after non-maximum suppression

    Parameters
    ----------
    input_shape: dict
        shape of input boxes, including proposal boxes and corresponding confidence scores

    input_dtype: str
        input data type: options are float16 and float32

    thresh: float
        iou threshold

    total_output_proposal_num: int
        the number of output proposal boxes

    kernel_name: str
        cce kernel name

    Returns
    -------
    tik_instance: TIK API
    """
    tik_instance = tik.Tik()
    total_input_proposal_num, _ = input_shape
    proposals = tik_instance.Tensor(input_dtype, (total_input_proposal_num, Constant.ELEMENT_NUM),
                                    name="in_proposals",
                                    scope=tik.scope_gm)

    nms_helper = _NMSHelper(tik_instance, proposals,
                            (total_input_proposal_num, Constant.ELEMENT_NUM), input_dtype, thresh)
    output_proposals_ub = nms_helper.selected_boxes_gen()
    output_index_ub = nms_helper.selected_idx_ub
    output_mask_ub = nms_helper.loops()

    # data move from ub to gm. def tensor in gm can be real shape, dont need to ceiling
    out_proposals_gm = tik_instance.Tensor(input_dtype, (total_output_proposal_num, Constant.VALID_COLUMN_NUM),
                                           name="out_proposals_gm", scope=tik.scope_gm)
    # address is 32B aligned
    out_index_gm = tik_instance.Tensor("int32", (total_output_proposal_num,), name="out_index_gm", scope=tik.scope_gm)
    out_mask_gm = tik_instance.Tensor("uint8", (total_output_proposal_num,), name="out_mask_gm", scope=tik.scope_gm)

    tik_instance.data_move(out_proposals_gm, output_proposals_ub, 0, nburst=1,
                           # `max. burst is 65535, unit is 32B, so support: 65535*32/2/8=131070 proposals if fp16.`
                           burst=(nms_helper.ceil_n * Constant.VALID_COLUMN_NUM * \
                                  nms_helper.bytes_each_elem // Constant.CONFIG_DATA_ALIGN),
                           src_stride=0, dst_stride=0)
    tik_instance.data_move(out_index_gm, output_index_ub, 0, nburst=1,
                           burst=(nms_helper.ceil_n * Constant.INT32_SIZE // Constant.CONFIG_DATA_ALIGN),
                           src_stride=0, dst_stride=0)
    tik_instance.data_move(out_mask_gm, output_mask_ub, 0, nburst=1,
                           # here need _ceiling() as ceilN can be 16; 16*1//32=0 is wrong
                           burst=_ceiling(nms_helper.ceil_n *
                                          Constant.UINT8_SIZE,
                                          Constant.CONFIG_DATA_ALIGN) // Constant.CONFIG_DATA_ALIGN,
                           src_stride=0, dst_stride=0)
    tik_instance.BuildCCE(kernel_name=kernel_name_var,
                          inputs=[proposals],
                          outputs=[out_proposals_gm, out_index_gm, out_mask_gm],
                          output_files_path=None,
                          enable_l2=False)
    return tik_instance


# 'pylint: disable=unused-argument,too-many-locals,too-many-arguments
def _nms_with_mask_basic_api(box_scores, selected_boxes, selected_idx, selected_mask, iou_thr,
                             kernel_name="nms_with_mask"):
    """
    algorithm: new nms_with_mask

    find the best target bounding box and eliminate redundant bounding boxes

    Parameters
    ----------
    box_scores: dict
        2-D shape and dtype of input tensor, only support [N, 8]
        including proposal boxes and corresponding confidence scores

    selected_boxes: dict
        2-D shape and dtype of output boxes tensor, only support [N,5]
        including proposal boxes and corresponding confidence scores

    selected_idx: dict
        the index of input proposal boxes

    selected_mask: dict
        the symbol judging whether the output proposal boxes is valid

    iou_thr: float
        iou threshold

    kernel_name: str
        cce kernel name, default value is "nms_with_mask"

    Returns
    -------
    None
    """
    input_shape = box_scores.get("shape")
    input_dtype = box_scores.get("dtype").lower()

    # check dtype
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="box_scores")

    # check input shape
    if input_shape[1] != Constant.ELEMENT_NUM:
        error_manager_vector.raise_err_check_params_rules(kernel_name,
                                                          "the 2nd-dim of input boxes must be equal to 8",
                                                          "box_scores.shape", input_shape)

    nms_large_n_instance = NMSLargeN(input_shape[0], input_dtype, iou_thr)
    return nms_large_n_instance.main_func(kernel_name)


def _used_ub_size(n, input_dtype):
    """
    used size in ub

    Parameters
    ----------
    N: int
        value of input_shape[0]
    input_dtype: str
        input data type

    Returns
    -------
    size used in ub
    """
    if input_dtype == 'float16':
        input_bytes_each_elem = Constant.FP16_SIZE
    elif input_dtype == 'float32':
        input_bytes_each_elem = Constant.FP32_SIZE
    vector_mask_max = Constant.BURST_PROPOSAL_NUM // 2
    ceil_n = _ceiling(n, vector_mask_max)
    valid_mask_size_int8 = ceil_n
    dsorts_size = ceil_n
    bytes_each_elem = Constant.FP32_SIZE

    # x1, y2, x2, y2
    xy_size = ceil_n * bytes_each_elem * 4
    inp_size = ceil_n * Constant.ELEMENT_NUM * bytes_each_elem
    tmp_ub_size = (ceil_n + Constant.BURST_PROPOSAL_NUM) * Constant.FP16_SIZE
    next_nonzero_size = Constant.SHAPE_NEXT_NONZERO * Constant.INT32_SIZE
    output_mask_int8 = valid_mask_size_int8 * Constant.INT8_SIZE
    # xx1, yy1, xx2, yy2
    xxyy_size = ceil_n * bytes_each_elem * 3
    output_mask_f16_size = Constant.BURST_PROPOSAL_NUM * Constant.FP16_SIZE
    # size of data_fp16_zero and data_fp16_one
    data_fp16_zero_one_size = Constant.BURST_PROPOSAL_NUM * Constant.FP16_SIZE * 2
    # size used in _init_for_valid_mask()
    valid_mask_int8 = valid_mask_size_int8 * Constant.INT8_SIZE
    # size of tmp_valid_mask_float16 and tmp_mask_float16
    tmp_mask_size = valid_mask_size_int8 * Constant.FP16_SIZE
    # size used in _init_for_vcmax()
    init_for_vcmax_size = (dsorts_size + Constant.MASK_VCMAX_FP16 * 2 +
                           Constant.SHAPE_NEXT_NONZERO * 2) * Constant.FP16_SIZE
    # size used in _input_trans(), 16 is size of pattern, 4 means x1/y1/x2/y2
    input_trans_size = 8 * Constant.UINT32_SIZE * 4
    selected_boxes_size = ceil_n * Constant.VALID_COLUMN_NUM * input_bytes_each_elem
    pattern_size = 16 * Constant.UINT16_SIZE
    selected_idx_size = ceil_n * Constant.INT32_SIZE
    # size used in _area()
    area_size = ceil_n * bytes_each_elem
    # size used in loops()
    loops_size = ceil_n * Constant.UINT8_SIZE

    return xy_size + inp_size + tmp_ub_size + next_nonzero_size + output_mask_int8 + xxyy_size + \
           output_mask_f16_size + data_fp16_zero_one_size + valid_mask_int8 + tmp_mask_size + \
           init_for_vcmax_size + input_trans_size + selected_boxes_size + pattern_size + selected_idx_size + \
           area_size + loops_size
