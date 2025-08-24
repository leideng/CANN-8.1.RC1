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
ssd_detection_output
"""
# 'pylint: disable=too-many-lines,too-many-branches
import math

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tik
from impl import constant_util as constant
from impl import ssd_decode_bbox
from impl import topk
from impl import nms
from impl.util import util_select_op_base
from impl.util.util_common import get_mask_rep_stride
from impl.util.util_tik_comm_func import sort_score_idx_by_desc
from impl import cnms_yolov3_ver as cnms_yolo


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant
    """
    # define repeat elements every time for vsrot32
    REPEAT_ELE = 32
    # every loop process 4096 units
    PER_LOOP_UNIT = 4096
    # location elements, [x1, y1, x2, y2]
    FOUR_DIRECTION = 4
    # b16 elements num of every block also uesed as b16 elements num of mask
    BLOCK_ELE = 16
    # int32 elements num of every block also uesed as int32 elements num of mask
    INT32_BLOCK_ELE = 8
    # the socres_index contains four elements also marked as the class num processed every cycle
    UNIT_ELE = 4
    REPEAT_TIMES_MAX = 255
    # 0b0001 0001 0001 0001 is equals to type 3
    PATTERN_TYPE = 3
    # 0b1100 1100 1100 1100 is equals to type 52428
    PATTERN_NUM = 52428
    DATALEN_4K = 4096
    DATALEN_2K = 2048
    DATALEN_1K = 1024
    DATALEN_128 = 128
    INT32_MASK = 64


# 'pylint: disable=super-with-arguments
# 'pylint: disable=too-many-locals,too-many-statements,too-many-arguments,unused-argument,too-many-lines
def get_op_support_info(bbox_delta, score, anchors,
                        out_boxnum, output_y,
                        num_classes,
                        share_location=True,
                        background_label_id=0,
                        iou_threshold=0.45,
                        top_k=400,
                        eta=1.0,
                        variance_encoded_in_target=False,
                        code_type=1,
                        keep_top_k=-1,
                        confidence_threshold=0.0,
                        kernel_name="ssd_detection_output"):
    """
    get split info
    bbox_delta: [batch, N*num_loc_classes*4]
    score: [batch, N*Num_class]
    anchors: [1/batch, 2, N*Num_classes]
    only support split N
    """
    if anchors.get("shape")[0] == bbox_delta.get("shape"):
        return util_select_op_base.get_split_n_info([0, 1, 2], [0, 1])
    # can not split priorbox
    return util_select_op_base.get_split_n_info([0, 1], [0, 1])


def ceil_div(value, factor):
    """
    Compute the smallest integer value that is greater than
    or equal to value/factor
    """
    result = (value + (factor - 1)) // factor
    return result


def _check_product_info(input_dict):
    """
    check product info

    Parameters
    ----------
    input_dict: input dict

    Returns
    -------
    None
    """
    tik_name = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)

    conf_dtype = input_dict.get("mbox_conf").get("dtype").lower()
    loc_dtype = input_dict.get("mbox_loc").get("dtype").lower()
    priorbox_dtype = input_dict.get("mbox_priorbox").get("dtype").lower()

    if not conf_dtype == loc_dtype and conf_dtype == loc_dtype \
            and loc_dtype == priorbox_dtype:
        error_detail = "the dtype of inputs should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid("ssd_detection_output",
                                                               "mbox_conf, mbox_loc", "mbox_priorbox", error_detail)

    if tik_name in (tbe_platform.ASCEND_310,):
        para_check.check_dtype(conf_dtype.lower(), ["float16"], param_name="input_conf")
    elif tik_name in (tbe_platform.ASCEND_910,):
        para_check.check_dtype(conf_dtype.lower(), ["float16"], param_name="input_conf")
    elif tik_name in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        para_check.check_dtype(conf_dtype.lower(), ["float16"], param_name="input_conf")
    elif tik_name in (tbe_platform.ASCEND_610, tbe_platform.BS9SX1A, tbe_platform.ASCEND_310P):
        para_check.check_dtype(conf_dtype.lower(), ["float16", "float32"], param_name="input_conf")


def _check_param_range(param_name, min_value, max_value, real_value,
                       op_name='ssd_detection_output', left_open_interval=False):
    """
    check param range,

    Parameters
    ----------
    param_name: param name
    min_value: min value
    max_value: max value
    real_value: real value
    op_name: op name

    Returns
    -------
    None
    """
    if left_open_interval:
        error_manager_vector.raise_err_input_param_range_invalid(op_name, param_name, str(min_value),
                                                                 str(max_value), str(real_value))

    error_manager_vector.raise_err_input_param_range_invalid(op_name, param_name, str(min_value),
                                                             str(max_value), str(real_value))


def _check_input_attr_value(input_dict):
    """
    check input attr value,

    Parameters
    ----------
    input_dict: input dict

    Returns
    -------
    None
    """
    if not (1 <= input_dict.get("num_classes") <= 1024):
        _check_param_range('num_classes', 1, 1024, input_dict.get("num_classes"))

    if not input_dict.get("share_location"):
        error_manager_vector.raise_err_input_value_invalid("ssd_detection_output", "share_location", "True",
                                                           str(input_dict.get("share_location")))

    if not (input_dict.get("background_label_id") >= -1 and input_dict.get(
            "background_label_id") <= (input_dict.get("num_classes") - 1)):
        _check_param_range('background_label_id', -1,
                           input_dict.get("num_classes") - 1,
                           input_dict.get("background_label_id"))

    if not (0 < input_dict.get("nms_threshold") <= 1):
        _check_param_range('nms_threshold', 0, 1,
                           input_dict.get("nms_threshold"),
                           left_open_interval=True)

    if not input_dict.get("eta") == 1:
        error_manager_vector.raise_err_input_value_invalid("ssd_detection_output", "eta", "1",
                                                           str(input_dict.get("eta")))

    if not (1 <= input_dict.get("code_type") <= 3):
        _check_param_range('code_type', 1, 3, input_dict.get("code_type"))

    if not ((1024 >= input_dict.get("keep_top_k") > 0) or input_dict.get("keep_top_k") == -1):
        error_manager_vector.raise_err_input_param_range_invalid("ssd_detection_output", "keep_top_k", "0", "1024",
                                                                 str(input_dict.get("keep_top_k")))

    if not (0 <= input_dict.get("confidence_threshold") <= 1):
        _check_param_range('confidence_threshold', 0, 1,
                           input_dict.get("confidence_threshold"))

    _check_input_topk_value(input_dict.get("mbox_loc").get("dtype").lower(),
                            input_dict.get("top_k"))


def _check_input_topk_value(dtype, topk_value):
    """
    check input topk value,

    Parameters
    ----------
    input_dict: input dict

    Returns
    -------
    None
    """
    if tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION) in ("Hi3796CV300ES",
                                                                     "Hi3796CV300CS",
                                                                     "SD3403"):
        if dtype == "float32":
            if not topk_value <= 1500:
                _check_param_range('top_k', 1, 1500, topk_value)
        else:
            if not topk_value <= 3000:
                _check_param_range('top_k', 1, 3000, topk_value)

        if not topk_value <= 3000:
            _check_param_range('top_k', 1, 3000, topk_value)
    else:
        if dtype == "float32":
            if not topk_value <= 3000:
                _check_param_range('top_k', 1, 3000, topk_value)
        else:
            if not topk_value <= 6000:
                _check_param_range('top_k', 1, 6000, topk_value)


def _check_input_data_logical_relationship(input_dict):
    """
    _check_input_data_logical_relationship

    Parameters
    ----------
    input_dict: input dict

    Returns
    -------
    None
    """
    conf_shape = input_dict.get("mbox_conf").get("shape")
    loc_shape = input_dict.get("mbox_loc").get("shape")
    priorbox_shape = input_dict.get("mbox_priorbox").get("shape")
    num_classes = input_dict.get("num_classes")

    if not conf_shape[0] == loc_shape[0] and conf_shape[0] == priorbox_shape[0] \
            and loc_shape[0] == priorbox_shape[0]:
        error_detail = "the batch num of inputs should be equal"
        error_manager_vector.raise_err_two_input_shape_invalid("ssd_detection_output",
                                                               "mbox_conf, mbox_loc", "mbox_priorbox", error_detail)

    if not conf_shape[1] // num_classes == loc_shape[1] // 4:
        rule_desc = "the second dimension of mbox_conf divided by num_classes(%d) " \
                    "should be equal to the second dimension of mbox_loc(%d) divided by 4" \
                    % (num_classes, loc_shape[1])
        error_manager_vector.raise_err_check_params_rules("ssd_detection_output", rule_desc,
                                                          "the second dimension of mbox_conf", conf_shape[1])

    if not loc_shape[1] // 4 == priorbox_shape[2] // 4:
        rule_desc = "the second dimension of mbox_loc divided by 4 should be equal to " \
                    "the third dimension of mbox_priorbox(%d) divided by 4" % priorbox_shape[2]
        error_manager_vector.raise_err_check_params_rules("ssd_detection_output", rule_desc,
                                                          "the second dimension of mbox_loc", loc_shape[1])

    if not input_dict.get("variance_encoded_in_target"):
        if not priorbox_shape[1] == 2:
            rule_desc = "the second dimension of mbox_prior should be equal to 2"
            error_manager_vector.raise_err_check_params_rules("ssd_detection_output", rule_desc,
                                                              "the second dimension of mbox_prior", priorbox_shape[1])
    else:
        if not (priorbox_shape[1] == 2 or priorbox_shape[1] == 1):
            rule_desc = "the second dimension of mbox_prior should be equal to 1 or 2"
            error_manager_vector.raise_err_check_params_rules("ssd_detection_output", rule_desc,
                                                              "the second dimension of mbox_prior", priorbox_shape[1])


# 'pylint: disable=invalid-name, too-many-arguments, too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def ssd_detection_output(bbox_delta, score, anchors,
                         out_boxnum, y,
                         num_classes,
                         share_location=True,
                         background_label_id=0,
                         iou_threshold=0.45,
                         top_k=400,
                         eta=1.0,
                         variance_encoded_in_target=False,
                         code_type=1,
                         keep_top_k=-1,
                         confidence_threshold=0.0,
                         kernel_name="ssd_detection_output"):
    """
    the entry function of ssd detection output

    Parameters
    ----------
    mbox_conf: dict, the shape of mbox conf
    mbox_loc: dict, the shape of mbox loc
    mbox_priorbox: dict, the shape of mbox priorbox
    out_box_num: dict, the shape of out box number
    y: dict, the shape of out box
    num_classes: class num
    share_location: share location
    background_label_id: background label id
    nms_threshold: nms threshold
    top_k: class top num value
    eta: eta
    variance_encoded_in_target: variance_encoded_in_target
    code_type: code type
    keep_top_k: keep nms num value
    confidence_threshold: topk threshold
    kernel_name: cce kernel name

    Returns
    -------
    tik_instance:
    """
    input_dict = {
        "mbox_loc": bbox_delta,
        "mbox_conf": score,
        "mbox_priorbox": anchors,
        "out_box_num": out_boxnum,
        "out_box": y,
        "num_classes": num_classes,
        "share_location": share_location,
        "background_label_id": background_label_id,
        "nms_threshold": iou_threshold,
        "top_k": top_k,
        "eta": eta,
        "variance_encoded_in_target": variance_encoded_in_target,
        "code_type": code_type,
        "keep_top_k": keep_top_k,
        "confidence_threshold": confidence_threshold,
        "kernel_name": kernel_name
    }

    tik_instance = tik.Tik()
    _check_product_info(input_dict)
    _check_input_attr_value(input_dict)
    _check_input_data_logical_relationship(input_dict)

    decode_bbox_process = ssd_decode_bbox.SSDDecodeBBox(input_dict, tik_instance)

    detection_out_process = SSDDetectionOutput(
        input_dict, tik_instance, decode_bbox_process.decode_bbox_out_gm.shape[2] - decode_bbox_process.burnest_len)

    block_num, outer_loop, outer_tail = decode_bbox_process.get_block_param()
    with tik_instance.for_range(0, block_num, block_num=block_num) as block_i:
        batch = tik_instance.Scalar("int32", "batch", 0)
        output_gm_list = [detection_out_process.nmsed_boxes_gm[batch, :, :],
                          detection_out_process.nmsed_scores_gm[batch, :],
                          detection_out_process.nmsed_classes_gm[batch, :],
                          detection_out_process.nmsed_num_gm[batch, :]]
        with tik_instance.for_range(0, outer_loop) as outer_i:
            batch.set_as(block_i * outer_loop + outer_i)
            if tbe_platform.api_check_support("tik.vcopy") or decode_bbox_process.ascend_name in (
                tbe_platform.ASCEND_610, tbe_platform.BS9SX1A, tbe_platform.ASCEND_310P):
                decode_bbox_process.parser_loc_data_v200(batch)
                decode_bbox_process.parser_priorbox_data_v200(batch)
            else:
                decode_bbox_process.parser_loc_data(batch)
                decode_bbox_process.parser_priorbox_data(batch)
            decode_bbox_process.parser_conf_data(batch)

            decode_bbox_process.compute_detection_out(batch)
            if tbe_platform.api_check_support("tik.vcopy"):
                with tik_instance.for_range(0, input_dict.get("num_classes")) as cls_idx:
                    with tik_instance.if_scope(cls_idx != input_dict.get("background_label_id")):
                        detection_out_process.pre_topk_selection_class(batch,
                                                                       cls_idx,
                                                                       decode_bbox_process.decode_bbox_cnms_gm,
                                                                       decode_bbox_process.conf_data_parser_gm)

                        detection_out_process.cnms_calcation_class(batch,
                                                                   cls_idx,
                                                                   output_gm_list)
                detection_out_process.store_cnms_data_output(batch, output_gm_list)
            else:
                detection_out_process.get_topk_target_info(
                    batch, decode_bbox_process.decode_bbox_out_gm)
        if outer_tail > 0:
            with tik_instance.if_scope(block_i < outer_tail):
                batch.set_as(block_num * outer_loop + block_i)

                decode_bbox_process.parser_loc_data(batch)
                decode_bbox_process.parser_priorbox_data(batch)
                decode_bbox_process.parser_conf_data(batch)

                decode_bbox_process.compute_detection_out(batch)
                if tbe_platform.api_check_support("tik.vcopy"):
                    with tik_instance.for_range(0, input_dict.get("num_classes")) as cls_idx:
                        with tik_instance.if_scope(cls_idx != input_dict.get("background_label_id")):
                            detection_out_process.pre_topk_selection_class(batch,
                                                                           cls_idx,
                                                                           decode_bbox_process.decode_bbox_cnms_gm,
                                                                           decode_bbox_process.conf_data_parser_gm)
                            detection_out_process.cnms_calcation_class(batch,
                                                                       cls_idx,
                                                                       output_gm_list)
                    detection_out_process.store_cnms_data_output(batch, output_gm_list)
                else:
                    detection_out_process.get_topk_target_info(
                        batch, decode_bbox_process.decode_bbox_out_gm)

    tik_instance.BuildCCE(kernel_name=kernel_name,
                          inputs=(decode_bbox_process.mbox_loc_gm,
                                  decode_bbox_process.mbox_conf_gm,
                                  decode_bbox_process.mbox_prior_gm),
                          outputs=(detection_out_process.out_box_num_gm,
                                   detection_out_process.out_box_gm))

    return tik_instance


# 'pylint: disable=too-many-instance-attributes
class SSDDetectionOutput(ssd_decode_bbox.SSDDectionParamInit):
    """
    define SSDDetectionOutput class
    """

    def __init__(self, input_dict, tik_instance, topk_src_len):
        """
        SSDDetectionOutput init function

        Parameters
        ----------
        input_dict: dict, inout dict
        tik_instance: tik instance
        topk_src_len: topk num

        Returns
        -------
        None
        """
        self.instance = tik_instance
        super(SSDDetectionOutput, self).__init__(input_dict)

        # paser input args
        self.nms_threshold = input_dict.get("nms_threshold")
        if input_dict.get("top_k") <= 0 or input_dict.get("top_k") > 1024:
            self.top_k = Constant.DATALEN_1K
        else:
            self.top_k = input_dict.get("top_k")
        self.eta = input_dict.get("eta")
        self.keep_top_k = input_dict.get("keep_top_k")
        self.confidence_threshold = input_dict.get("confidence_threshold")

        # define for topk1
        self.topk1_in_gm_len = topk_src_len
        self.topk1_in_gm = self.instance.Tensor(self.dtype,
                                                (self.batch, self.topk1_in_gm_len, 8),
                                                name="topk1_in_gm",
                                                is_workspace=True,
                                                scope=tbe_platform.scope_gm)

        self.topk1_swap_gm = self.instance.Tensor(self.dtype,
                                                  (self.batch, self.topk1_in_gm_len, 8),
                                                  name="topk1_swap_gm",
                                                  is_workspace=True,
                                                  scope=tbe_platform.scope_gm)

        topk1_out_gm_len = math.ceil(input_dict["top_k"] / 16) * 16
        self.topk1_out_gm = self.instance.Tensor(self.dtype,
                                                 (self.batch, topk1_out_gm_len + 4, 8),
                                                 name="topk1_out_gm",
                                                 is_workspace=True,
                                                 scope=tbe_platform.scope_gm)
        # define for nms
        self.nms_box_num_gm = self.instance.Tensor(
            "int32",
            (self.batch, self.num_classes, 8),
            name="nms_box_num_gm",
            is_workspace=True,
            scope=tbe_platform.scope_gm)

        self.post_nms_topn = math.ceil(self.keep_top_k / 16) * 16
        if self.keep_top_k <= 0 or self.keep_top_k > self.top_k:
            self.post_nms_topn = math.ceil(self.top_k / 16) * 16
        self.nms_swap_gm = self.instance.Tensor(
            self.dtype,
            (self.num_classes * self.batch, self.post_nms_topn, 8),
            name="nms_swap_gm",
            is_workspace=True,
            scope=tbe_platform.scope_gm)

        self.nms_box_gm = self.instance.Tensor(
            self.dtype,
            (self.batch, self.num_classes, self.post_nms_topn, 8),
            name="nms_box_gm",
            is_workspace=True,
            scope=tbe_platform.scope_gm)

        # define for topk2
        topk2_in_gm_len = self.num_classes * self.post_nms_topn
        self.topk2_in_gm = self.instance.Tensor(self.dtype,
                                                (self.batch, topk2_in_gm_len, 8),
                                                name="topk2_in_gm",
                                                is_workspace=True,
                                                scope=tbe_platform.scope_gm)

        self.topk2_swap_gm = self.instance.Tensor(self.dtype,
                                                  (self.batch, topk2_in_gm_len, 8),
                                                  name="topk2_swap_gm",
                                                  is_workspace=True,
                                                  scope=tbe_platform.scope_gm)
        self.topk2_num = self.instance.Scalar("int32", "topk2_num", 0)

        # define for outbox
        out_box_len = ceil_div(self.keep_top_k, Constant.DATALEN_128) * Constant.DATALEN_128
        if self.keep_top_k <= 0:
            out_box_len = math.ceil(Constant.DATALEN_1K / Constant.DATALEN_128) * Constant.DATALEN_128
        self.out_box_len = out_box_len
        self.out_box_gm = self.instance.Tensor(self.dtype,
                                               (self.batch, out_box_len, 8),
                                               name="out_box_gm",
                                               scope=tbe_platform.scope_gm)

        self.out_box_gm_tmp = self.instance.Tensor(self.dtype,
                                                   (self.batch,
                                                    out_box_len, 8),
                                                   name="out_box_gm_tmp",
                                                   is_workspace=True,
                                                   scope=tbe_platform.scope_gm)

        self.out_box_num_gm = self.instance.Tensor("int32",
                                                   (self.batch, 8),
                                                   name="out_box_num_gm",
                                                   scope=tbe_platform.scope_gm)
        # define for topk3
        self.topk3_in_gm = self.instance.Tensor(self.dtype,
                                                (self.batch, topk2_in_gm_len, 8),
                                                name="topk3_in_gm",
                                                is_workspace=True,
                                                scope=tbe_platform.scope_gm)

        self.topk3_swap_gm = self.instance.Tensor(self.dtype,
                                                  (self.batch, topk2_in_gm_len, 8),
                                                  name="topk3_swap_gm",
                                                  is_workspace=True,
                                                  scope=tbe_platform.scope_gm)
        self.topk3_num = self.instance.Scalar("int32", "topk3_num", 0)
        self.topk3_out_gm = self.instance.Tensor(self.dtype,
                                                 (self.batch, out_box_len, 8),
                                                 name="topk3_out_gm",
                                                 is_workspace=True,
                                                 scope=tbe_platform.scope_gm)

        self.nmsed_boxes_gm = self.instance.Tensor(self.dtype, (self.batch, 4, self.max_total_size),
                                                   name="nmsed_boxes_gm",
                                                   is_workspace=True,
                                                   scope=tbe_platform.scope_gm)
        self.nmsed_scores_gm = self.instance.Tensor(self.dtype, (self.batch, self.max_total_size),
                                                    name="nmsed_scores_gm",
                                                    is_workspace=True,
                                                    scope=tbe_platform.scope_gm)
        self.nmsed_classes_gm = self.instance.Tensor(self.dtype, (self.batch, self.max_total_size),
                                                     name="nmsed_classes_gm",
                                                     is_workspace=True,
                                                     scope=tbe_platform.scope_gm)
        self.nmsed_num_gm = self.instance.Tensor("int32", (self.batch, 8),
                                                 name="nmsed_num_gm",
                                                 is_workspace=True,
                                                 is_atomic_add=True,
                                                 scope=tbe_platform.scope_gm)

        self.boxes_num = self.max_size_per_class
        idx_size = ceil_div(self.boxes_num, Constant.DATALEN_4K) * Constant.DATALEN_4K
        idx_init = [i for i in range(idx_size)]
        self.idx_gm = self.instance.Tensor("uint32",
                                           [idx_size, ],
                                           name="idx_gm",
                                           scope=tbe_platform.scope_gm,
                                           init_value=idx_init)
        self.workspace_ub_list = [None, None, None, None, None]  # xx1, yy1, xx2, yy2, scores

    def sort_each_class_prepare(self, batch, class_index, topk_src_data):
        """
        sort each class prepare

        Parameters
        ----------
        batch: batch num
        class_index: class num
        topk_src_data: topk data

        Returns
        -------
        None
        """
        with self.instance.new_stmt_scope():
            topk1_in_data_tmp_ub = self.instance.Tensor(self.dtype,
                                                        (self.ub_capacity,),
                                                        name="topk1_in_data_tmp_ub",
                                                        scope=tbe_platform.scope_ubuf)

            data_move_loop = self.topk1_in_gm_len * 8 // self.ub_capacity
            data_move_tail = self.topk1_in_gm_len * 8 % self.ub_capacity

            with self.instance.for_range(0, data_move_loop) as data_move_index:
                topk1_in_offset = data_move_index * (self.ub_capacity // 8)
                self.instance.data_move(topk1_in_data_tmp_ub,
                                        topk_src_data[batch, class_index, topk1_in_offset, 0],
                                        0, 1,
                                        (self.ub_capacity // self.burnest_len), 0, 0)

                self.instance.data_move(self.topk1_in_gm[batch, topk1_in_offset, 0],
                                        topk1_in_data_tmp_ub,
                                        0, 1,
                                        (self.ub_capacity // self.burnest_len), 0, 0)

            if data_move_tail > 0:
                topk1_in_offset = data_move_loop * self.ub_capacity // 8
                self.instance.data_move(topk1_in_data_tmp_ub,
                                        topk_src_data[batch, class_index, topk1_in_offset, 0],
                                        0, 1,
                                        (data_move_tail // self.burnest_len), 0, 0)

                self.instance.data_move(self.topk1_in_gm[batch, topk1_in_offset, 0],
                                        topk1_in_data_tmp_ub,
                                        0, 1,
                                        (data_move_tail // self.burnest_len), 0, 0)

    def get_tersor_data_burst_val(self, is_scalar, tersor_num_data, burst_val_tmp_scalar):
        """
        get tersor data burst val

        Parameters
        ----------
        is_scalar: whether tersor_num_data is scalar or not
        tersor_num_data: tersor data num
        burst_val_tmp_scalar: data move burst value

        Returns
        -------
        None
        """
        with self.instance.new_stmt_scope():
            nms_box_num_tmp_scalar = self.instance.Scalar(
                "int32", "nms_box_num_tmp_scalar", 0)

            if is_scalar:
                nms_box_num_tmp_scalar.set_as(tersor_num_data)
            else:
                nms_box_num_tmp_scalar.set_as(tersor_num_data[0])

            with self.instance.if_scope(nms_box_num_tmp_scalar % 2 != 0):
                nms_box_num_tmp_scalar.set_as(nms_box_num_tmp_scalar + 1)

            with self.instance.if_scope(self.dsize == 4):
                burst_val_tmp_scalar.set_as(nms_box_num_tmp_scalar)
            with self.instance.else_scope():
                burst_val_tmp_scalar.set_as(nms_box_num_tmp_scalar >> 1)

    def sort_all_class_prepare(self, batch, class_index, topk_num_ecah_class):
        """
        sort all class prepare

        Parameters
        ----------
        batch: batch num
        class_index: class index
        data_offset: data offset

        Returns
        -------
        None
        """
        with self.instance.new_stmt_scope():
            topk2_in_data_len = math.ceil(self.top_k / 16) * 16
            topk2_in_data_tmp_ub = self.instance.Tensor(
                self.dtype, (topk2_in_data_len, 8),
                name="topk2_in_data_tmp_ub",
                scope=tbe_platform.scope_ubuf)

            nms_bbox_num_ub = self.instance.Tensor("int32", (8,),
                                                   name="nms_bbox_num_ub",
                                                   scope=tbe_platform.scope_ubuf)
            self.instance.data_move(nms_bbox_num_ub,
                                    self.nms_box_num_gm[batch, class_index, 0],
                                    0, 1, 1, 0, 0)
            nms_num_scalar = self.instance.Scalar("int32", "nms_num_scalar",
                                                  nms_bbox_num_ub[0])

            burst_val_tmp_scalar = self.instance.Scalar("int32",
                                                        "burst_val_tmp_scalar", 0)
            self.get_tersor_data_burst_val(False, nms_bbox_num_ub,
                                           burst_val_tmp_scalar)

            with self.instance.if_scope(burst_val_tmp_scalar > 0):
                self.instance.data_move(topk2_in_data_tmp_ub,
                                        self.nms_box_gm[batch, class_index, 0, 0],
                                        0, 1,
                                        burst_val_tmp_scalar, 0, 0)
                self.instance.data_move(self.topk2_in_gm[batch, topk_num_ecah_class, 0],
                                        topk2_in_data_tmp_ub,
                                        0, 1,
                                        burst_val_tmp_scalar, 0, 0)

                with self.instance.for_range(0, nms_num_scalar) as num_index:
                    topk2_in_data_tmp_ub[num_index, 0].set_as(topk2_in_data_tmp_ub[num_index, 6])
                    topk2_in_data_tmp_ub[num_index, 1].set_as(topk2_in_data_tmp_ub[num_index, 5])

                self.instance.data_move(self.topk3_in_gm[batch, topk_num_ecah_class, 0],
                                        topk2_in_data_tmp_ub,
                                        0, 1,
                                        burst_val_tmp_scalar, 0, 0)

    def adjust_topk_crood(self, batch, topk_num_ecah_class):
        """
        modify x1 and y1 value

        Parameters
        ----------
        batch: batch num
        topk_num_ecah_class: out box data num

        Returns
        -------
        None
        """
        with self.instance.new_stmt_scope():
            box_data_ub = self.instance.Tensor(self.dtype,
                                               (self.out_box_gm.shape[1], 8),
                                               name="box_data_ub",
                                               scope=tbe_platform.scope_ubuf)
            topk3_out_ub = self.instance.Tensor(self.dtype,
                                                (self.out_box_gm.shape[1], 8),
                                                name="topk3_out_ub",
                                                scope=tbe_platform.scope_ubuf)
            burst_val_tmp_scalar = self.instance.Scalar("int32",
                                                        "burst_val_tmp_scalar", 0)

            with self.instance.if_scope(tik.all(self.keep_top_k > -1,
                                                topk_num_ecah_class > self.keep_top_k)):
                if self.keep_top_k > 0:
                    self.instance.data_move(box_data_ub,
                                            self.out_box_gm_tmp[batch, 0, 0],
                                            0, 1,
                                            math.ceil(self.keep_top_k * 8 / self.burnest_len),
                                            0, 0)
                    self.instance.data_move(topk3_out_ub, self.topk3_out_gm[batch, 0, 0],
                                            0, 1,
                                            math.ceil(self.keep_top_k * 8 / self.burnest_len),
                                            0, 0)

            with self.instance.else_scope():
                self.get_tersor_data_burst_val(True, topk_num_ecah_class,
                                               burst_val_tmp_scalar)
                with self.instance.if_scope(burst_val_tmp_scalar > 0):
                    self.instance.data_move(box_data_ub,
                                            self.topk2_in_gm[batch, 0, 0],
                                            0, 1, burst_val_tmp_scalar, 0, 0)
            self.set_crood_data_order(batch, topk_num_ecah_class, box_data_ub, topk3_out_ub)

    def set_crood_data_order(self, batch, topk_num_ecah_class, box_data_ub, topk3_out_ub):
        """
        modify out box data order as same as caffe

        Parameters
        ----------
        batch: batch num
        topk_num_ecah_class: out box data num

        Returns
        -------
        None
        """
        combin_out = self.instance.Tensor(self.dtype, (8, self.out_box_gm.shape[1]),
                                          name="combin_out", scope=tbe_platform.scope_ubuf)
        vnchw_src = self.instance.Tensor(self.dtype, (16 * self.out_box_gm.shape[1],),
                                         name="vnchw_src", scope=tbe_platform.scope_ubuf)
        vnchw_dst = self.instance.Tensor(self.dtype, (16 * self.out_box_gm.shape[1],),
                                         name="vnchw_dst", scope=tbe_platform.scope_ubuf)
        # fill in zero
        with self.instance.if_scope(topk_num_ecah_class < self.out_box_gm.shape[1]):
            start_offset = topk_num_ecah_class * self.dsize * 8 / constant.BLOCK_SIZE
            data_move_burst = (self.out_box_gm.shape[1] - start_offset) * 8 * self.dsize / constant.BLOCK_SIZE
            vector_dup_repeat = (data_move_burst * constant.BLOCK_SIZE + 127) / 128
            with self.instance.if_scope(data_move_burst > 0):
                scalar_zero = self.instance.Scalar(init_value=0, dtype=self.dtype)
                self.instance.vector_dup(self.mask, vnchw_dst, scalar_zero, vector_dup_repeat, 1, 8)
                self.instance.data_move(self.out_box_gm[batch, start_offset, 0],
                                        vnchw_dst, 0, 1, data_move_burst, 0, 0)

        with self.instance.for_range(0, topk_num_ecah_class) as combin_index:

            with self.instance.if_scope(tik.all(self.keep_top_k > -1,
                                                topk_num_ecah_class > self.keep_top_k)):
                with self.instance.if_scope(combin_index < self.keep_top_k):
                    combin_out[0, combin_index].set_as(topk3_out_ub[combin_index, 0])
                    combin_out[1, combin_index].set_as(topk3_out_ub[combin_index, 1])
                    combin_out[2, combin_index].set_as(box_data_ub[combin_index, 4])
            with self.instance.else_scope():
                combin_out[0, combin_index].set_as(box_data_ub[combin_index, 6])
                combin_out[1, combin_index].set_as(box_data_ub[combin_index, 5])
                combin_out[2, combin_index].set_as(box_data_ub[combin_index, 4])

        self.instance.vextract(combin_out[3, 0], box_data_ub,
                               self.out_box_gm.shape[1] // 16, 0)
        self.instance.vextract(combin_out[4, 0], box_data_ub,
                               self.out_box_gm.shape[1] // 16, 1)
        self.instance.vadds(self.mask,
                            combin_out[3, 0], combin_out[3, 0],
                            -1.0,
                            self.out_box_gm.shape[1] // self.mask,
                            1, 1, 8, 8)
        self.instance.vadds(self.mask,
                            combin_out[4, 0], combin_out[4, 0],
                            -1.0,
                            self.out_box_gm.shape[1] // self.mask,
                            1, 1, 8, 8)

        self.instance.vextract(combin_out[5, 0], box_data_ub,
                               self.out_box_gm.shape[1] // 16, 2)
        self.instance.vextract(combin_out[6, 0], box_data_ub,
                               self.out_box_gm.shape[1] // 16, 3)

        self.instance.data_move(vnchw_src[0], combin_out[0, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[16], combin_out[1, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[32], combin_out[2, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[48], combin_out[3, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[64], combin_out[4, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[80], combin_out[5, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)
        self.instance.data_move(vnchw_src[96], combin_out[6, 0], 0,
                                self.out_box_gm.shape[1] // self.burnest_len,
                                1, 0, 15)

        length = self.out_box_gm.shape[1] * 16 // 16
        tail_loop_times = ((length * 16) // (16 * 16)) % 255

        src_list = [vnchw_src[16 * i] for i in range(16)]
        dst_list = [vnchw_dst[16 * i] for i in range(16)]
        self.instance.vnchwconv(False, False, dst_list, src_list,
                                tail_loop_times, 16, 16)

        with self.instance.if_scope(tik.all(self.keep_top_k > -1,
                                            topk_num_ecah_class > self.keep_top_k)):
            if self.keep_top_k > 0:
                if self.dtype == "float32" or self.keep_top_k < 128:
                    with self.instance.for_range(0, self.keep_top_k) as index:
                        self.instance.data_move(self.out_box_gm[batch, index, 0],
                                                vnchw_dst[index * 16], 0, 1, 1, 0, 0)
                else:
                    last_block = self.keep_top_k - 2
                    with self.instance.for_range(0, last_block) as index:
                        self.instance.data_move(self.out_box_gm[batch, index, 0],
                                                vnchw_dst[index * 16], 0, 1, 1, 0, 0)
                    # copy last two
                    with self.instance.for_range(0, 8) as idx:
                        vnchw_dst[last_block * 16 + 8 + idx].set_as(vnchw_dst[(last_block + 1) * 16 + idx])
                    self.instance.data_move(self.out_box_gm[batch, last_block, 0],
                                            vnchw_dst[last_block * 16], 0, 1, 1, 0, 0)
        with self.instance.else_scope():
            with self.instance.for_range(0, topk_num_ecah_class) as index:
                self.instance.data_move(self.out_box_gm[batch, index, 0],
                                        vnchw_dst[index * 16], 0, 1, 1, 0, 0)
            # just fill first block
            if self.dtype == "float32":
                with self.instance.if_scope(topk_num_ecah_class < self.out_box_gm.shape[1]):
                    scalar_zero = self.instance.Scalar(init_value=0, dtype=self.dtype)
                    self.instance.vector_dup(self.mask, vnchw_dst, scalar_zero, 1, 1, 8)
                    self.instance.data_move(self.out_box_gm[batch, topk_num_ecah_class, 0],
                                            vnchw_dst, 0, 1, 1, 0, 0)
            else:
                with self.instance.if_scope(topk_num_ecah_class < self.out_box_gm.shape[1] - 1):
                    scalar_zero = self.instance.Scalar(init_value=0, dtype=self.dtype)
                    self.instance.vector_dup(self.mask, vnchw_dst, scalar_zero, 1, 1, 8)
                    self.instance.data_move(self.out_box_gm[batch, topk_num_ecah_class, 0],
                                            vnchw_dst, 0, 1, 1, 0, 0)

    def sort_each_class(self, batch, topk1_data_num, topk1_out_actual_num):
        """
        sort box

        Parameters
        ----------
        batch: batch num
        topk1_data_num: topk data num

        Returns
        -------
        None
        """
        topk_input_data = {
            "proposal_num": topk1_data_num,
            "k": self.top_k,
            "score_threshold": self.confidence_threshold,
            "regions_orig": self.topk1_in_gm,
            "mem_swap": self.topk1_swap_gm,
        }

        topk_out_data = {
            "batch_id": batch,
            "regions_sorted": self.topk1_out_gm,
            "proposal_actual_num": topk1_out_actual_num,
        }

        topk.tik_topk(self.instance, topk_input_data, topk_out_data)

    def nms_each_class(self, batch, class_index, topk1_out_actual_num):
        """
        nms box

        Parameters
        ----------
        batch: batch num
        class_index: class index

        Returns
        -------
        None
        """
        input_offset = batch * (((self.top_k + 15) // 16) * 16 + 4) * 8
        image_info = (817.55, 40)
        nms.cce_nms((self.dtype, self.ub_size,
                     self.nms_threshold, batch,
                     self.top_k, self.post_nms_topn,
                     input_offset, image_info,
                     self.instance, self.num_classes, class_index, batch),
                    self.nms_swap_gm,
                    self.topk1_out_gm,
                    topk1_out_actual_num,
                    self.nms_box_num_gm, self.nms_box_gm, False, used_in_ssd=True)

    def get_nms_all_class_result(self, batch, topk_num_ecah_class):
        """
        handle nms result

        Parameters
        ----------
        batch: batch num
        topk_num_ecah_class: nms result

        Returns
        -------
        None
        """
        with self.instance.if_scope(tik.all(self.keep_top_k > -1,
                                            topk_num_ecah_class > self.keep_top_k)):
            with self.instance.new_stmt_scope():
                topk2_tail_init_tmp_ub = self.instance.Tensor(
                    self.dtype, (128,), name="topk2_in_data_tmp_ub", scope=tbe_platform.scope_ubuf)
                self.instance.vector_dup(self.mask, topk2_tail_init_tmp_ub, 0,
                                         128 // self.mask, 1, 8)

                topk2_tail_num = self.instance.Scalar("int32", "topk_num_ecah_class", 16)
                burst_tail_scalar = self.instance.Scalar("int32", "burst_tail_scalar", 0)
                self.get_tersor_data_burst_val(True, topk2_tail_num, burst_tail_scalar)

                self.instance.data_move(self.topk2_in_gm[batch, topk_num_ecah_class, 0],
                                        topk2_tail_init_tmp_ub,
                                        0, 1, burst_tail_scalar, 0, 0)
                self.instance.data_move(self.topk3_in_gm[batch, topk_num_ecah_class, 0],
                                        topk2_tail_init_tmp_ub,
                                        0, 1, burst_tail_scalar, 0, 0)

            self.sort_all_class(batch)
            self.sort_for_get_label(batch)

        with self.instance.new_stmt_scope():
            # set out box num and tensor
            out_box_num_ub = self.instance.Tensor(
                "int32", (8,), name="out_box_num_ub", scope=tbe_platform.scope_ubuf)
            with self.instance.if_scope(tik.all(self.keep_top_k > -1,
                                                topk_num_ecah_class > self.keep_top_k)):
                out_box_num_ub[0].set_as(self.topk2_num)
            with self.instance.else_scope():
                out_box_num_ub[0].set_as(topk_num_ecah_class)
            self.instance.data_move(self.out_box_num_gm[batch, 0], out_box_num_ub,
                                    0, 1, 1, 0, 0)

        self.adjust_topk_crood(batch, topk_num_ecah_class)

    def get_topk_target_info(self, batch, topk_src_data):
        """
        get box result

        Parameters
        ----------
        batch: batch num
        topk_src_data: bbox data

        Returns
        -------
        None
        """
        topk_num_ecah_class = self.instance.Scalar(dtype="int32",
                                                   name="topk_num_ecah_class",
                                                   init_value=0)
        with self.instance.new_stmt_scope():
            topk2_init_tmp_ub = self.instance.Tensor(
                self.dtype, (128,), name="topk2_init_tmp_ub", scope=tbe_platform.scope_ubuf)
            self.instance.vector_dup(self.mask, topk2_init_tmp_ub, 0,
                                     128 // self.mask, 1, 8)

            move_loops = self.topk2_in_gm.shape[1] // 16
            with self.instance.for_range(0, move_loops) as move_index:
                move_offset = 16 * move_index
                self.instance.data_move(self.topk2_in_gm[batch, move_offset, 0],
                                        topk2_init_tmp_ub, 0, 1,
                                        128 // self.burnest_len, 0, 0)
                self.instance.data_move(self.topk3_in_gm[batch, move_offset, 0],
                                        topk2_init_tmp_ub, 0, 1,
                                        128 // self.burnest_len, 0, 0)

        topk1_out_actual_num = self.instance.Scalar("int32",
                                                    "topk1_out_actual_num",
                                                    0)
        with self.instance.for_range(0, self.num_classes) as class_index:
            with self.instance.if_scope(class_index != self.background_label_id):
                self.sort_each_class_prepare(batch, class_index, topk_src_data)
                self.sort_each_class(batch, topk_src_data.shape[2] - self.burnest_len,
                                     topk1_out_actual_num)
                self.nms_each_class(batch, class_index, topk1_out_actual_num)
                self.sort_all_class_prepare(batch, class_index, topk_num_ecah_class)

                with self.instance.new_stmt_scope():
                    nms_bbox_num_ub = self.instance.Tensor("int32", (8,),
                                                           name="nms_bbox_num_ub",
                                                           scope=tbe_platform.scope_ubuf)
                    self.instance.data_move(nms_bbox_num_ub,
                                            self.nms_box_num_gm[batch, class_index, 0],
                                            0, 1, 1, 0, 0)

                    topk_num_ecah_class_ub = self.instance.Tensor(
                        "int32", (8,),
                        name="topk_num_ecah_class_ub",
                        scope=tbe_platform.scope_ubuf)
                    topk_num_ecah_class_ub[0].set_as(topk_num_ecah_class)

                    topk_num_ecah_class_vadd_ub = self.instance.Tensor(
                        "int32", (8,),
                        name="topk_num_ecah_class_vadd_ub",
                        scope=tbe_platform.scope_ubuf)

                    self.instance.vadd(1, topk_num_ecah_class_vadd_ub,
                                       nms_bbox_num_ub, topk_num_ecah_class_ub,
                                       1, 1, 1, 1, 0, 0, 0)

                    topk_num_ecah_class.set_as(topk_num_ecah_class_vadd_ub[0])

        self.get_nms_all_class_result(batch, topk_num_ecah_class)

    def store_cnms_data_output(self, batch_idx, bbox_out_list):
        """
        store cnms output data into gm with SSD out formate
        :param batch_idx:
        :param bbox_out_list:
        :return:
        """
        boxes_out, scores_out, classes_out, box_num_out = bbox_out_list
        nms_len = self.instance.Scalar("int32", "nms_len", init_value=0)
        nms_len.set_as(box_num_out[batch_idx, 0])

        with self.instance.if_scope(nms_len > 0):
            with self.instance.if_scope(nms_len > self.out_box_len):
                nms_len.set_as(self.out_box_len)
            box_outnum_ub = self.instance.Tensor("int32",
                                                 (1, 8),
                                                 name="box_outnum_ub",
                                                 scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.instance, box_outnum_ub, 8)
            box_outnum_ub[0, 0].set_as(nms_len)
            # the size of box_outnum_ub is 32 Byte(1 Block)
            self.instance.data_move(self.out_box_num_gm[batch_idx, 0], box_outnum_ub, 0, 1, 1, 0, 0)

        with self.instance.new_stmt_scope():
            out_box_ub = self.instance.Tensor(self.dtype,
                                              (self.out_box_len, 8),
                                              name="out_box_ub",
                                              scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.instance, out_box_ub, self.out_box_len * 8)
            with self.instance.for_range(0, nms_len) as idx:
                out_box_ub[idx, 0].set_as(batch_idx)
                out_box_ub[idx, 1].set_as(classes_out[batch_idx, idx])
                out_box_ub[idx, 2].set_as(scores_out[batch_idx, idx])
                out_box_ub[idx, 3].set_as(boxes_out[batch_idx, 0, idx])
                out_box_ub[idx, 4].set_as(boxes_out[batch_idx, 1, idx])
                out_box_ub[idx, 5].set_as(boxes_out[batch_idx, 2, idx])
                out_box_ub[idx, 6].set_as(boxes_out[batch_idx, 3, idx])
                out_box_ub[idx, 7].set_as(0)

            burst_times = (self.out_box_len * 8) // Constant.BLOCK_ELE
            with self.instance.if_scope(burst_times > 0):
                self.instance.data_move(self.out_box_gm, out_box_ub, 0, 1, burst_times, 0, 0)

    def partial_init_tensor(self, dst, size, start, init_value=0):
        """
        init party memory of tensor
        :param dst: ub memory
        :param size: tensor size
        :param start: start init address
        :param init_value:
        :return:
        """
        vector_mask, rep_stride = get_mask_rep_stride(dst)
        aligned_start = ceil_div(start, 32) * 32
        length = size - aligned_start
        max_lens = Constant.REPEAT_TIMES_MAX * vector_mask
        loop_num = length // max_lens
        tail = length % max_lens
        repeat_times = tail // vector_mask
        tail_aligned = tail % vector_mask

        with self.instance.for_range(start, aligned_start) as idx:
            dst[idx].set_as(init_value)

        off = self.instance.Scalar("uint32")
        with self.instance.for_range(0, loop_num) as idx:
            off.set_as(vector_mask * Constant.REPEAT_TIMES_MAX * idx)
            self.instance.vec_dup(vector_mask,
                                  dst[aligned_start + off],
                                  init_value,
                                  Constant.REPEAT_TIMES_MAX,
                                  rep_stride)
        with self.instance.if_scope(tik.all(tail != 0, repeat_times > 0)):
            offset = length - tail
            self.instance.vec_dup(vector_mask,
                                  dst[aligned_start + offset],
                                  init_value,
                                  repeat_times,
                                  rep_stride)
        with self.instance.if_scope(tail_aligned != 0):
            with self.instance.for_range(0, tail_aligned) as idx:
                dst[aligned_start + length - tail_aligned + idx].set_as(init_value)

    def cnms_calcation_class(self, batch_idx, class_idx, bbox_out_list):
        """
        execute cnms calculation for per classes
        :param batch_idx:
        :param class_idx:
        :param bbox_out_list:
        :return:
        """
        shape_aligned = Constant.PER_LOOP_UNIT

        x1_ub = self.workspace_ub_list[0]
        y1_ub = self.workspace_ub_list[1]
        x2_ub = self.workspace_ub_list[2]
        y2_ub = self.workspace_ub_list[3]
        scores_ub = self.workspace_ub_list[4]

        # select by scores_threshold
        eff_lens = self.instance.Scalar("uint32", "eff_lens", 0)
        eff_lens.set_as(shape_aligned)
        self.scores_threshold_selection_class(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, self.score_thresh, eff_lens)

        with self.instance.if_scope(eff_lens > 0):
            # do iou selection
            self.ssd_iou_selection(x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, eff_lens)
            # do post topk
            self.post_topk_selection_class(eff_lens)
            # store data
            with self.instance.if_scope(eff_lens > 0):
                self.store_cnms_data_per_class(batch_idx, class_idx,
                                               x1_ub, x2_ub, y1_ub, y2_ub, scores_ub,
                                               bbox_out_list, eff_lens)

    def store_cnms_data_per_class(self, batch_idx, class_idx,
                                  xx1, xx2, yy1, yy2, scores_ub, bbox_out_list, eff_lens):
        """
        save data which select by cnms
        :param batch_idx:
        :param class_idx:
        :param xx1:
        :param xx2:
        :param yy1:
        :param yy2:
        :param scores_ub:
        :param bbox_out_list:
        :param eff_lens:
        :return:
        """
        boxes_out, scores_out, class_out, box_num_out = bbox_out_list
        valid_detection = self.instance.Scalar("int32", "valid_detection", 0)
        valid_detection.set_as(box_num_out[batch_idx, 0])

        box_outnum_ub = self.instance.Tensor("int32",
                                             (self.batch, 8),
                                             name="box_outnum_ub",
                                             scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.instance, box_outnum_ub, 8, 0)
        box_outnum_ub[batch_idx, 0].set_as(valid_detection + eff_lens)
        self.instance.data_move(box_num_out, box_outnum_ub, 0, 1, 1, 0, 0)

        with self.instance.new_stmt_scope():
            out_length = ceil_div(self.max_size_per_class, Constant.BLOCK_ELE) * Constant.BLOCK_ELE
            class_out_ub = self.instance.Tensor(self.dtype,
                                                (1, out_length),
                                                name="class_out_ub",
                                                scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.instance, class_out_ub, out_length)
            burst_len = ceil_div((valid_detection + eff_lens), Constant.BLOCK_ELE) * Constant.BLOCK_ELE
            self.instance.data_move(class_out_ub, class_out, 0, 1, burst_len, 0, 0)
            _cls_idx = self.instance.Scalar(self.dtype, "_cls_idx")
            _cls_idx.set_as(class_idx)
            with self.instance.for_range(valid_detection, valid_detection + eff_lens) as idx:
                class_out_ub[0, idx].set_as(_cls_idx)
            self.instance.data_move(class_out, class_out_ub, 0, 1, burst_len, 0, 0)

        repeat_times = ceil_div(eff_lens, Constant.BLOCK_ELE)
        with self.instance.if_scope(repeat_times > 0):
            self.instance.data_move(boxes_out[batch_idx, 0, valid_detection], xx1, 0, 1, repeat_times, 0, 0)
            self.instance.data_move(boxes_out[batch_idx, 1, valid_detection], yy1, 0, 1, repeat_times, 0, 0)
            self.instance.data_move(boxes_out[batch_idx, 2, valid_detection], xx2, 0, 1, repeat_times, 0, 0)
            self.instance.data_move(boxes_out[batch_idx, 3, valid_detection], yy2, 0, 1, repeat_times, 0, 0)
            self.instance.data_move(scores_out[batch_idx, valid_detection], scores_ub, 0, 1, repeat_times, 0, 0)

    def ssd_iou_selection(self, x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, eff_lens):
        """
        execute iou selection,
        :param x1_ub:
        :param x2_ub:
        :param y1_ub:
        :param y2_ub:
        :param scores_ub:
        :param eff_lens:
        :return:
        """
        shape_aligned = Constant.PER_LOOP_UNIT
        mask, _ = get_mask_rep_stride(x1_ub)

        # iou Selection for only topk data for per class
        single_area = self.instance.Tensor(self.dtype, [shape_aligned, ], name="single_area",
                                           scope=tbe_platform.scope_ubuf)
        # get area of every windows
        cnms_yolo.get_rectangle_area(self.instance, [x1_ub, x2_ub, y1_ub, y2_ub], single_area, shape_aligned)

        iou = self.instance.Tensor(self.dtype, [shape_aligned, ], name="iou",
                                   scope=tbe_platform.scope_ubuf)
        # calculate the iou, exit when the output windows is more than eff_lens
        overlap = self.instance.Tensor(self.dtype, [shape_aligned, ], name="overlap",
                                       scope=tbe_platform.scope_ubuf)
        # define tmp tensor for following use, to reduce the cycle of apply/release memory
        tmp1 = self.instance.Tensor(self.dtype, [shape_aligned, ], name="tmp1", scope=tbe_platform.scope_ubuf)
        tmp2 = self.instance.Tensor(self.dtype, [shape_aligned, ], name="tmp2", scope=tbe_platform.scope_ubuf)
        mask_shape_lens = Constant.PER_LOOP_UNIT // Constant.BLOCK_ELE
        mask_uint16 = self.instance.Tensor("uint16", [mask_shape_lens, ], name="mask_uint16",
                                           scope=tbe_platform.scope_ubuf)
        iou_thresh = self.iou_thresh / (1 + self.iou_thresh)

        # calculate ioues for every windows
        with self.instance.for_range(0, self.top_k) as idx:
            with self.instance.if_scope(idx < eff_lens):
                cnms_yolo.get_overlap(self.instance, [x1_ub, x2_ub, y1_ub, y2_ub], 
                                      [overlap, tmp1, tmp2], idx, shape_aligned)
                cnms_yolo.init_tensor(self.instance, tmp2, shape_aligned)
                _aligned_length = ceil_div(eff_lens, mask) * mask
                cnms_yolo.cal_iou(self.instance, [single_area, iou, tmp2], idx, _aligned_length, iou_thresh)
                self.partial_init_tensor(iou, _aligned_length, eff_lens)
                cnms_yolo.gen_mask(self.instance, overlap, iou, mask_uint16, size=Constant.PER_LOOP_UNIT)
                cnms_yolo.update_input_v300_resvd(self.instance,
                                                  x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, single_area,
                                                  eff_lens, tmp1, tmp2, mask_uint16)
            with self.instance.else_scope():
                self.instance.tik_break()

    def data_move(self, dst, src, length):
        """
        move data beteen gm and ub
        :param dst: memory space in UB or GM, if in GM, src must in UB
        :param src: memory space in UB or GM, if in GM, dst must in UB
        :param length: Data length which is Aligned by 32 Bytes
        :return:
        """
        burst_len = length // Constant.BLOCK_ELE
        with self.instance.if_scope(burst_len > 0):
            self.instance.data_move(dst, src, 0, 1, burst_len, 0, 0)

    def pre_topk_select_data(self, ub_list, loc_gm_list, topk_ub_list, handle_length, top_k):
        """
        execute topk select for per block data
        :param ub_list:
        :param loc_gm_list:
        :param topk_ub_list:
        :param handle_length:
        :param top_k:
        :return:
        """
        topk_length = ceil_div(top_k, Constant.REPEAT_ELE) * Constant.REPEAT_ELE
        x1_ub, y1_ub, x2_ub, y2_ub, scores_ub = ub_list
        x1_gm, y1_gm, x2_gm, y2_gm, scores_gm = loc_gm_list
        topk_x1_ub, topk_y1_ub, topk_x2_ub, topk_y2_ub, topk_scores_ub = topk_ub_list

        # step1: prepare data from topk selection
        cnms_yolo.init_tensor(self.instance, x1_ub, Constant.PER_LOOP_UNIT)
        cnms_yolo.init_tensor(self.instance, y1_ub, Constant.PER_LOOP_UNIT)
        cnms_yolo.init_tensor(self.instance, x2_ub, Constant.PER_LOOP_UNIT)
        cnms_yolo.init_tensor(self.instance, y2_ub, Constant.PER_LOOP_UNIT)
        cnms_yolo.init_tensor(self.instance, scores_ub, Constant.PER_LOOP_UNIT)
        self.data_move(x1_ub, topk_x1_ub, topk_length)
        self.data_move(y1_ub, topk_y1_ub, topk_length)
        self.data_move(x2_ub, topk_x2_ub, topk_length)
        self.data_move(y2_ub, topk_y2_ub, topk_length)
        self.data_move(scores_ub, topk_scores_ub, topk_length)

        self.data_move(x1_ub[topk_length:], x1_gm, handle_length)
        self.data_move(y1_ub[topk_length:], y1_gm, handle_length)
        self.data_move(x2_ub[topk_length:], x2_gm, handle_length)
        self.data_move(y2_ub[topk_length:], y2_gm, handle_length)
        self.data_move(scores_ub[topk_length:], scores_gm, handle_length)

        cnms_yolo.init_tensor(self.instance, topk_x1_ub, topk_length)
        cnms_yolo.init_tensor(self.instance, topk_y1_ub, topk_length)
        cnms_yolo.init_tensor(self.instance, topk_x2_ub, topk_length)
        cnms_yolo.init_tensor(self.instance, topk_y2_ub, topk_length)
        cnms_yolo.init_tensor(self.instance, topk_scores_ub, topk_length)

        total_data_length = Constant.DATALEN_2K
        score_idx_lens = total_data_length * Constant.UNIT_ELE
        scores_idx_out = self.instance.Tensor(self.dtype, [score_idx_lens * 2], name="scores_idx_out",
                                              scope=tbe_platform.scope_ubuf)
        scores_idx_ub = self.instance.Tensor(self.dtype, [score_idx_lens * 2], name="scores_idx_ub",
                                             scope=tbe_platform.scope_ubuf)
        index_sort_ub = self.instance.Tensor(self.dtype, [total_data_length * 2], name="index_sort_ub",
                                             scope=tbe_platform.scope_ubuf)
        with self.instance.new_stmt_scope():
            index_ub = self.instance.Tensor("uint32", [total_data_length, ], name="index_ub",
                                            scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_index(self.instance, self.idx_gm, index_ub, 0, total_data_length)
            repeat_times = total_data_length // Constant.REPEAT_ELE
            self.instance.vsort32(scores_idx_ub, scores_ub, index_ub, repeat_times)
            sort_score_idx_by_desc(self.instance, scores_idx_ub, scores_idx_out, score_idx_lens)
            cnms_yolo.init_tensor(self.instance, scores_idx_ub, total_data_length)
            mask, _ = get_mask_rep_stride(scores_idx_ub)
            repeat_times = total_data_length * Constant.FOUR_DIRECTION // mask
            self.instance.vreducev2(None, scores_idx_ub, scores_idx_out, Constant.PATTERN_TYPE, repeat_times, 1, 8, 0)

            pattern_shape = mask // Constant.BLOCK_ELE
            vreducev2_pattern = self.instance.Tensor("uint16", [pattern_shape, ], name="vreducev2_pattern",
                                                    scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.instance, vreducev2_pattern, pattern_shape, Constant.PATTERN_NUM)
            self.instance.vreducev2(mask, index_sort_ub, scores_idx_out,
                                    vreducev2_pattern, repeat_times, 1, 8, 0, mask_mode="counter")

        if tbe_platform.api_check_support("tik.vgather"):
            lo_index = self.instance.Scalar("int32", init_value=2)
            mask_length = ceil_div(top_k, Constant.DATALEN_128) * Constant.DATALEN_128
            topk_index_ub = self.instance.Tensor("int32", [mask_length, ], name="topk_index_ub",
                                                     scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.instance, topk_index_ub, mask_length)

            with self.instance.if_scope(top_k >= Constant.BLOCK_ELE):
                top_k_loop = top_k // Constant.BLOCK_ELE * Constant.BLOCK_ELE
                top_k_tail = top_k - top_k_loop

                self.data_move(topk_scores_ub, scores_idx_ub, top_k_loop)

                index_sort_ub_int32 = index_sort_ub.reinterpret_cast_to("int32")

                int32_loop = top_k // Constant.INT32_BLOCK_ELE * Constant.INT32_BLOCK_ELE
                burst_len = top_k // Constant.INT32_BLOCK_ELE
                int32_tail = top_k - int32_loop

                self.instance.data_move(topk_index_ub, index_sort_ub_int32, 0, 1, burst_len, 0, 0)

                with self.instance.if_scope(top_k_tail > 0):
                    with self.instance.for_range(top_k_loop, top_k) as top_k_idx:
                        topk_scores_ub[top_k_idx].set_as(scores_idx_ub[top_k_idx])

                with self.instance.if_scope(int32_tail > 0):
                    with self.instance.for_range(int32_loop, top_k) as top_k_int32_idx:
                        scores_index_offset = top_k_int32_idx * Constant.UNIT_ELE
                        topk_index_ub[top_k_int32_idx].set_as(
                            scores_idx_out[scores_index_offset + 2:
                                scores_index_offset + 4].reinterpret_cast_to("int32"))

            with self.instance.else_scope():
                with self.instance.for_range(0, top_k) as idx:
                    topk_scores_ub[idx].set_as(scores_idx_ub[idx])
                    scores_index_offset = idx * Constant.UNIT_ELE
                    topk_index_ub[idx].set_as(
                    scores_idx_out[scores_index_offset + 2: scores_index_offset + 4].reinterpret_cast_to("int32"))

            mul_loop = mask_length // Constant.INT32_MASK
            self.instance.vmuls(Constant.INT32_MASK, topk_index_ub, topk_index_ub, lo_index, mul_loop, 1, 1, 8, 8)
            loop_len = top_k // Constant.DATALEN_128
            loop_tail = top_k % Constant.DATALEN_128
            with self.instance.if_scope(loop_len > 0):
                self.instance.vgather(Constant.DATALEN_128, topk_x1_ub, x1_ub, topk_index_ub, loop_len,
                                          8, 0, 0, mask_mode="normal")
                self.instance.vgather(Constant.DATALEN_128, topk_y1_ub, y1_ub, topk_index_ub, loop_len,
                                          8, 0, 0, mask_mode="normal")
                self.instance.vgather(Constant.DATALEN_128, topk_x2_ub, x2_ub, topk_index_ub, loop_len,
                                          8, 0, 0, mask_mode="normal")
                self.instance.vgather(Constant.DATALEN_128, topk_y2_ub, y2_ub, topk_index_ub, loop_len,
                                          8, 0, 0, mask_mode="normal")
            with self.instance.if_scope(loop_tail > 0):
                self.instance.vgather(loop_tail, topk_x1_ub[loop_len * Constant.DATALEN_128],
                                          x1_ub, topk_index_ub[loop_len * Constant.DATALEN_128],
                                          1, 8, 0, mask_mode="normal")
                self.instance.vgather(loop_tail, topk_y1_ub[loop_len * Constant.DATALEN_128],
                                          y1_ub, topk_index_ub[loop_len * Constant.DATALEN_128],
                                          1, 8, 0, mask_mode="normal")
                self.instance.vgather(loop_tail, topk_x2_ub[loop_len * Constant.DATALEN_128],
                                          x2_ub, topk_index_ub[loop_len * Constant.DATALEN_128],
                                          1, 8, 0, mask_mode="normal")
                self.instance.vgather(loop_tail, topk_y2_ub[loop_len * Constant.DATALEN_128],
                                          y2_ub, topk_index_ub[loop_len * Constant.DATALEN_128],
                                          1, 8, 0, mask_mode="normal")
        else:
            lo_index = self.instance.Scalar("uint32", init_value=0)
            with self.instance.for_range(0, top_k) as idx:
                topk_scores_ub[idx].set_as(scores_idx_ub[idx])
                scores_index_offset = idx * Constant.UNIT_ELE
                lo_index.set_as(
                    scores_idx_out[scores_index_offset + 2: scores_index_offset + 4].reinterpret_cast_to("uint32"))
                topk_x1_ub[idx].set_as(x1_ub[lo_index])
                topk_y1_ub[idx].set_as(y1_ub[lo_index])
                topk_x2_ub[idx].set_as(x2_ub[lo_index])
                topk_y2_ub[idx].set_as(y2_ub[lo_index])

    def pre_topk_selection_class(self, batch_idx, class_idx, bbox_cnms_gm, scores_gm):
        """
        topk selection for each class
        :param batch_idx:
        :param class_idx:
        :param bbox_cnms_gm:
        :param scores_gm:
        :return:
        """

        shape_aligned = Constant.PER_LOOP_UNIT
        x1_ub = self.instance.Tensor(self.dtype, [shape_aligned, ], name="x1_ub", scope=tbe_platform.scope_ubuf)
        y1_ub = self.instance.Tensor(self.dtype, [shape_aligned, ], name="y1_ub", scope=tbe_platform.scope_ubuf)
        x2_ub = self.instance.Tensor(self.dtype, [shape_aligned, ], name="x2_ub", scope=tbe_platform.scope_ubuf)
        y2_ub = self.instance.Tensor(self.dtype, [shape_aligned, ], name="y2_ub", scope=tbe_platform.scope_ubuf)
        scores_ub = self.instance.Tensor(self.dtype, [shape_aligned, ], name="scores_ub", scope=tbe_platform.scope_ubuf)
        cnms_yolo.init_tensor(self.instance, x1_ub, shape_aligned)
        cnms_yolo.init_tensor(self.instance, y1_ub, shape_aligned)
        cnms_yolo.init_tensor(self.instance, x2_ub, shape_aligned)
        cnms_yolo.init_tensor(self.instance, y2_ub, shape_aligned)
        cnms_yolo.init_tensor(self.instance, scores_ub, shape_aligned)

        with self.instance.new_stmt_scope():
            handle_length = Constant.DATALEN_1K
            topk_length = ceil_div(self.top_k, Constant.REPEAT_ELE) * Constant.REPEAT_ELE
            topk_x1_ub = self.instance.Tensor(self.dtype, [topk_length, ], name="topk_x1_ub",
                                              scope=tbe_platform.scope_ubuf)
            topk_y1_ub = self.instance.Tensor(self.dtype, [topk_length, ], name="topk_y1_ub",
                                              scope=tbe_platform.scope_ubuf)
            topk_x2_ub = self.instance.Tensor(self.dtype, [topk_length, ], name="topk_x2_ub",
                                              scope=tbe_platform.scope_ubuf)
            topk_y2_ub = self.instance.Tensor(self.dtype, [topk_length, ], name="topk_y2_ub",
                                              scope=tbe_platform.scope_ubuf)
            topk_scores_ub = self.instance.Tensor(self.dtype, [topk_length, ], name="topk_scores_ub",
                                                  scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.instance, topk_x1_ub, topk_length)
            cnms_yolo.init_tensor(self.instance, topk_y1_ub, topk_length)
            cnms_yolo.init_tensor(self.instance, topk_x2_ub, topk_length)
            cnms_yolo.init_tensor(self.instance, topk_y2_ub, topk_length)
            cnms_yolo.init_tensor(self.instance, topk_scores_ub, topk_length)

            loc_repeat_times = scores_gm.shape[2] // handle_length
            loc_tails_length = scores_gm.shape[2] % handle_length
            with self.instance.for_range(0, loc_repeat_times) as idx:
                self.pre_topk_select_data([x1_ub, y1_ub, x2_ub, y2_ub, scores_ub],
                                          [bbox_cnms_gm[batch_idx, 0, (idx * handle_length):],
                                           bbox_cnms_gm[batch_idx, 1, (idx * handle_length):],
                                           bbox_cnms_gm[batch_idx, 2, (idx * handle_length):],
                                           bbox_cnms_gm[batch_idx, 3, (idx * handle_length):],
                                           scores_gm[batch_idx, class_idx, (idx * handle_length):]],
                                          [topk_x1_ub, topk_y1_ub, topk_x2_ub, topk_y2_ub, topk_scores_ub],
                                          handle_length, self.top_k)

            with self.instance.if_scope(loc_tails_length > 0):
                self.pre_topk_select_data([x1_ub, y1_ub, x2_ub, y2_ub, scores_ub],
                                          [bbox_cnms_gm[batch_idx, 0, (loc_repeat_times * handle_length):],
                                           bbox_cnms_gm[batch_idx, 1, (loc_repeat_times * handle_length):],
                                           bbox_cnms_gm[batch_idx, 2, (loc_repeat_times * handle_length):],
                                           bbox_cnms_gm[batch_idx, 3, (loc_repeat_times * handle_length):],
                                           scores_gm[batch_idx, class_idx, (loc_repeat_times * handle_length):]],
                                          [topk_x1_ub, topk_y1_ub, topk_x2_ub, topk_y2_ub, topk_scores_ub],
                                          loc_tails_length, self.top_k)
            cnms_yolo.init_tensor(self.instance, x1_ub, shape_aligned)
            cnms_yolo.init_tensor(self.instance, y1_ub, shape_aligned)
            cnms_yolo.init_tensor(self.instance, x2_ub, shape_aligned)
            cnms_yolo.init_tensor(self.instance, y2_ub, shape_aligned)
            cnms_yolo.init_tensor(self.instance, scores_ub, shape_aligned)
            self.data_move(x1_ub, topk_x1_ub, topk_length)
            self.data_move(y1_ub, topk_y1_ub, topk_length)
            self.data_move(x2_ub, topk_x2_ub, topk_length)
            self.data_move(y2_ub, topk_y2_ub, topk_length)
            self.data_move(scores_ub, topk_scores_ub, topk_length)
        cnms_yolo.exchange_coordinate(self.instance, [x1_ub, x2_ub, y1_ub, y2_ub], shape_aligned)
        self.workspace_ub_list[0] = x1_ub
        self.workspace_ub_list[1] = y1_ub
        self.workspace_ub_list[2] = x2_ub
        self.workspace_ub_list[3] = y2_ub
        self.workspace_ub_list[4] = scores_ub

    def post_topk_selection_class(self, eff_lens):
        """
        post topk selection, if keep_top_k > 0, set eff_lens to keep_top_k
        :param eff_lens:
        :return:
        """
        topk_scalar = self.instance.Scalar("int32", "topk_scalar", 0)
        topk_scalar.set_as(self.keep_top_k)
        if self.keep_top_k > 0:
            with self.instance.if_scope(topk_scalar < eff_lens):
                eff_lens.set_as(self.keep_top_k)

    def scores_threshold_selection_class(self,
                                         x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, scores_value, eff_lens):
        """

        :param x1_ub:
        :param x2_ub:
        :param y1_ub:
        :param y2_ub:
        :param scores_ub:
        :param scores_value:
        :param eff_lens:
        :return:
        """
        shape_size = Constant.PER_LOOP_UNIT
        shape = (shape_size,)
        mask_shape = (ceil_div(shape_size, Constant.BLOCK_ELE),)
        with self.instance.new_stmt_scope():
            scores_thresh = self.instance.Tensor(self.dtype, shape, name="scores_threshold",
                                                 scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.instance, scores_thresh, shape_size, scores_value)

            tmp1 = self.instance.Tensor(self.dtype, [shape_size, ], name="tmp1", scope=tbe_platform.scope_ubuf)
            tmp2 = self.instance.Tensor(self.dtype, [shape_size, ], name="tmp2", scope=tbe_platform.scope_ubuf)
            _single_area = self.instance.Tensor(self.dtype, [shape_size, ], name="_single_area",
                                                scope=tbe_platform.scope_ubuf)

            mask_uint16 = self.instance.Tensor("uint16", mask_shape, name="mask_uint16", scope=tbe_platform.scope_ubuf)
            cnms_yolo.init_tensor(self.instance, mask_uint16, ceil_div(shape_size, Constant.BLOCK_ELE), 0)
            cnms_yolo.gen_mask(self.instance, scores_thresh, scores_ub, mask_uint16)
            cnms_yolo.update_input_v300_resvd(self.instance,
                                              x1_ub, x2_ub, y1_ub, y2_ub, scores_ub, _single_area,
                                              eff_lens, tmp1, tmp2, mask_uint16)

    def sort_all_class(self, batch):
        """
        sort box

        Parameters
        ----------
        batch: batch num
        Returns
        -------
        None
        """
        topk_in_num = self.keep_top_k
        if self.keep_top_k <= 0:
            topk_in_num = self.top_k
        topk_input_data = {
            "proposal_num": self.topk2_in_gm.shape[1],
            "k": topk_in_num,
            "score_threshold": self.confidence_threshold,
            "regions_orig": self.topk2_in_gm,
            "mem_swap": self.topk2_swap_gm,
        }

        topk_out_data = {
            "batch_id": batch,
            "regions_sorted": self.out_box_gm_tmp,
            "proposal_actual_num": self.topk2_num,
        }

        topk.tik_topk(self.instance, topk_input_data, topk_out_data)

    def sort_for_get_label(self, batch):
        """
        sort box

        Parameters
        ----------
        batch: batch num
        Returns
        -------
        None
        """
        topk_in_num = self.keep_top_k
        if self.keep_top_k <= 0:
            topk_in_num = self.top_k

        topk_input_data = {
            "proposal_num": self.topk3_in_gm.shape[1],
            "k": topk_in_num,
            "score_threshold": self.confidence_threshold,
            "regions_orig": self.topk3_in_gm,
            "mem_swap": self.topk3_swap_gm,
        }

        topk_out_data = {
            "batch_id": batch,
            "regions_sorted": self.topk3_out_gm,
            "proposal_actual_num": self.topk3_num,
        }

        topk.tik_topk(self.instance, topk_input_data, topk_out_data)
