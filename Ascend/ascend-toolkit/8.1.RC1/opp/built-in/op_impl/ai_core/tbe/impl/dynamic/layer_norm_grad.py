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
dynamic layer_norm_grad
"""
import tbe.common.register as tbe_register
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-many-arguments,unused-argument,unnecessary-pass
@tbe_register.register_param_generalization("LayerNormGrad")
def layer_norm_grad_generalization(input_dy,
                                   input_x,
                                   input_variance,
                                   input_mean,
                                   input_gamma,
                                   output_pd_x,
                                   output_pd_gamma,
                                   output_pd_beta,
                                   impl_mode=None,
                                   generalize_config=None):
    """
    for now only support dy (-1, -1, N)  and shape_gamma is (N,)
    """

    def filter_non_last_dim(filter_dict):
        if filter_dict is None:
            return filter_dict
        input_shape = list(filter_dict.get("ori_shape"))
        if not input_shape:
            return filter_dict
        input_last_dim = input_shape[-1]
        new_shape = [-1] * len(input_shape)
        new_shape[-1] = input_last_dim
        new_range = [(1, -1)] * len(input_shape)
        new_range[-1] = (input_last_dim, input_last_dim)
        filter_dict["ori_shape"] = tuple(new_shape)
        filter_dict["ori_range"] = tuple(new_range)

        return filter_dict

    result = []
    filter_dict_list = \
        [input_dy, input_x, input_variance, input_mean, input_gamma, output_pd_x, output_pd_gamma, output_pd_beta]
    result_list = [filter_non_last_dim(filter_dict) for filter_dict in filter_dict_list]

    result.append(result_list)
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def layer_norm_grad(input_dy,
                    input_x,
                    input_variance,
                    input_mean,
                    input_gamma,
                    output_pd_x,
                    output_pd_gamma,
                    output_pd_beta,
                    kernel_name="layer_norm_grad"):
    """
    algorithm: layernorm_grad
    calculating: gradient of layernorm
                 compute partial derivation of x, gamma and beta
        pd_xl    = data_dy*data_gamma
        pd_var   = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
                   *np.power((data_variance + EPSLON), (-1.5))),
                   reduce_axis, keepdims=True)
        pd_mean  = np.sum(((-1.0)*pd_xl
                   *np.power((data_variance + EPSLON), (-0.5))),
                   reduce_axis, keepdims=True)
                   + pd_var*(1.0/m)
                   *np.sum(((-2.0)*(data_x - data_mean)), reduce_axis,
                   keepdims=True)
        pd_x     = pd_xl*np.power((data_variance + EPSLON), (-0.5))
                   + pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
        pd_gamma = np.sum((data_dy*(data_x - data_mean)
                   *np.power((data_variance + EPSLON), (-0.5))), param_axis,
                   keepdims=True)
        pd_beta  = np.sum(data_dy, param_axis, keepdims=True)

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    output_pd_x: dict
        shape and dtype of output, only support float16, float32
    output_pd_gamma: dict
        shape and dtype of output, only support float16, float32
    output_pd_beta: dict
        shape and dtype of output, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_grad"

    Returns
    -------
    None
    """
    pass
