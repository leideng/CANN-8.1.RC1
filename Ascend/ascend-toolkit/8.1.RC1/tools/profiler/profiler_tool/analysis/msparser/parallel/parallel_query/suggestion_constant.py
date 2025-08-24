#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.


class SuggestionConstant:
    SUGGESTIONS = {
        "data-parallel": {
            "without_gradient_segmentation":
                "Gradient segmentation is not performed on the model. You can apply a proper gradient segmentation "
                "policy to improve the parallelism degree of computation and AllReduce operators, "
                "thereby shortening the step tail time. For details, see the all_reduce_fusion_config "
                "parameter settings in the MindSpore distributed parallelism tutorial.",
            "bad_gradient_segmentation":
                "Gradient segmentation of the model has not achieved the optimal effect. In ideal situations{}. "
                "You can adjust the gradient segmentation policy to improve the parallelism degree of computation "
                "and AllReduce operators. For details, see the all_reduce_fusion_config parameter "
                "settings in the MindSpore distributed parallelism tutorial.",
            "optimal_gradient_segmentation":
                "The model has achieved an ideal gradient segmentation effect."},
        "model-parallel": {
            "bad_operator_tiling_with_auto":
                "Operator tiling of the model has not achieved the optimal effect. In ideal situations, the "
                "proportion of the pure communication time should be less than 10% (current value: {}). "
                "You can enable the AllToAll communication operator and use a different policy search algorithm. "
                "For details, see the enable_alltoall and search_mode parameter settings in the MindSpore "
                "distributed parallelism tutorial. If the effect is still unsatisfactory, you can manually tile "
                "the operators to improve the parallelism degree of computation and communication operators.",
            "bad_operator_tiling_with_manual":
                "Operator tiling of the model has not achieved the optimal effect. In ideal situations, the "
                "proportion of the pure communication time should be less than 10% (current value: {}). "
                "You can adjust the operator tiling policy and enable the AllToAll communication operator to "
                "improve the parallelism degree of computation and communication operators. For details, "
                "see the enable_alltoall parameter settings in the MindSpore distributed parallelism tutorial.",
            "optimal_operator_tiling":
                "The model has achieved an ideal operator tiling effect."},
        "pipeline-parallel": {
            "bad_operator_tiling":
                "Operator tiling of the model has not achieved the optimal effect. In ideal situations, "
                "the proportion of the pure communication time (receive op not contained) should be less than "
                "10% (current value: {}). You can adjust the operator tiling policy and enable the AllToAll "
                "communication operator to improve the parallelism degree of computation and communication "
                "operators. For details, see the enable_alltoall parameter settings in the "
                "MindSpore distributed parallelism tutorial.",
            "bad_stage_division":
                "Stage division of the model has not achieved the optimal effect. In ideal situations{}. "
                "You can adjust the stage division policy to balance the computing workload among stages and "
                "improve the parallelism degree of computation and communication operators. For details, see the "
                "pipeline_stages, pipeline_stage, and micro_size parameter settings in the "
                "MindSpore distributed parallelism tutorial.",
            "optimal_stage_division":
                "The model has achieved ideal operator tiling and stage division effects."}
    }
