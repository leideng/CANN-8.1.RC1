
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DEQUANT_BIAS_H_
#define ACLNN_DEQUANT_BIAS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDequantBiasGetWorkspaceSize
 * parameters :
 * x : required
 * weightScale : required
 * activateScaleOptional : optional
 * biasOptional : optional
 * outputDtype : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDequantBiasGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightScale,
    const aclTensor *activateScaleOptional,
    const aclTensor *biasOptional,
    int64_t outputDtype,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDequantBias
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDequantBias(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
