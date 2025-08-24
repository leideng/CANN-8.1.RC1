
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MASKED_SOFTMAX_WITH_REL_POS_BIAS_H_
#define ACLNN_MASKED_SOFTMAX_WITH_REL_POS_BIAS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize
 * parameters :
 * x : required
 * attenMaskOptional : optional
 * relativePosBias : required
 * scaleValue : optional
 * innerPrecisionMode : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *attenMaskOptional,
    const aclTensor *relativePosBias,
    double scaleValue,
    int64_t innerPrecisionMode,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMaskedSoftmaxWithRelPosBias
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMaskedSoftmaxWithRelPosBias(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
