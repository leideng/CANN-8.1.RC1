
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RMS_NORM_GRAD_H_
#define ACLNN_RMS_NORM_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRmsNormGradGetWorkspaceSize
 * parameters :
 * dy : required
 * x : required
 * rstd : required
 * gamma : required
 * dxOut : required
 * dgammaOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNormGradGetWorkspaceSize(
    const aclTensor *dy,
    const aclTensor *x,
    const aclTensor *rstd,
    const aclTensor *gamma,
    const aclTensor *dxOut,
    const aclTensor *dgammaOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRmsNormGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNormGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
