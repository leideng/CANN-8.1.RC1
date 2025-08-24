
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DEEP_NORM_GRAD_H_
#define ACLNN_DEEP_NORM_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDeepNormGradGetWorkspaceSize
 * parameters :
 * dy : required
 * x : required
 * gx : required
 * gamma : required
 * mean : required
 * rstd : required
 * alpha : optional
 * dxOut : required
 * dgxOut : required
 * dbetaOut : required
 * dgammaOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeepNormGradGetWorkspaceSize(
    const aclTensor *dy,
    const aclTensor *x,
    const aclTensor *gx,
    const aclTensor *gamma,
    const aclTensor *mean,
    const aclTensor *rstd,
    double alpha,
    const aclTensor *dxOut,
    const aclTensor *dgxOut,
    const aclTensor *dbetaOut,
    const aclTensor *dgammaOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDeepNormGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeepNormGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
