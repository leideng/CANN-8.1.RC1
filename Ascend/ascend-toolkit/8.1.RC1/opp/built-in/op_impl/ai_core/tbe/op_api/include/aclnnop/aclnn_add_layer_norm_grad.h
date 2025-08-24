
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADD_LAYER_NORM_GRAD_H_
#define ACLNN_ADD_LAYER_NORM_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAddLayerNormGradGetWorkspaceSize
 * parameters :
 * dy : required
 * x1 : required
 * x2 : required
 * rstd : required
 * mean : required
 * gamma : required
 * dsumOptional : optional
 * dxOut : required
 * dgammaOut : required
 * dbetaOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddLayerNormGradGetWorkspaceSize(
    const aclTensor *dy,
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *rstd,
    const aclTensor *mean,
    const aclTensor *gamma,
    const aclTensor *dsumOptional,
    const aclTensor *dxOut,
    const aclTensor *dgammaOut,
    const aclTensor *dbetaOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAddLayerNormGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddLayerNormGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
