
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GROUP_NORM_SWISH_GRAD_H_
#define ACLNN_GROUP_NORM_SWISH_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGroupNormSwishGradGetWorkspaceSize
 * parameters :
 * dy : required
 * mean : required
 * rstd : required
 * x : required
 * gamma : required
 * beta : required
 * numGroups : required
 * dataFormatOptional : optional
 * swishScale : optional
 * dgammaIsRequire : optional
 * dbetaIsRequire : optional
 * dxOut : required
 * dgammaOut : required
 * dbetaOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGroupNormSwishGradGetWorkspaceSize(
    const aclTensor *dy,
    const aclTensor *mean,
    const aclTensor *rstd,
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *beta,
    int64_t numGroups,
    char *dataFormatOptional,
    double swishScale,
    bool dgammaIsRequire,
    bool dbetaIsRequire,
    const aclTensor *dxOut,
    const aclTensor *dgammaOut,
    const aclTensor *dbetaOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGroupNormSwishGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGroupNormSwishGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
