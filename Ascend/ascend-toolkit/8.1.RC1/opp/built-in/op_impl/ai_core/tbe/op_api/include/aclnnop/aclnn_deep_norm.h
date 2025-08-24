
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DEEP_NORM_H_
#define ACLNN_DEEP_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDeepNormGetWorkspaceSize
 * parameters :
 * x : required
 * gx : required
 * beta : required
 * gamma : required
 * alpha : optional
 * epsilon : optional
 * meanOut : required
 * rstdOut : required
 * yOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeepNormGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gx,
    const aclTensor *beta,
    const aclTensor *gamma,
    double alpha,
    double epsilon,
    const aclTensor *meanOut,
    const aclTensor *rstdOut,
    const aclTensor *yOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDeepNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDeepNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
