
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADD_LAYER_NORM_H_
#define ACLNN_ADD_LAYER_NORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAddLayerNormGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * gamma : required
 * beta : required
 * biasOptional : optional
 * epsilon : optional
 * additionalOutput : optional
 * yOut : required
 * meanOut : required
 * rstdOut : required
 * xOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddLayerNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    const aclTensor *beta,
    const aclTensor *biasOptional,
    double epsilon,
    bool additionalOutput,
    const aclTensor *yOut,
    const aclTensor *meanOut,
    const aclTensor *rstdOut,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAddLayerNorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddLayerNorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
