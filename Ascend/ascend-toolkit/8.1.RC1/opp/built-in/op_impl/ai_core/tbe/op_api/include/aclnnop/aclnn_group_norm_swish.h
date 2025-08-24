
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GROUP_NORM_SWISH_H_
#define ACLNN_GROUP_NORM_SWISH_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGroupNormSwishGetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * beta : required
 * numGroups : required
 * dataFormatOptional : optional
 * eps : optional
 * activateSwish : optional
 * swishScale : optional
 * yOut : required
 * meanOut : required
 * rstdOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGroupNormSwishGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *beta,
    int64_t numGroups,
    char *dataFormatOptional,
    double eps,
    bool activateSwish,
    double swishScale,
    const aclTensor *yOut,
    const aclTensor *meanOut,
    const aclTensor *rstdOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGroupNormSwish
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGroupNormSwish(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
