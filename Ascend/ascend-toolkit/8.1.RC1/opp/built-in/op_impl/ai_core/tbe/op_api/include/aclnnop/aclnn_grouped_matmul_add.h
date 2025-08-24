
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GROUPED_MATMUL_ADD_H_
#define ACLNN_GROUPED_MATMUL_ADD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGroupedMatmulAddGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * groupList : required
 * yRef : required
 * transposeX : optional
 * transposeWeight : optional
 * groupType : optional
 * yRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *groupList,
    aclTensor *yRef,
    bool transposeX,
    bool transposeWeight,
    int64_t groupType,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGroupedMatmulAdd
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGroupedMatmulAdd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
