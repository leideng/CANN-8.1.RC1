
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWI_GLU_QUANT_H_
#define ACLNN_SWI_GLU_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwiGluQuantGetWorkspaceSize
 * parameters :
 * x : required
 * smoothScalesOptional : optional
 * offsetsOptional : optional
 * groupIndexOptional : optional
 * activateLeft : optional
 * quantModeOptional : optional
 * yOut : required
 * scaleOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwiGluQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *smoothScalesOptional,
    const aclTensor *offsetsOptional,
    const aclTensor *groupIndexOptional,
    bool activateLeft,
    char *quantModeOptional,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwiGluQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwiGluQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
