
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BACKGROUND_REPLACE_H_
#define ACLNN_BACKGROUND_REPLACE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBackgroundReplaceGetWorkspaceSize
 * parameters :
 * bkg : required
 * src : required
 * mask : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBackgroundReplaceGetWorkspaceSize(
    const aclTensor *bkg,
    const aclTensor *src,
    const aclTensor *mask,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBackgroundReplace
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBackgroundReplace(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
