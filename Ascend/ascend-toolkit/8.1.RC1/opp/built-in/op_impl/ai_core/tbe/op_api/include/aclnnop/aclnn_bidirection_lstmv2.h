
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BIDIRECTION_LSTMV2_H_
#define ACLNN_BIDIRECTION_LSTMV2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBidirectionLSTMV2GetWorkspaceSize
 * parameters :
 * x : required
 * initH : required
 * initC : required
 * wIh : required
 * wHh : required
 * bIhOptional : optional
 * bHhOptional : optional
 * wIhReverseOptional : optional
 * wHhReverseOptional : optional
 * bIhReverseOptional : optional
 * bHhReverseOptional : optional
 * batchSizeOptional : optional
 * numLayers : optional
 * isbias : optional
 * batchFirst : optional
 * bidirection : optional
 * packed : optional
 * yOut : required
 * outputHOut : required
 * outputCOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBidirectionLSTMV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *initH,
    const aclTensor *initC,
    const aclTensor *wIh,
    const aclTensor *wHh,
    const aclTensor *bIhOptional,
    const aclTensor *bHhOptional,
    const aclTensor *wIhReverseOptional,
    const aclTensor *wHhReverseOptional,
    const aclTensor *bIhReverseOptional,
    const aclTensor *bHhReverseOptional,
    const aclTensor *batchSizeOptional,
    int64_t numLayers,
    bool isbias,
    bool batchFirst,
    bool bidirection,
    bool packed,
    const aclTensor *yOut,
    const aclTensor *outputHOut,
    const aclTensor *outputCOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBidirectionLSTMV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBidirectionLSTMV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
