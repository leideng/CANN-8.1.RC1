
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_INIT_ROUTING_QUANT_V2_H_
#define ACLNN_MOE_INIT_ROUTING_QUANT_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeInitRoutingQuantV2GetWorkspaceSize
 * parameters :
 * x : required
 * expertIdx : required
 * scaleOptional : optional
 * offsetOptional : optional
 * activeNum : optional
 * expertCapacity : optional
 * expertNum : optional
 * dropPadMode : optional
 * expertTokensCountOrCumsumFlag : optional
 * expertTokensBeforeCapacityFlag : optional
 * quantMode : optional
 * expandedXOut : required
 * expandedRowIdxOut : required
 * expertTokensCountOrCumsumOutOptional : optional
 * expertTokensBeforeCapacityOutOptional : optional
 * dynamicQuantScaleOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeInitRoutingQuantV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIdx,
    const aclTensor *scaleOptional,
    const aclTensor *offsetOptional,
    int64_t activeNum,
    int64_t expertCapacity,
    int64_t expertNum,
    int64_t dropPadMode,
    int64_t expertTokensCountOrCumsumFlag,
    bool expertTokensBeforeCapacityFlag,
    int64_t quantMode,
    const aclTensor *expandedXOut,
    const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOutOptional,
    const aclTensor *expertTokensBeforeCapacityOutOptional,
    const aclTensor *dynamicQuantScaleOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeInitRoutingQuantV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeInitRoutingQuantV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
