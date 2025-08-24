
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADVANCE_STEP_H_
#define ACLNN_ADVANCE_STEP_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAdvanceStepGetWorkspaceSize
 * parameters :
 * inputTokens : required
 * sampledTokenIds : required
 * inputPositions : required
 * seqLens : required
 * slotMapping : required
 * blockTables : required
 * numSeqs : required
 * numQueries : required
 * blockSize : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAdvanceStepGetWorkspaceSize(
    const aclTensor *inputTokens,
    const aclTensor *sampledTokenIds,
    const aclTensor *inputPositions,
    const aclTensor *seqLens,
    const aclTensor *slotMapping,
    const aclTensor *blockTables,
    int64_t numSeqs,
    int64_t numQueries,
    int64_t blockSize,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAdvanceStep
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAdvanceStep(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
