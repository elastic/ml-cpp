/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CHierarchicalResultsPopulator.h>

#include <core/CLogger.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>
#include <model/CLimits.h>

namespace ml {
namespace model {

CHierarchicalResultsPopulator::CHierarchicalResultsPopulator(const CLimits& limits) : m_Limits(limits) {
}

void CHierarchicalResultsPopulator::visit(const CHierarchicalResults& results, const TNode& node, bool pivot) {
    if (!this->isLeaf(node) || !this->shouldWriteResult(m_Limits, results, node, pivot)) {
        return;
    }

    if (!node.s_Model) {
        LOG_ERROR(<< "No model for " << node.s_Spec.print());
        return;
    }

    const CDataGatherer& gatherer = node.s_Model->dataGatherer();

    std::size_t pid;
    if (!gatherer.personId(*node.s_Spec.s_PersonFieldValue, pid)) {
        LOG_ERROR(<< "No identifier for '" << *node.s_Spec.s_PersonFieldValue << "'");
        return;
    }

    SAnnotatedProbability& probability = node.s_AnnotatedProbability;
    for (std::size_t i = 0; i < probability.s_AttributeProbabilities.size(); ++i) {
        const SAttributeProbability& attribute = probability.s_AttributeProbabilities[i];
        attribute.s_CurrentBucketValue =
            node.s_Model->currentBucketValue(attribute.s_Feature, pid, attribute.s_Cid, node.s_BucketStartTime + node.s_BucketLength / 2);
        attribute.s_BaselineBucketMean = node.s_Model->baselineBucketMean(attribute.s_Feature,
                                                                          pid,
                                                                          attribute.s_Cid,
                                                                          attribute.s_Type,
                                                                          attribute.s_Correlated,
                                                                          node.s_BucketStartTime + node.s_BucketLength / 2);
    }

    probability.s_CurrentBucketCount = node.s_Model->currentBucketCount(pid, node.s_BucketStartTime);
    probability.s_BaselineBucketCount = node.s_Model->baselineBucketCount(pid);
}
}
}
