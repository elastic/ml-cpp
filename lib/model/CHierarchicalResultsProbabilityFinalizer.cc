/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <model/CHierarchicalResultsProbabilityFinalizer.h>

#include <maths/CTools.h>

namespace ml {
namespace model {

void CHierarchicalResultsProbabilityFinalizer::visit(const CHierarchicalResults& /*results*/,
                                                     const TNode& node,
                                                     bool /*pivot*/) {
    if (node.s_RawAnomalyScore > 0.0) {
        node.s_AnnotatedProbability.s_Probability = maths::CTools::inverseDeviation(node.s_RawAnomalyScore);
    }
}
}
}
