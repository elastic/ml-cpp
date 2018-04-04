/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CHierarchicalResultsProbabilityFinalizer.h>

#include <maths/CTools.h>

namespace ml {
namespace model {

void CHierarchicalResultsProbabilityFinalizer::visit(const CHierarchicalResults& /*results*/, const TNode& node, bool /*pivot*/) {
    if (node.s_RawAnomalyScore > 0.0) {
        node.s_AnnotatedProbability.s_Probability = maths::CTools::inverseDeviation(node.s_RawAnomalyScore);
    }
}
}
}
