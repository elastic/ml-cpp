/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <model/CHierarchicalResultsProbabilityFinalizer.h>

#include <maths/CTools.h>

namespace ml {
namespace model {

void CHierarchicalResultsProbabilityFinalizer::visit(const CHierarchicalResults& /*results*/,
                                                     const TNode& node,
                                                     bool /*pivot*/) {
    if (node.s_RawAnomalyScore > 0.0) {
        node.s_AnnotatedProbability.s_Probability =
            maths::CTools::inverseAnomalyScore(node.s_RawAnomalyScore);
    }
}
}
}
