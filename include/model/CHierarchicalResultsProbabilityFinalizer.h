/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CHierarchicalResultsProbabilityFinalizer_h
#define INCLUDED_ml_model_CHierarchicalResultsProbabilityFinalizer_h

#include <model/CHierarchicalResults.h>
#include <model/ModelTypes.h>

namespace ml {
namespace model {

//! \brief Ensures that all probabilities are equal to the inverse
//! deviation of the anomaly scores.
//!
//! DESCRIPTION:\n
//! Because of our handling of the extended probability range for
//! joint probabilities it is not necessarily the case that the
//! node probabilities are the inverse of their anomaly scores
//! after aggregation. However, it is important that this is true
//! when we write them out for normalization to work as expected.
//! This visitor ensures this invariant holds in a bottom up
//! breadth first pass over the results.
class MODEL_EXPORT CHierarchicalResultsProbabilityFinalizer : public CHierarchicalResultsVisitor {
public:
    //! Finalize the probability of \p node.
    virtual void visit(const CHierarchicalResults& results, const TNode& node, bool pivot);
};
}
}

#endif // INCLUDED_ml_model_CHierarchicalResultsProbabilityFinalizer_h
