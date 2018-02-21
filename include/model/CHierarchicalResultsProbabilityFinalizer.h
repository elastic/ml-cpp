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

#ifndef INCLUDED_ml_model_CHierarchicalResultsProbabilityFinalizer_h
#define INCLUDED_ml_model_CHierarchicalResultsProbabilityFinalizer_h

#include <model/CHierarchicalResults.h>
#include <model/ModelTypes.h>

namespace ml
{
namespace model
{

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
class MODEL_EXPORT CHierarchicalResultsProbabilityFinalizer : public CHierarchicalResultsVisitor
{
    public:
        //! Finalize the probability of \p node.
        virtual void visit(const CHierarchicalResults &results, const TNode &node, bool pivot);
};

}
}

#endif // INCLUDED_ml_model_CHierarchicalResultsProbabilityFinalizer_h
