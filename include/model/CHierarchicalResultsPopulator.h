/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CHierarchicalResultsPopulator_h
#define INCLUDED_ml_model_CHierarchicalResultsPopulator_h

#include <model/CHierarchicalResults.h>
#include <model/ImportExport.h>

namespace ml {
namespace model {
class CLimits;

//! \brief FIXME
//!
//! DESCRIPTION:\n
//! FIXME
class MODEL_EXPORT CHierarchicalResultsPopulator : public CHierarchicalResultsVisitor {
public:
    //! Constructor
    CHierarchicalResultsPopulator(const CLimits& limits);

    //! Visit \p node.
    virtual void visit(const CHierarchicalResults& results, const TNode& node, bool pivot);

private:
    const CLimits& m_Limits;
};
}
}

#endif // INCLUDED_ml_model_CHierarchicalResultsPopulator_h
