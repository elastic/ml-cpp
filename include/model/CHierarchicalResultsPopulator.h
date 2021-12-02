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

    virtual ~CHierarchicalResultsPopulator() override = default;

    //! Visit \p node.
    void visit(const CHierarchicalResults& results, const TNode& node, bool pivot) override;

private:
    const CLimits& m_Limits;
};
}
}

#endif // INCLUDED_ml_model_CHierarchicalResultsPopulator_h
