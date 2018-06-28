/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CPriorStateSerialiser_h
#define INCLUDED_ml_maths_CPriorStateSerialiser_h

#include <core/CNonInstantiatable.h>

#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <memory>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
class CMultivariatePrior;
class CPrior;
struct SDistributionRestoreParams;

//! \brief Convert CPrior sub-classes to/from text representations.
//!
//! DESCRIPTION:\n
//! Encapsulate the conversion of arbitrary CPrior sub-classes to/from
//! textual state.  In particular, the field name associated with each prior
//! distribution is then in one file.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The serialisation format must be a hierarchical format that supports
//! name/value pairs where the value may be a nested set of name/value
//! pairs. Text format makes it easier to provide backwards/forwards
//! compatibility in the future as the classes evolve.
class MATHS_EXPORT CPriorStateSerialiser {
public:
    using TPriorUPtr = std::unique_ptr<CPrior>;
    using TPriorSPtr = std::shared_ptr<CPrior>;
    using TMultivariatePriorPtr = std::unique_ptr<CMultivariatePrior>;

public:
    //! Construct the appropriate CPrior sub-class from its state
    //! document representation.  Sets \p ptr to NULL on failure.
    bool operator()(const SDistributionRestoreParams& params,
                    TPriorUPtr& ptr,
                    core::CStateRestoreTraverser& traverser) const;

    //! Construct the appropriate CPrior sub-class from its state
    //! document representation.  Sets \p ptr to NULL on failure.
    bool operator()(const SDistributionRestoreParams& params,
                    TPriorSPtr& ptr,
                    core::CStateRestoreTraverser& traverser) const;

    //! Persist state by passing information to the supplied inserter
    void operator()(const CPrior& prior, core::CStatePersistInserter& inserter) const;

    //! Construct the appropriate CMultivariatePrior sub-class from
    //! its state document representation.  Sets \p ptr to NULL on
    //! failure.
    bool operator()(const SDistributionRestoreParams& params,
                    TMultivariatePriorPtr& ptr,
                    core::CStateRestoreTraverser& traverser) const;

    //! Persist state by passing information to the supplied inserter
    void operator()(const CMultivariatePrior& prior,
                    core::CStatePersistInserter& inserter) const;
};
}
}

#endif // INCLUDED_ml_maths_CPriorStateSerialiser_h
