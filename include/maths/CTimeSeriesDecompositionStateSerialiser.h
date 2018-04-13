/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CTimeSeriesDecompositionStateSerialiser_h
#define INCLUDED_ml_maths_CTimeSeriesDecompositionStateSerialiser_h

#include <core/CNonInstantiatable.h>
#include <core/CoreTypes.h>

#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/shared_ptr.hpp>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
class CTimeSeriesDecompositionInterface;
struct STimeSeriesDecompositionRestoreParams;

//! \brief Convert CTimeSeriesDecompositionInterface sub-classes to/from
//! text representations.
//!
//! DESCRIPTION:\n
//! Encapsulate the conversion of arbitrary CTimeSeriesDecompositionInterface
//! sub-classes to/from textual state. In particular, the field name associated
//! with each type of decomposition is then in one file.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The serialisation format must be a hierarchical format that supports
//! name/value pairs where the value may be a nested set of name/value
//! pairs. Text format makes it easier to provide backwards/forwards
//! compatibility in the future as the classes evolve.
class MATHS_EXPORT CTimeSeriesDecompositionStateSerialiser {
public:
    //! Shared pointer to the CTimeSeriesDecompositionInterface abstract
    //! base class.
    using TDecompositionPtr = boost::shared_ptr<CTimeSeriesDecompositionInterface>;

public:
    //! Construct the appropriate CTimeSeriesDecompositionInterface
    //! sub-class from its state document representation. Sets \p result
    //! to NULL on failure.
    bool operator()(const STimeSeriesDecompositionRestoreParams& params,
                    TDecompositionPtr& result,
                    core::CStateRestoreTraverser& traverser) const;

    //! Persist state by passing information to the supplied inserter.
    void operator()(const CTimeSeriesDecompositionInterface& decomposition, core::CStatePersistInserter& inserter) const;
};
}
}

#endif // INCLUDED_ml_maths_CTimeSeriesDecompositionStateSerialiser_h
