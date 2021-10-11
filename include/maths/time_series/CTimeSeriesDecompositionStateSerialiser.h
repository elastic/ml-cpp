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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionStateSerialiser_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionStateSerialiser_h

#include <core/CNonInstantiatable.h>
#include <core/CoreTypes.h>

#include <maths/common/ImportExport.h>
#include <maths/common/MathsTypes.h>

#include <memory>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
namespace common {
struct STimeSeriesDecompositionRestoreParams;
}
namespace time_series {
class CTimeSeriesDecompositionInterface;

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
    using TDecompositionUPtr = std::unique_ptr<CTimeSeriesDecompositionInterface>;
    using TDecompositionSPtr = std::shared_ptr<CTimeSeriesDecompositionInterface>;

public:
    //! Construct the appropriate CTimeSeriesDecompositionInterface
    //! sub-class from its state document representation. Sets \p result
    //! to NULL on failure.
    bool operator()(const common::STimeSeriesDecompositionRestoreParams& params,
                    TDecompositionUPtr& result,
                    core::CStateRestoreTraverser& traverser) const;

    //! Construct the appropriate CTimeSeriesDecompositionInterface
    //! sub-class from its state document representation. Sets \p result
    //! to NULL on failure.
    bool operator()(const common::STimeSeriesDecompositionRestoreParams& params,
                    TDecompositionSPtr& result,
                    core::CStateRestoreTraverser& traverser) const;

    //! Persist state by passing information to the supplied inserter.
    void operator()(const CTimeSeriesDecompositionInterface& decomposition,
                    core::CStatePersistInserter& inserter) const;
};
}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionStateSerialiser_h
