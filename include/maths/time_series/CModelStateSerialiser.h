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

#ifndef INCLUDED_ml_maths_time_series_CModelStateSerialiser_h
#define INCLUDED_ml_maths_time_series_CModelStateSerialiser_h

#include <maths/time_series/ImportExport.h>

#include <memory>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
namespace common {
class CModel;
struct SModelRestoreParams;
}
namespace time_series {

//! \brief Convert CModel sub-classes to/from text representations.
//!
//! DESCRIPTION:\n
//! Encapsulate the conversion of arbitrary CModel sub-classes to/from
//! textual state.  In particular, the field name associated with each prior
//! distribution is then in one file.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The serialisation format must be a hierarchical format that supports
//! name/value pairs where the value may be a nested set of name/value
//! pairs. Text format is used to make it easier to provide backwards
//! compatibility in the future as the classes evolve.
class MATHS_TIME_SERIES_EXPORT CModelStateSerialiser {
public:
    using TModelPtr = std::unique_ptr<common::CModel>;

public:
    //! Construct the appropriate CPrior sub-class from its state
    //! document representation. Sets \p result to NULL on failure.
    bool operator()(const common::SModelRestoreParams& params,
                    TModelPtr& result,
                    core::CStateRestoreTraverser& traverser) const;

    //! Persist state by passing information to the supplied inserter
    void operator()(const common::CModel& model, core::CStatePersistInserter& inserter) const;
};
}
}
}

#endif // INCLUDED_ml_maths_time_series_CModelStateSerialiser_h
