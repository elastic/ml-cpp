/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CModelStateSerialiser_h
#define INCLUDED_ml_maths_CModelStateSerialiser_h

#include <maths/ImportExport.h>

#include <boost/shared_ptr.hpp>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths
{
class CModel;
struct SModelRestoreParams;

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
class MATHS_EXPORT CModelStateSerialiser
{
    public:
        using TModelPtr = boost::shared_ptr<CModel>;

    public:
        //! Construct the appropriate CPrior sub-class from its state
        //! document representation. Sets \p result to NULL on failure.
        bool operator()(const SModelRestoreParams &params,
                        TModelPtr &result,
                        core::CStateRestoreTraverser &traverser) const;

        //! Persist state by passing information to the supplied inserter
        void operator()(const CModel &model, core::CStatePersistInserter &inserter) const;
};

}
}

#endif // INCLUDED_ml_maths_CModelStateSerialiser_h
