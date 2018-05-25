/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesDecompositionStateSerialiser.h>

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CRestoreParams.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesDecompositionStub.h>

#include <boost/bind.hpp>

#include <string>
#include <typeinfo>

namespace ml {
namespace maths {

namespace {

// We use short field names to reduce the state size
// There needs to be one constant here per sub-class
// of CTimeSeriesDecompositionInterface.
// DO NOT change the existing tags if new sub-classes are added.
const std::string TIME_SERIES_DECOMPOSITION_TAG("a");
const std::string TIME_SERIES_DECOMPOSITION_STUB_TAG("b");
const std::string EMPTY_STRING;

//! Implements restore into the supplied pointer.
template<typename PTR>
bool restore(const STimeSeriesDecompositionRestoreParams& params,
             PTR& result,
             core::CStateRestoreTraverser& traverser) {
    std::size_t numResults{0};
    do {
        const std::string& name = traverser.name();
        if (name == TIME_SERIES_DECOMPOSITION_TAG) {
            result.reset(new CTimeSeriesDecomposition(params, traverser));
            ++numResults;
        } else if (name == TIME_SERIES_DECOMPOSITION_STUB_TAG) {
            result.reset(new CTimeSeriesDecompositionStub());
            ++numResults;
        } else {
            LOG_ERROR(<< "No decomposition corresponds to name " << traverser.name());
            return false;
        }
    } while (traverser.next());

    if (numResults != 1) {
        LOG_ERROR(<< "Expected 1 (got " << numResults << ") decomposition tags");
        result.reset();
        return false;
    }

    return true;
}
}

bool CTimeSeriesDecompositionStateSerialiser::
operator()(const STimeSeriesDecompositionRestoreParams& params,
           TDecompositionUPtr& result,
           core::CStateRestoreTraverser& traverser) const {
    return restore(params, result, traverser);
}

bool CTimeSeriesDecompositionStateSerialiser::
operator()(const STimeSeriesDecompositionRestoreParams& params,
           TDecompositionSPtr& result,
           core::CStateRestoreTraverser& traverser) const {
    return restore(params, result, traverser);
}

void CTimeSeriesDecompositionStateSerialiser::
operator()(const CTimeSeriesDecompositionInterface& decomposition,
           core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CTimeSeriesDecomposition*>(&decomposition) != nullptr) {
        inserter.insertLevel(
            TIME_SERIES_DECOMPOSITION_TAG,
            boost::bind(&CTimeSeriesDecomposition::acceptPersistInserter,
                        dynamic_cast<const CTimeSeriesDecomposition*>(&decomposition), _1));
    } else if (dynamic_cast<const CTimeSeriesDecompositionStub*>(&decomposition) != nullptr) {
        inserter.insertValue(TIME_SERIES_DECOMPOSITION_STUB_TAG, "");
    } else {
        LOG_ERROR(<< "Decomposition with type '" << typeid(decomposition).name()
                  << "' has no defined name");
    }
}
}
}
