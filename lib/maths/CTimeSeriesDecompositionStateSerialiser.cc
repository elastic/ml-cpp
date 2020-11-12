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

#include <memory>
#include <string>
#include <typeinfo>

namespace ml {
namespace maths {

namespace {

// We use short field names to reduce the state size
// There needs to be one constant here per sub-class
// of CTimeSeriesDecompositionInterface.
// DO NOT change the existing tags if new sub-classes are added.
const core::TPersistenceTag TIME_SERIES_DECOMPOSITION_TAG("a", "time_series_decomposition");
const core::TPersistenceTag TIME_SERIES_DECOMPOSITION_STUB_TAG("b", "time_series_decomposition_stub");
const std::string EMPTY_STRING;

//! Implements restore for std::shared_ptr.
template<typename T>
void doRestore(std::shared_ptr<CTimeSeriesDecompositionInterface>& ptr) {
    ptr = std::make_shared<T>();
}

//! Implements restore for std::unique_ptr.
template<typename T>
void doRestore(std::unique_ptr<CTimeSeriesDecompositionInterface>& ptr) {
    ptr = std::make_unique<T>();
}

//! Implements restore for std::shared_ptr.
template<typename T>
void doRestore(const STimeSeriesDecompositionRestoreParams& params,
               std::shared_ptr<CTimeSeriesDecompositionInterface>& ptr,
               core::CStateRestoreTraverser& traverser) {
    ptr = std::make_shared<T>(params, traverser);
}

//! Implements restore for std::unique_ptr.
template<typename T>
void doRestore(const STimeSeriesDecompositionRestoreParams& params,
               std::unique_ptr<CTimeSeriesDecompositionInterface>& ptr,
               core::CStateRestoreTraverser& traverser) {
    ptr = std::make_unique<T>(params, traverser);
}

//! Implements restore into the supplied pointer.
template<typename PTR>
bool restore(const STimeSeriesDecompositionRestoreParams& params,
             PTR& ptr,
             core::CStateRestoreTraverser& traverser) {
    std::size_t numResults{0};
    do {
        const std::string& name = traverser.name();
        if (name == TIME_SERIES_DECOMPOSITION_TAG) {
            doRestore<CTimeSeriesDecomposition>(params, ptr, traverser);
            ++numResults;
        } else if (name == TIME_SERIES_DECOMPOSITION_STUB_TAG) {
            doRestore<CTimeSeriesDecompositionStub>(ptr);
            ++numResults;
        } else {
            LOG_ERROR(<< "No decomposition corresponds to name " << traverser.name());
            return false;
        }
    } while (traverser.next());

    if (numResults != 1) {
        LOG_ERROR(<< "Expected 1 (got " << numResults << ") decomposition tags");
        ptr.reset();
        return false;
    }

    return true;
}
}

bool CTimeSeriesDecompositionStateSerialiser::
operator()(const STimeSeriesDecompositionRestoreParams& params,
           TDecompositionUPtr& ptr,
           core::CStateRestoreTraverser& traverser) const {
    return restore(params, ptr, traverser);
}

bool CTimeSeriesDecompositionStateSerialiser::
operator()(const STimeSeriesDecompositionRestoreParams& params,
           TDecompositionSPtr& ptr,
           core::CStateRestoreTraverser& traverser) const {
    return restore(params, ptr, traverser);
}

void CTimeSeriesDecompositionStateSerialiser::
operator()(const CTimeSeriesDecompositionInterface& decomposition,
           core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CTimeSeriesDecomposition*>(&decomposition) != nullptr) {
        inserter.insertLevel(
            TIME_SERIES_DECOMPOSITION_TAG,
            std::bind(&CTimeSeriesDecomposition::acceptPersistInserter,
                      dynamic_cast<const CTimeSeriesDecomposition*>(&decomposition),
                      std::placeholders::_1));
    } else if (dynamic_cast<const CTimeSeriesDecompositionStub*>(&decomposition) != nullptr) {
        inserter.insertValue(TIME_SERIES_DECOMPOSITION_STUB_TAG, "");
    } else {
        LOG_ERROR(<< "Decomposition with type '" << typeid(decomposition).name()
                  << "' has no defined name");
    }
}
}
}
