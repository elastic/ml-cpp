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

#include <maths/time_series/CTimeSeriesMultibucketFeatureSerialiser.h>

#include <core/CMemory.h>
#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/time_series/CTimeSeriesMultibucketFeatures.h>

namespace ml {
namespace maths {
namespace time_series {
namespace {
const std::string UNIVARIATE_MEAN_TAG{"a"};
const std::string MULTIVARIATE_MEAN_TAG{"b"};
}

bool CTimeSeriesMultibucketFeatureSerialiser::
operator()(TScalarFeaturePtr& result, core::CStateRestoreTraverser& traverser) const {
    do {
        const std::string& name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(
            UNIVARIATE_MEAN_TAG,
            result = std::make_unique<CTimeSeriesMultibucketScalarMean>(),
            traverser.traverseSubLevel([&](core::CStateRestoreTraverser& traverser_) {
                return result->acceptRestoreTraverser(traverser_);
            }),
            /**/)
    } while (traverser.next());
    return true;
}

bool CTimeSeriesMultibucketFeatureSerialiser::
operator()(TVectorFeaturePtr& result, core::CStateRestoreTraverser& traverser) const {
    do {
        const std::string& name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(
            MULTIVARIATE_MEAN_TAG,
            result = std::make_unique<CTimeSeriesMultibucketVectorMean>(),
            traverser.traverseSubLevel([&](core::CStateRestoreTraverser& traverser_) {
                return result->acceptRestoreTraverser(traverser_);
            }),
            /**/)
    } while (traverser.next());
    return true;
}

void CTimeSeriesMultibucketFeatureSerialiser::
operator()(const TScalarFeaturePtr& feature, core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CTimeSeriesMultibucketScalarMean*>(feature.get()) != nullptr) {
        inserter.insertLevel(UNIVARIATE_MEAN_TAG, [&](core::CStatePersistInserter& inserter_) {
            feature->acceptPersistInserter(inserter_);
        });
    } else {
        LOG_ERROR(<< "Unknown feature with type '" << typeid(feature).name());
    }
}

void CTimeSeriesMultibucketFeatureSerialiser::
operator()(const TVectorFeaturePtr& feature, core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CTimeSeriesMultibucketVectorMean*>(feature.get()) != nullptr) {
        inserter.insertLevel(MULTIVARIATE_MEAN_TAG, [&](core::CStatePersistInserter& inserter_) {
            feature->acceptPersistInserter(inserter_);
        });
    } else {
        LOG_ERROR(<< "Unknown feature with type '" << typeid(feature).name());
    }
}
}
}
}
