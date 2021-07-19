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

#include <maths/CTimeSeriesMultibucketFeatureSerialiser.h>

#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CTimeSeriesMultibucketFeatures.h>

namespace ml {
namespace maths {
namespace {
const std::string UNIVARIATE_MEAN_TAG{"a"};
const std::string MULTIVARIATE_MEAN_TAG{"b"};
}

bool CTimeSeriesMultibucketFeatureSerialiser::
operator()(TUnivariateFeaturePtr& result, core::CStateRestoreTraverser& traverser) const {
    do {
        const std::string& name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(
            UNIVARIATE_MEAN_TAG,
            result = std::make_unique<CTimeSeriesMultibucketMean<double>>(),
            traverser.traverseSubLevel(
                std::bind<bool>(&TUnivariateFeature::acceptRestoreTraverser,
                                result.get(), std::placeholders::_1)),
            /**/)
    } while (traverser.next());
    return true;
}

bool CTimeSeriesMultibucketFeatureSerialiser::
operator()(TMultivariateFeaturePtr& result, core::CStateRestoreTraverser& traverser) const {
    do {
        const std::string& name{traverser.name()};
        RESTORE_SETUP_TEARDOWN(
            MULTIVARIATE_MEAN_TAG,
            result = std::make_unique<CTimeSeriesMultibucketMean<TDouble10Vec>>(),
            traverser.traverseSubLevel(
                std::bind<bool>(&TMultivariateFeature::acceptRestoreTraverser,
                                result.get(), std::placeholders::_1)),
            /**/)
    } while (traverser.next());
    return true;
}

void CTimeSeriesMultibucketFeatureSerialiser::
operator()(const TUnivariateFeaturePtr& feature, core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CTimeSeriesMultibucketMean<double>*>(feature.get()) != nullptr) {
        inserter.insertLevel(UNIVARIATE_MEAN_TAG,
                             std::bind(&TUnivariateFeature::acceptPersistInserter,
                                       feature.get(), std::placeholders::_1));
    } else {
        LOG_ERROR(<< "Feature with type '" << typeid(feature).name() << "' has no defined name");
    }
}

void CTimeSeriesMultibucketFeatureSerialiser::
operator()(const TMultivariateFeaturePtr& feature, core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CTimeSeriesMultibucketMean<TDouble10Vec>*>(feature.get()) != nullptr) {
        inserter.insertLevel(MULTIVARIATE_MEAN_TAG,
                             std::bind(&TMultivariateFeature::acceptPersistInserter,
                                       feature.get(), std::placeholders::_1));
    } else {
        LOG_ERROR(<< "Feature with type '" << typeid(feature).name() << "' has no defined name");
    }
}
}
}
