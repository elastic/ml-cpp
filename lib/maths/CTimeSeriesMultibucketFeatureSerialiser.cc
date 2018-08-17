/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
            result = boost::make_unique<CTimeSeriesMultibucketMean<double>>(),
            traverser.traverseSubLevel(boost::bind<bool>(
                &TUnivariateFeature::acceptRestoreTraverser, result.get(), _1)),
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
            result = boost::make_unique<CTimeSeriesMultibucketMean<TDouble10Vec>>(),
            traverser.traverseSubLevel(boost::bind<bool>(
                &TMultivariateFeature::acceptRestoreTraverser, result.get(), _1)),
            /**/)
    } while (traverser.next());
    return true;
}

void CTimeSeriesMultibucketFeatureSerialiser::
operator()(const TUnivariateFeaturePtr& feature, core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CTimeSeriesMultibucketMean<double>*>(feature.get()) != nullptr) {
        inserter.insertLevel(UNIVARIATE_MEAN_TAG, boost::bind(&TUnivariateFeature::acceptPersistInserter,
                                                              feature.get(), _1));
    } else {
        LOG_ERROR(<< "Feature with type '" << typeid(feature).name() << "' has no defined name");
    }
}

void CTimeSeriesMultibucketFeatureSerialiser::
operator()(const TMultivariateFeaturePtr& feature, core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CTimeSeriesMultibucketMean<TDouble10Vec>*>(feature.get()) != nullptr) {
        inserter.insertLevel(MULTIVARIATE_MEAN_TAG,
                             boost::bind(&TMultivariateFeature::acceptPersistInserter,
                                         feature.get(), _1));
    } else {
        LOG_ERROR(<< "Feature with type '" << typeid(feature).name() << "' has no defined name");
    }
}
}
}
