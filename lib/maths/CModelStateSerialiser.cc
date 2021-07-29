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

#include <maths/CModelStateSerialiser.h>

#include <maths/CTimeSeriesModel.h>

namespace ml {
namespace maths {
namespace {
const std::string UNIVARIATE_TIME_SERIES_TAG{"a"};
const std::string MULTIVARIATE_TIME_SERIES_TAG{"b"};
const std::string MODEL_STUB_TAG{"c"};
}

bool CModelStateSerialiser::operator()(const SModelRestoreParams& params,
                                       TModelPtr& result,
                                       core::CStateRestoreTraverser& traverser) const {
    std::size_t numResults = 0;

    do {
        const std::string& name = traverser.name();
        if (name == UNIVARIATE_TIME_SERIES_TAG) {
            result = std::make_unique<CUnivariateTimeSeriesModel>(params, traverser);
            ++numResults;
        } else if (name == MULTIVARIATE_TIME_SERIES_TAG) {
            result = std::make_unique<CMultivariateTimeSeriesModel>(params, traverser);
            ++numResults;
        } else if (name == MODEL_STUB_TAG) {
            result = std::make_unique<CModelStub>();
            ++numResults;
        } else {
            LOG_ERROR(<< "No model corresponds to name " << traverser.name());
            return false;
        }
    } while (traverser.next());

    if (numResults != 1) {
        LOG_ERROR(<< "Expected 1 (got " << numResults << ") model tags");
        result.reset();
        return false;
    }

    return true;
}

void CModelStateSerialiser::operator()(const CModel& model,
                                       core::CStatePersistInserter& inserter) const {
    if (dynamic_cast<const CUnivariateTimeSeriesModel*>(&model) != nullptr) {
        inserter.insertLevel(UNIVARIATE_TIME_SERIES_TAG,
                             std::bind(&CModel::acceptPersistInserter, &model,
                                       std::placeholders::_1));
    } else if (dynamic_cast<const CMultivariateTimeSeriesModel*>(&model) != nullptr) {
        inserter.insertLevel(MULTIVARIATE_TIME_SERIES_TAG,
                             std::bind(&CModel::acceptPersistInserter, &model,
                                       std::placeholders::_1));
    } else if (dynamic_cast<const CModelStub*>(&model) != nullptr) {
        inserter.insertValue(MODEL_STUB_TAG, "");
    } else {
        LOG_ERROR(<< "Model with type '" << typeid(model).name() << "' has no defined name");
    }
}
}
}
