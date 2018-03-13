/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2017 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <maths/CModelStateSerialiser.h>

#include <maths/CTimeSeriesModel.h>

#include <boost/bind.hpp>

namespace ml {
namespace maths {
namespace {
const std::string UNIVARIATE_TIME_SERIES_TAG{"a"};
const std::string MULTIVARIATE_TIME_SERIES_TAG{"b"};
const std::string MODEL_STUB_TAG{"c"};
}

bool CModelStateSerialiser::operator()(const SModelRestoreParams &params,
                                       TModelPtr &result,
                                       core::CStateRestoreTraverser &traverser) const {
    std::size_t numResults = 0;

    do {
        const std::string &name = traverser.name();
        if (name == UNIVARIATE_TIME_SERIES_TAG) {
            result.reset(new CUnivariateTimeSeriesModel(params, traverser));
            ++numResults;
        } else if (name == MULTIVARIATE_TIME_SERIES_TAG) {
            result.reset(new CMultivariateTimeSeriesModel(params, traverser));
            ++numResults;
        } else if (name == MODEL_STUB_TAG) {
            result.reset(new CModelStub());
            ++numResults;
        } else {
            LOG_ERROR("No model corresponds to name " << traverser.name());
            return false;
        }
    } while (traverser.next());

    if (numResults != 1) {
        LOG_ERROR("Expected 1 (got " << numResults << ") model tags");
        result.reset();
        return false;
    }

    return true;
}

void CModelStateSerialiser::operator()(const CModel &model,
                                       core::CStatePersistInserter &inserter) const {
    if (dynamic_cast<const CUnivariateTimeSeriesModel *>(&model) != 0) {
        inserter.insertLevel(UNIVARIATE_TIME_SERIES_TAG,
                             boost::bind(&CModel::acceptPersistInserter, &model, _1));
    } else if (dynamic_cast<const CMultivariateTimeSeriesModel *>(&model) != 0) {
        inserter.insertLevel(MULTIVARIATE_TIME_SERIES_TAG,
                             boost::bind(&CModel::acceptPersistInserter, &model, _1));
    } else if (dynamic_cast<const CModelStub *>(&model) != 0) {
        inserter.insertValue(MODEL_STUB_TAG, "");
    } else {
        LOG_ERROR("Model with type '" << typeid(model).name() << "' has no defined name");
    }
}
}
}
