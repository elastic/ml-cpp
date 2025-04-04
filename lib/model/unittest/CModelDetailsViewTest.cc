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

#include <core/CLogger.h>
#include <core/Constants.h>

#include <maths/common/CNormalMeanPrecConjugate.h>

#include <maths/time_series/CTimeSeriesDecomposition.h>
#include <maths/time_series/CTimeSeriesModel.h>

#include <model/CDataGatherer.h>
#include <model/CModelPlotData.h>
#include <model/CResourceMonitor.h>
#include <model/CSearchKey.h>

#include "Mocks.h"
#include "ModelTestHelpers.h"

#include <boost/test/unit_test.hpp>

#include <memory>
#include <ranges>
#include <vector>

BOOST_TEST_DONT_PRINT_LOG_VALUE(ml::model::CModelPlotData::TFeatureStrByFieldDataUMapUMapCItr);

BOOST_AUTO_TEST_SUITE(CModelDetailsViewTest)

using namespace ml;

class CTestFixture {
protected:
    model::CResourceMonitor m_ResourceMonitor;
};

BOOST_FIXTURE_TEST_CASE(testModelPlot, CTestFixture) {
    using TDoubleVec = std::vector<double>;
    using TMockModelPtr = std::unique_ptr<model::CMockModel>;

    constexpr core_t::TTime bucketLength{600};
    const model::CSearchKey key;
    const model::SModelParams params{bucketLength};
    model_t::TFeatureVec features;

    model::CAnomalyDetectorModel::TDataGathererPtr gatherer;
    TMockModelPtr model;

    auto setupTest = [&]() {
        gatherer = model::CDataGathererBuilder(model_t::analysisCategory(features[0]),
                                               features, params, key, 0)
                       .personFieldName("p")
                       .buildSharedPtr();
        for (const std::vector<std::string> persons{"p11", "p12", "p21", "p22"};
             const auto& person : persons) {
            bool addedPerson{false};
            gatherer->addPerson(person, m_ResourceMonitor, addedPerson);
        }

        const model::CMockModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;
        model = std::make_unique<model::CMockModel>(params, gatherer, influenceCalculators);

        const maths::time_series::CTimeSeriesDecomposition trend;
        const maths::common::CNormalMeanPrecConjugate prior{
            maths::common::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData)};
        const maths::common::CModelParams timeSeriesModelParams{
            bucketLength, 1.0, 0.001, 0.2, 6 * core::constants::HOUR, 24 * core::constants::HOUR};
        const maths::time_series::CUnivariateTimeSeriesModel timeSeriesModel{
            timeSeriesModelParams, 0, trend, prior};
        model::CMockModel::TMathsModelUPtrVec models;
        for (int i = 0; i < 4; ++i) {
            models.emplace_back(timeSeriesModel.clone(i));
        }
        model->mockTimeSeriesModels(std::move(models));
    };

    auto testModelPlot = [&](model_t::EFeature feature, const TDoubleVec& values) {
        features.assign(1, feature);
        setupTest();
        std::size_t pid{0};
        for (auto value : values) {
            model->mockAddBucketValue(feature, pid++, 0, 0, {value});
        }

        model::CModelPlotData plotData;
        model->details()->modelPlot(0, 90.0, {}, plotData);
        BOOST_TEST_REQUIRE(plotData.begin() != plotData.end());
        for (const auto & [ _, plotDataValues ] : plotData) {
            BOOST_REQUIRE_EQUAL(values.size(), plotDataValues.size());
            for (const auto & [ fst, snd ] : plotDataValues) {
                BOOST_TEST_REQUIRE(gatherer->personId(fst, pid));
                BOOST_REQUIRE_EQUAL(1, snd.s_ValuesPerOverField.size());
                for (const auto & [ field_name, val ] : snd.s_ValuesPerOverField) {
                    BOOST_REQUIRE_EQUAL(values[pid], val);
                }
            }
        }
    };

    LOG_DEBUG(<< "Individual sum");
    testModelPlot(model_t::E_IndividualSumByBucketAndPerson, {2.0, 3.0, 0.0, 0.0});

    LOG_DEBUG(<< "Individual count");
    testModelPlot(model_t::E_IndividualCountByBucketAndPerson, {0.0, 1.0, 3.0});
}

BOOST_AUTO_TEST_SUITE_END()
