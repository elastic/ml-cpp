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
#include <vector>
#include <ranges>

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
        std::string const person11{"p11"};
        std::string const person12{"p12"};
        std::string const person21{"p21"};
        std::string const person22{"p22"};
        bool addedPerson{false};
        gatherer->addPerson(person11, m_ResourceMonitor, addedPerson);
        gatherer->addPerson(person12, m_ResourceMonitor, addedPerson);
        gatherer->addPerson(person21, m_ResourceMonitor, addedPerson);
        gatherer->addPerson(person22, m_ResourceMonitor, addedPerson);

        model.reset(new model::CMockModel{params, gatherer, {}});

        maths::time_series::CTimeSeriesDecomposition const trend;
        maths::common::CNormalMeanPrecConjugate const prior{
            maths::common::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData)};
        maths::common::CModelParams const timeSeriesModelParams{
            bucketLength, 1.0, 0.001, 0.2, 6 * core::constants::HOUR, 24 * core::constants::HOUR};
        maths::time_series::CUnivariateTimeSeriesModel const timeSeriesModel{
            timeSeriesModelParams, 0, trend, prior};
        model::CMockModel::TMathsModelUPtrVec models;
        models.emplace_back(timeSeriesModel.clone(0));
        models.emplace_back(timeSeriesModel.clone(1));
        models.emplace_back(timeSeriesModel.clone(2));
        models.emplace_back(timeSeriesModel.clone(3));
        model->mockTimeSeriesModels(std::move(models));
    };

    LOG_DEBUG(<< "Individual sum");
    {
        features.assign(1, model_t::E_IndividualSumByBucketAndPerson);
        setupTest();

        TDoubleVec values{2.0, 3.0, 0.0, 0.0};
        std::size_t pid{0};
        for (auto value : values) {
            model->mockAddBucketValue(model_t::E_IndividualSumByBucketAndPerson,
                                      pid++, 0, 0, {value});
        }

        model::CModelPlotData plotData;
        model->details()->modelPlot(0, 90.0, {}, plotData);
        BOOST_TEST_REQUIRE(plotData.begin() != plotData.end());
        for (const auto& plotDataValues : plotData | std::views::values) {
            BOOST_REQUIRE_EQUAL(values.size(), plotDataValues.size());
            for (const auto& [fst, snd] : plotDataValues) {
                BOOST_TEST_REQUIRE(gatherer->personId(fst, pid));
                BOOST_REQUIRE_EQUAL(1, snd.s_ValuesPerOverField.size());
                for (const auto& val : snd.s_ValuesPerOverField |
                                           std::views::values) {
                    BOOST_REQUIRE_EQUAL(values[pid], val);
                }
            }
        }
    }

    LOG_DEBUG(<< "Individual count");
    {
        features.assign(1, model_t::E_IndividualCountByBucketAndPerson);
        setupTest();

        const TDoubleVec values{0.0, 1.0, 3.0};
        std::size_t pid{0};
        for (auto value : values) {
            model->mockAddBucketValue(model_t::E_IndividualCountByBucketAndPerson,
                                      pid++, 0, 0, {value});
        }

        model::CModelPlotData plotData;
        model->details()->modelPlot(0, 90.0, {}, plotData);
        BOOST_TEST_REQUIRE(plotData.begin() != plotData.end());
        for (const auto& plotDataValues : plotData | std::views::values) {
            BOOST_REQUIRE_EQUAL(values.size(), plotDataValues.size());
            for (const auto& [fst, snd] : plotDataValues) {
                BOOST_TEST_REQUIRE(gatherer->personId(fst, pid));
                BOOST_REQUIRE_EQUAL(1, snd.s_ValuesPerOverField.size());
                for (const auto& val : snd.s_ValuesPerOverField |
                                           std::views::values) {
                    BOOST_REQUIRE_EQUAL(values[pid], val);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
