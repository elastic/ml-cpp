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

#include "CModelDetailsViewTest.h"

#include <core/CLogger.h>

#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesModel.h>

#include <model/CDataGatherer.h>
#include <model/CModelPlotData.h>
#include <model/CSearchKey.h>

#include "Mocks.h"

#include <boost/scoped_ptr.hpp>

#include <vector>

using namespace ml;

namespace {

const std::string EMPTY_STRING;

}// unnamed

void CModelDetailsViewTest::testModelPlot() {
    LOG_DEBUG("*** CModelDetailsViewTest::testModelPlot ***");

    using TDoubleVec = std::vector<double>;
    using TStrVec = std::vector<std::string>;
    using TMockModelPtr = boost::scoped_ptr<model::CMockModel>;

    core_t::TTime bucketLength{600};
    model::CSearchKey key;
    model::SModelParams params{bucketLength};
    model_t::TFeatureVec features;

    model::CAnomalyDetectorModel::TDataGathererPtr gatherer;
    TMockModelPtr model;

    auto setupTest = [&]() {
        gatherer.reset(new model::CDataGatherer{model_t::analysisCategory(features[0]),
                                                model_t::E_None,
                                                params,
                                                EMPTY_STRING,
                                                EMPTY_STRING,
                                                EMPTY_STRING,
                                                "p",
                                                EMPTY_STRING,
                                                EMPTY_STRING,
                                                TStrVec(),
                                                false,
                                                key,
                                                features,
                                                0,
                                                0});
        std::string person11{"p11"};
        std::string person12{"p12"};
        std::string person21{"p21"};
        std::string person22{"p22"};
        bool addedPerson{false};
        gatherer->addPerson(person11, m_ResourceMonitor, addedPerson);
        gatherer->addPerson(person12, m_ResourceMonitor, addedPerson);
        gatherer->addPerson(person21, m_ResourceMonitor, addedPerson);
        gatherer->addPerson(person22, m_ResourceMonitor, addedPerson);

        model.reset(new model::CMockModel{params, gatherer, {/*we don't care about influence*/}});

        maths::CTimeSeriesDecomposition trend;
        maths::CNormalMeanPrecConjugate prior{
            maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData)};
        maths::CModelParams timeSeriesModelParams{bucketLength, 1.0, 0.001, 0.2};
        maths::CUnivariateTimeSeriesModel timeSeriesModel{timeSeriesModelParams, 0, trend, prior};
        model->mockTimeSeriesModels({model::CMockModel::TMathsModelPtr(timeSeriesModel.clone(0)),
                                     model::CMockModel::TMathsModelPtr(timeSeriesModel.clone(1)),
                                     model::CMockModel::TMathsModelPtr(timeSeriesModel.clone(2)),
                                     model::CMockModel::TMathsModelPtr(timeSeriesModel.clone(3))});
    };

    LOG_DEBUG("Individual sum");
    {
        features.assign(1, model_t::E_IndividualSumByBucketAndPerson);
        setupTest();

        TDoubleVec values{2.0, 3.0, 0.0, 0.0};
        {
            std::size_t pid{0};
            for (auto value : values) {
                model->mockAddBucketValue(
                    model_t::E_IndividualSumByBucketAndPerson, pid++, 0, 0, {value});
            }
        }

        model::CModelPlotData plotData;
        model->details()->modelPlot(0, 90.0, {}, plotData);
        CPPUNIT_ASSERT(plotData.begin() != plotData.end());
        for (const auto &featureByFieldData : plotData) {
            CPPUNIT_ASSERT_EQUAL(values.size(), featureByFieldData.second.size());
            for (const auto &byFieldData : featureByFieldData.second) {
                std::size_t pid;
                CPPUNIT_ASSERT(gatherer->personId(byFieldData.first, pid));
                CPPUNIT_ASSERT_EQUAL(std::size_t(1),
                                     byFieldData.second.s_ValuesPerOverField.size());
                for (const auto &currentBucketValue : byFieldData.second.s_ValuesPerOverField) {
                    CPPUNIT_ASSERT_EQUAL(values[pid], currentBucketValue.second);
                }
            }
        }
    }

    LOG_DEBUG("Individual count");
    {
        features.assign(1, model_t::E_IndividualCountByBucketAndPerson);
        setupTest();

        TDoubleVec values{0.0, 1.0, 3.0};
        {
            std::size_t pid{0};
            for (auto value : values) {
                model->mockAddBucketValue(
                    model_t::E_IndividualCountByBucketAndPerson, pid++, 0, 0, {value});
            }
        }

        model::CModelPlotData plotData;
        model->details()->modelPlot(0, 90.0, {}, plotData);
        CPPUNIT_ASSERT(plotData.begin() != plotData.end());
        for (const auto &featureByFieldData : plotData) {
            CPPUNIT_ASSERT_EQUAL(values.size(), featureByFieldData.second.size());
            for (const auto &byFieldData : featureByFieldData.second) {
                std::size_t pid;
                CPPUNIT_ASSERT(gatherer->personId(byFieldData.first, pid));
                CPPUNIT_ASSERT_EQUAL(std::size_t(1),
                                     byFieldData.second.s_ValuesPerOverField.size());
                for (const auto &currentBucketValue : byFieldData.second.s_ValuesPerOverField) {
                    CPPUNIT_ASSERT_EQUAL(values[pid], currentBucketValue.second);
                }
            }
        }
    }
}

CppUnit::Test *CModelDetailsViewTest::suite() {
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CModelDetailsViewTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CModelDetailsViewTest>(
        "CModelDetailsViewTest::testModelPlot", &CModelDetailsViewTest::testModelPlot));

    return suiteOfTests;
}
