/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesModel.h>

#include <model/CForecastModelPersist.h>

#include <test/BoostTestPointerOutput.h>
#include <test/CTestTmpDir.h>

#include <boost/test/unit_test.hpp>

#include <cstdio>

BOOST_AUTO_TEST_SUITE(CForecastModelPersistTest)

using namespace ml;
using namespace model;

BOOST_AUTO_TEST_CASE(testPersistAndRestore) {
    core_t::TTime bucketLength{1800};
    double minimumSeasonalVarianceScale = 0.2;
    SModelParams params{bucketLength};
    params.s_DecayRate = 0.001;
    params.s_LearnRate = 1.0;
    params.s_MinimumTimeToDetectChange = 6 * core::constants::HOUR;
    params.s_MaximumTimeToTestForChange = core::constants::DAY;
    maths::CTimeSeriesDecomposition trend(params.s_DecayRate, bucketLength);

    maths::CNormalMeanPrecConjugate prior{maths::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, params.s_DecayRate)};
    maths::CModelParams timeSeriesModelParams{bucketLength,
                                              params.s_LearnRate,
                                              params.s_DecayRate,
                                              minimumSeasonalVarianceScale,
                                              params.s_MinimumTimeToDetectChange,
                                              params.s_MaximumTimeToTestForChange};
    maths::CUnivariateTimeSeriesModel timeSeriesModel{timeSeriesModelParams, 1, trend, prior};

    CForecastModelPersist::CPersist persister(ml::test::CTestTmpDir::tmpDir());
    persister.addModel(&timeSeriesModel, 10, 50,
                       model_t::EFeature::E_IndividualCountByBucketAndPerson,
                       "some_by_field");
    trend.dataType(maths_t::E_MixedData);
    maths::CNormalMeanPrecConjugate otherPrior{maths::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_MixedData, params.s_DecayRate)};

    maths::CUnivariateTimeSeriesModel otherTimeSeriesModel{timeSeriesModelParams,
                                                           2, trend, otherPrior};

    persister.addModel(&otherTimeSeriesModel, 5, 45,
                       model_t::EFeature::E_IndividualLowMeanByPerson, "some_other_by_field");

    trend.dataType(maths_t::E_DiscreteData);
    maths::CNormalMeanPrecConjugate otherPriorEmptyByField{
        maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_DiscreteData,
                                                             params.s_DecayRate)};
    maths::CUnivariateTimeSeriesModel otherTimeSeriesModelEmptyByField{
        timeSeriesModelParams, 3, trend, otherPriorEmptyByField};

    persister.addModel(&otherTimeSeriesModelEmptyByField, 25, 65,
                       model_t::EFeature::E_IndividualHighMedianByPerson, "");
    std::string persistedModels = persister.finalizePersistAndGetFile();

    {
        CForecastModelPersist::CRestore restorer(params, minimumSeasonalVarianceScale,
                                                 persistedModels);
        CForecastModelPersist::TMathsModelPtr restoredModel;
        core_t::TTime firstDataTime;
        core_t::TTime lastDataTime;
        std::string restoredByFieldValue;
        model_t::EFeature restoredFeature;

        // test timeSeriesModel
        BOOST_TEST_REQUIRE(restorer.nextModel(restoredModel, firstDataTime, lastDataTime,
                                              restoredFeature, restoredByFieldValue));

        BOOST_TEST_REQUIRE(restoredModel);
        BOOST_REQUIRE_EQUAL(model_t::EFeature::E_IndividualCountByBucketAndPerson,
                            restoredFeature);
        BOOST_REQUIRE_EQUAL(core_t::TTime(10), firstDataTime);
        BOOST_REQUIRE_EQUAL(core_t::TTime(50), lastDataTime);
        BOOST_REQUIRE_EQUAL(std::string("some_by_field"), restoredByFieldValue);
        BOOST_REQUIRE_EQUAL(bucketLength, restoredModel->params().bucketLength());
        BOOST_REQUIRE_EQUAL(size_t(1), restoredModel->identifier());
        BOOST_REQUIRE_EQUAL(maths_t::E_ContinuousData, restoredModel->dataType());

        CForecastModelPersist::TMathsModelPtr timeSeriesModelForForecast{
            timeSeriesModel.cloneForForecast()};
        BOOST_REQUIRE_EQUAL(timeSeriesModelForForecast->params().learnRate(),
                            restoredModel->params().learnRate());
        BOOST_REQUIRE_EQUAL(params.s_DecayRate, restoredModel->params().decayRate());
        BOOST_REQUIRE_EQUAL(minimumSeasonalVarianceScale,
                            restoredModel->params().minimumSeasonalVarianceScale());
        BOOST_REQUIRE_EQUAL(params.s_MinimumTimeToDetectChange,
                            restoredModel->params().minimumTimeToDetectChange());
        BOOST_REQUIRE_EQUAL(params.s_MaximumTimeToTestForChange,
                            restoredModel->params().maximumTimeToTestForChange());
        BOOST_REQUIRE_EQUAL(timeSeriesModelForForecast->checksum(42),
                            restoredModel->checksum(42));

        // test otherTimeSeriesModel
        restoredModel.reset();
        BOOST_TEST_REQUIRE(restorer.nextModel(restoredModel, firstDataTime, lastDataTime,
                                              restoredFeature, restoredByFieldValue));
        BOOST_TEST_REQUIRE(restoredModel);
        BOOST_REQUIRE_EQUAL(core_t::TTime(5), firstDataTime);
        BOOST_REQUIRE_EQUAL(core_t::TTime(45), lastDataTime);
        BOOST_REQUIRE_EQUAL(model_t::EFeature::E_IndividualLowMeanByPerson, restoredFeature);
        BOOST_REQUIRE_EQUAL(std::string("some_other_by_field"), restoredByFieldValue);
        BOOST_REQUIRE_EQUAL(bucketLength, restoredModel->params().bucketLength());
        BOOST_REQUIRE_EQUAL(size_t(2), restoredModel->identifier());
        BOOST_REQUIRE_EQUAL(maths_t::E_MixedData, restoredModel->dataType());
        CForecastModelPersist::TMathsModelPtr otherTimeSeriesModelForForecast{
            otherTimeSeriesModel.cloneForForecast()};
        BOOST_REQUIRE_EQUAL(otherTimeSeriesModelForForecast->params().learnRate(),
                            restoredModel->params().learnRate());
        BOOST_REQUIRE_EQUAL(params.s_DecayRate, restoredModel->params().decayRate());
        BOOST_REQUIRE_EQUAL(minimumSeasonalVarianceScale,
                            restoredModel->params().minimumSeasonalVarianceScale());
        BOOST_REQUIRE_EQUAL(params.s_MinimumTimeToDetectChange,
                            restoredModel->params().minimumTimeToDetectChange());
        BOOST_REQUIRE_EQUAL(params.s_MaximumTimeToTestForChange,
                            restoredModel->params().maximumTimeToTestForChange());
        BOOST_REQUIRE_EQUAL(otherTimeSeriesModelForForecast->checksum(42),
                            restoredModel->checksum(42));

        // test otherTimeSeriesModelEmptyByField
        restoredModel.reset();
        BOOST_TEST_REQUIRE(restorer.nextModel(restoredModel, firstDataTime, lastDataTime,
                                              restoredFeature, restoredByFieldValue));
        BOOST_TEST_REQUIRE(restoredModel);
        BOOST_REQUIRE_EQUAL(core_t::TTime(25), firstDataTime);
        BOOST_REQUIRE_EQUAL(core_t::TTime(65), lastDataTime);
        BOOST_REQUIRE_EQUAL(model_t::EFeature::E_IndividualHighMedianByPerson, restoredFeature);
        BOOST_REQUIRE_EQUAL(std::string(), restoredByFieldValue);
        BOOST_REQUIRE_EQUAL(bucketLength, restoredModel->params().bucketLength());
        BOOST_REQUIRE_EQUAL(size_t(3), restoredModel->identifier());
        BOOST_REQUIRE_EQUAL(maths_t::E_DiscreteData, restoredModel->dataType());
        CForecastModelPersist::TMathsModelPtr otherTimeSeriesModelEmptyByFieldForForecast{
            otherTimeSeriesModelEmptyByField.cloneForForecast()};
        BOOST_REQUIRE_EQUAL(otherTimeSeriesModelEmptyByFieldForForecast->checksum(42),
                            restoredModel->checksum(42));

        BOOST_TEST_REQUIRE(!restorer.nextModel(restoredModel, firstDataTime, lastDataTime,
                                               restoredFeature, restoredByFieldValue));
    }
    std::remove(persistedModels.c_str());
}

BOOST_AUTO_TEST_CASE(testPersistAndRestoreEmpty) {
    core_t::TTime bucketLength{1800};
    double minimumSeasonalVarianceScale = 0.2;
    SModelParams params{bucketLength};

    CForecastModelPersist::CPersist persister(ml::test::CTestTmpDir::tmpDir());
    std::string persistedModels = persister.finalizePersistAndGetFile();
    {
        CForecastModelPersist::CRestore restorer(params, minimumSeasonalVarianceScale,
                                                 persistedModels);
        CForecastModelPersist::TMathsModelPtr restoredModel;
        core_t::TTime firstDataTime;
        core_t::TTime lastDataTime;
        std::string restoredByFieldValue;
        model_t::EFeature restoredFeature;

        BOOST_TEST_REQUIRE(!restorer.nextModel(restoredModel, firstDataTime, lastDataTime,
                                               restoredFeature, restoredByFieldValue));
    }
    std::remove(persistedModels.c_str());
}

BOOST_AUTO_TEST_SUITE_END()
