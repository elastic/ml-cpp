/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CForecastModelPersistTest.h"

#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesModel.h>

#include <model/CForecastModelPersist.h>

#include <test/CTestTmpDir.h>

#include <cstdio>

using namespace ml;
using namespace model;

void CForecastModelPersistTest::testPersistAndRestore() {
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
    persister.addModel(&timeSeriesModel, model_t::EFeature::E_IndividualCountByBucketAndPerson,
                       "some_by_field");
    trend.dataType(maths_t::E_MixedData);
    maths::CNormalMeanPrecConjugate otherPrior{maths::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_MixedData, params.s_DecayRate)};

    maths::CUnivariateTimeSeriesModel otherTimeSeriesModel{timeSeriesModelParams,
                                                           2, trend, otherPrior};

    persister.addModel(&otherTimeSeriesModel, model_t::EFeature::E_IndividualLowMeanByPerson,
                       "some_other_by_field");

    trend.dataType(maths_t::E_DiscreteData);
    maths::CNormalMeanPrecConjugate otherPriorEmptyByField{
        maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_DiscreteData,
                                                             params.s_DecayRate)};
    maths::CUnivariateTimeSeriesModel otherTimeSeriesModelEmptyByField{
        timeSeriesModelParams, 3, trend, otherPriorEmptyByField};

    persister.addModel(&otherTimeSeriesModelEmptyByField,
                       model_t::EFeature::E_IndividualHighMedianByPerson, "");
    std::string persistedModels = persister.finalizePersistAndGetFile();

    {
        CForecastModelPersist::CRestore restorer(params, minimumSeasonalVarianceScale,
                                                 persistedModels);
        CForecastModelPersist::TMathsModelPtr restoredModel;
        std::string restoredByFieldValue;
        model_t::EFeature restoredFeature;

        // test timeSeriesModel
        CPPUNIT_ASSERT(restorer.nextModel(restoredModel, restoredFeature, restoredByFieldValue));

        CPPUNIT_ASSERT(restoredModel);
        CPPUNIT_ASSERT_EQUAL(model_t::EFeature::E_IndividualCountByBucketAndPerson,
                             restoredFeature);
        CPPUNIT_ASSERT_EQUAL(std::string("some_by_field"), restoredByFieldValue);
        CPPUNIT_ASSERT_EQUAL(bucketLength, restoredModel->params().bucketLength());
        CPPUNIT_ASSERT_EQUAL(size_t(1), restoredModel->identifier());
        CPPUNIT_ASSERT_EQUAL(maths_t::E_ContinuousData, restoredModel->dataType());

        CForecastModelPersist::TMathsModelPtr timeSeriesModelForForecast{
            timeSeriesModel.cloneForForecast()};
        CPPUNIT_ASSERT_EQUAL(timeSeriesModelForForecast->params().learnRate(),
                             restoredModel->params().learnRate());
        CPPUNIT_ASSERT_EQUAL(params.s_DecayRate, restoredModel->params().decayRate());
        CPPUNIT_ASSERT_EQUAL(minimumSeasonalVarianceScale,
                             restoredModel->params().minimumSeasonalVarianceScale());
        CPPUNIT_ASSERT_EQUAL(params.s_MinimumTimeToDetectChange,
                             restoredModel->params().minimumTimeToDetectChange());
        CPPUNIT_ASSERT_EQUAL(params.s_MaximumTimeToTestForChange,
                             restoredModel->params().maximumTimeToTestForChange());
        CPPUNIT_ASSERT_EQUAL(timeSeriesModelForForecast->checksum(42),
                             restoredModel->checksum(42));

        // test otherTimeSeriesModel
        restoredModel.reset();
        CPPUNIT_ASSERT(restorer.nextModel(restoredModel, restoredFeature, restoredByFieldValue));
        CPPUNIT_ASSERT(restoredModel);
        CPPUNIT_ASSERT_EQUAL(model_t::EFeature::E_IndividualLowMeanByPerson, restoredFeature);
        CPPUNIT_ASSERT_EQUAL(std::string("some_other_by_field"), restoredByFieldValue);
        CPPUNIT_ASSERT_EQUAL(bucketLength, restoredModel->params().bucketLength());
        CPPUNIT_ASSERT_EQUAL(size_t(2), restoredModel->identifier());
        CPPUNIT_ASSERT_EQUAL(maths_t::E_MixedData, restoredModel->dataType());
        CForecastModelPersist::TMathsModelPtr otherTimeSeriesModelForForecast{
            otherTimeSeriesModel.cloneForForecast()};
        CPPUNIT_ASSERT_EQUAL(otherTimeSeriesModelForForecast->params().learnRate(),
                             restoredModel->params().learnRate());
        CPPUNIT_ASSERT_EQUAL(params.s_DecayRate, restoredModel->params().decayRate());
        CPPUNIT_ASSERT_EQUAL(minimumSeasonalVarianceScale,
                             restoredModel->params().minimumSeasonalVarianceScale());
        CPPUNIT_ASSERT_EQUAL(params.s_MinimumTimeToDetectChange,
                             restoredModel->params().minimumTimeToDetectChange());
        CPPUNIT_ASSERT_EQUAL(params.s_MaximumTimeToTestForChange,
                             restoredModel->params().maximumTimeToTestForChange());
        CPPUNIT_ASSERT_EQUAL(otherTimeSeriesModelForForecast->checksum(42),
                             restoredModel->checksum(42));

        // test otherTimeSeriesModelEmptyByField
        restoredModel.reset();
        CPPUNIT_ASSERT(restorer.nextModel(restoredModel, restoredFeature, restoredByFieldValue));
        CPPUNIT_ASSERT(restoredModel);
        CPPUNIT_ASSERT_EQUAL(model_t::EFeature::E_IndividualHighMedianByPerson, restoredFeature);
        CPPUNIT_ASSERT_EQUAL(std::string(), restoredByFieldValue);
        CPPUNIT_ASSERT_EQUAL(bucketLength, restoredModel->params().bucketLength());
        CPPUNIT_ASSERT_EQUAL(size_t(3), restoredModel->identifier());
        CPPUNIT_ASSERT_EQUAL(maths_t::E_DiscreteData, restoredModel->dataType());
        CForecastModelPersist::TMathsModelPtr otherTimeSeriesModelEmptyByFieldForForecast{
            otherTimeSeriesModelEmptyByField.cloneForForecast()};
        CPPUNIT_ASSERT_EQUAL(otherTimeSeriesModelEmptyByFieldForForecast->checksum(42),
                             restoredModel->checksum(42));

        CPPUNIT_ASSERT(!restorer.nextModel(restoredModel, restoredFeature, restoredByFieldValue));
    }
    std::remove(persistedModels.c_str());
}

void CForecastModelPersistTest::testPersistAndRestoreEmpty() {
    core_t::TTime bucketLength{1800};
    double minimumSeasonalVarianceScale = 0.2;
    SModelParams params{bucketLength};

    CForecastModelPersist::CPersist persister(ml::test::CTestTmpDir::tmpDir());
    std::string persistedModels = persister.finalizePersistAndGetFile();
    {
        CForecastModelPersist::CRestore restorer(params, minimumSeasonalVarianceScale,
                                                 persistedModels);
        CForecastModelPersist::TMathsModelPtr restoredModel;
        std::string restoredByFieldValue;
        model_t::EFeature restoredFeature;

        CPPUNIT_ASSERT(!restorer.nextModel(restoredModel, restoredFeature, restoredByFieldValue));
    }
    std::remove(persistedModels.c_str());
}

CppUnit::Test* CForecastModelPersistTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CForecastModelPersistTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastModelPersistTest>(
        "CForecastModelPersistTest::testPersistAndRestore",
        &CForecastModelPersistTest::testPersistAndRestore));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastModelPersistTest>(
        "CForecastModelPersistTest::testPersistAndRestoreEmpty",
        &CForecastModelPersistTest::testPersistAndRestoreEmpty));

    return suiteOfTests;
}
