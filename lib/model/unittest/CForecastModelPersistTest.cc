/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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

#include "CForecastModelPersistTest.h"

#include <core/CLogger.h>
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
    LOG_DEBUG("+----------------------------------------------------+");
    LOG_DEBUG("|  CForecastModelPersistTest::testPersistAndRestore  |");
    LOG_DEBUG("+----------------------------------------------------+");

    core_t::TTime bucketLength{1800};
    double minimumSeasonalVarianceScale = 0.2;
    SModelParams params{bucketLength};
    params.s_DecayRate = 0.001;
    params.s_LearnRate = 1.0;
    maths::CTimeSeriesDecomposition trend(params.s_DecayRate, bucketLength);
    maths::CNormalMeanPrecConjugate prior{
        maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, params.s_DecayRate)};
    maths::CModelParams timeSeriesModelParams{bucketLength, params.s_LearnRate, params.s_DecayRate, minimumSeasonalVarianceScale};
    maths::CUnivariateTimeSeriesModel timeSeriesModel{timeSeriesModelParams, 1, trend, prior};

    CForecastModelPersist::CPersist persister(ml::test::CTestTmpDir::tmpDir());
    persister.addModel(&timeSeriesModel, model_t::EFeature::E_IndividualCountByBucketAndPerson, "some_by_field");

    maths::CNormalMeanPrecConjugate otherPrior{
        maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_MixedData, params.s_DecayRate)};
    maths::CUnivariateTimeSeriesModel otherTimeSeriesModel{timeSeriesModelParams, 2, trend, otherPrior};

    persister.addModel(&otherTimeSeriesModel, model_t::EFeature::E_IndividualLowMeanByPerson, "some_other_by_field");

    maths::CNormalMeanPrecConjugate otherPriorEmptyByField{
        maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_DiscreteData, params.s_DecayRate)};
    maths::CUnivariateTimeSeriesModel otherTimeSeriesModelEmptyByField{timeSeriesModelParams, 3, trend, otherPriorEmptyByField};

    persister.addModel(&otherTimeSeriesModelEmptyByField, model_t::EFeature::E_IndividualHighMedianByPerson, "");
    std::string persistedModels = persister.finalizePersistAndGetFile();

    {
        CForecastModelPersist::CRestore restorer(params, minimumSeasonalVarianceScale, persistedModels);
        CForecastModelPersist::TMathsModelPtr restoredModel;
        std::string restoredByFieldValue;
        model_t::EFeature restoredFeature;

        // test timeSeriesModel
        CPPUNIT_ASSERT(restorer.nextModel(restoredModel, restoredFeature, restoredByFieldValue));

        CPPUNIT_ASSERT(restoredModel);
        CPPUNIT_ASSERT_EQUAL(model_t::EFeature::E_IndividualCountByBucketAndPerson, restoredFeature);
        CPPUNIT_ASSERT_EQUAL(std::string("some_by_field"), restoredByFieldValue);
        CPPUNIT_ASSERT_EQUAL(bucketLength, restoredModel->params().bucketLength());
        CPPUNIT_ASSERT_EQUAL(size_t(1), restoredModel->identifier());
        CPPUNIT_ASSERT_EQUAL(maths_t::E_ContinuousData, restoredModel->dataType());

        CForecastModelPersist::TMathsModelPtr timeSeriesModelForForecast{timeSeriesModel.cloneForForecast()};
        CPPUNIT_ASSERT_EQUAL(timeSeriesModelForForecast->params().learnRate(), restoredModel->params().learnRate());
        CPPUNIT_ASSERT_EQUAL(params.s_DecayRate, restoredModel->params().decayRate());
        CPPUNIT_ASSERT_EQUAL(minimumSeasonalVarianceScale, restoredModel->params().minimumSeasonalVarianceScale());

        CPPUNIT_ASSERT_EQUAL(timeSeriesModelForForecast->checksum(42), restoredModel->checksum(42));

        // test otherTimeSeriesModel
        restoredModel.reset();
        CPPUNIT_ASSERT(restorer.nextModel(restoredModel, restoredFeature, restoredByFieldValue));
        CPPUNIT_ASSERT(restoredModel);
        CPPUNIT_ASSERT_EQUAL(model_t::EFeature::E_IndividualLowMeanByPerson, restoredFeature);
        CPPUNIT_ASSERT_EQUAL(std::string("some_other_by_field"), restoredByFieldValue);
        CPPUNIT_ASSERT_EQUAL(bucketLength, restoredModel->params().bucketLength());
        CPPUNIT_ASSERT_EQUAL(size_t(2), restoredModel->identifier());
        CPPUNIT_ASSERT_EQUAL(maths_t::E_MixedData, restoredModel->dataType());
        CForecastModelPersist::TMathsModelPtr otherTimeSeriesModelForForecast{otherTimeSeriesModel.cloneForForecast()};
        CPPUNIT_ASSERT_EQUAL(otherTimeSeriesModelForForecast->params().learnRate(), restoredModel->params().learnRate());
        CPPUNIT_ASSERT_EQUAL(params.s_DecayRate, restoredModel->params().decayRate());
        CPPUNIT_ASSERT_EQUAL(minimumSeasonalVarianceScale, restoredModel->params().minimumSeasonalVarianceScale());
        CPPUNIT_ASSERT_EQUAL(otherTimeSeriesModelForForecast->checksum(42), restoredModel->checksum(42));

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
        CPPUNIT_ASSERT_EQUAL(otherTimeSeriesModelEmptyByFieldForForecast->checksum(42), restoredModel->checksum(42));

        CPPUNIT_ASSERT(!restorer.nextModel(restoredModel, restoredFeature, restoredByFieldValue));
    }
    std::remove(persistedModels.c_str());
}

CppUnit::Test* CForecastModelPersistTest::suite(void) {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CForecastModelPersistTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastModelPersistTest>("CForecastModelPersistTest::testPersistAndRestore",
                                                                             &CForecastModelPersistTest::testPersistAndRestore));

    return suiteOfTests;
}
