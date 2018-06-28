/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CTimeSeriesChangeDetectorTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CTriple.h>
#include <core/Constants.h>

#include <maths/CGammaRateConjugate.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CModel.h>
#include <maths/CMultimodalPrior.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/COneOfNPrior.h>
#include <maths/CRestoreParams.h>
#include <maths/CTimeSeriesChangeDetector.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CXMeansOnline1d.h>

#include <test/CRandomNumbers.h>

#include "TestUtils.h"

#include <cmath>
#include <memory>
#include <vector>

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrCBuf = boost::circular_buffer<TTimeDoublePr>;
using TDecompositionPtr = std::shared_ptr<maths::CTimeSeriesDecomposition>;
using TPriorPtr = std::shared_ptr<maths::CPrior>;

core_t::TTime BUCKET_LENGTH{1800};
const double DECAY_RATE{0.0002};

TPriorPtr makeResidualModel() {
    maths::CGammaRateConjugate gamma{maths::CGammaRateConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, 0.1, DECAY_RATE)};
    maths::CLogNormalMeanPrecConjugate lognormal{maths::CLogNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, 1.0, DECAY_RATE)};
    maths::CNormalMeanPrecConjugate normal{maths::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, DECAY_RATE)};

    maths::COneOfNPrior::TPriorPtrVec mode;
    mode.reserve(3u);
    mode.emplace_back(gamma.clone());
    mode.emplace_back(lognormal.clone());
    mode.emplace_back(normal.clone());
    maths::COneOfNPrior modePrior{mode, maths_t::E_ContinuousData, DECAY_RATE};
    maths::CXMeansOnline1d clusterer{maths_t::E_ContinuousData,
                                     maths::CAvailableModeDistributions::ALL,
                                     maths_t::E_ClustersFractionWeight,
                                     DECAY_RATE,
                                     0.05,
                                     12.0,
                                     1.0};
    maths::CMultimodalPrior multimodal{maths_t::E_ContinuousData, clusterer,
                                       modePrior, DECAY_RATE};

    maths::COneOfNPrior::TPriorPtrVec models;
    mode.emplace_back(gamma.clone());
    mode.emplace_back(lognormal.clone());
    mode.emplace_back(normal.clone());
    mode.emplace_back(multimodal.clone());

    return TPriorPtr{
        maths::COneOfNPrior{mode, maths_t::E_ContinuousData, DECAY_RATE}.clone()};
}
}

void CTimeSeriesChangeDetectorTest::testNoChange() {
    test::CRandomNumbers rng;

    TDoubleVec variances{1.0, 10.0, 20.0, 30.0, 100.0, 1000.0};
    TDoubleVec scales{0.1, 1.0, 2.0, 3.0, 5.0, 8.0};

    TDoubleVec samples;
    for (std::size_t t = 0u; t < 100; ++t) {
        if (t % 10 == 0) {
            LOG_DEBUG(<< t << "%");
        }

        switch (t % 3) {
        case 0:
            rng.generateNormalSamples(10.0, variances[(t / 3) % variances.size()],
                                      1000, samples);
            break;
        case 1:
            rng.generateLogNormalSamples(1.0, scales[(t / 3) % scales.size()], 1000, samples);
            break;
        case 2:
            rng.generateGammaSamples(10.0, 10.0 * scales[(t / 3) % scales.size()],
                                     1000, samples);
            break;
        }

        TDecompositionPtr trendModel(
            new maths::CTimeSeriesDecomposition{DECAY_RATE, BUCKET_LENGTH});
        TPriorPtr residualModel(makeResidualModel());

        auto addSampleToModel = [&trendModel, &residualModel](core_t::TTime time, double x) {
            trendModel->addPoint(time, x);
            double detrended{trendModel->detrend(time, x, 0.0)};
            residualModel->addSamples({detrended}, maths_t::CUnitWeights::SINGLE_UNIT);
            residualModel->propagateForwardsByTime(1.0);
        };

        core_t::TTime time{0};
        for (std::size_t i = 0u; i < 950; ++i) {
            addSampleToModel(time, samples[i]);
            time += BUCKET_LENGTH;
        }

        maths::CUnivariateTimeSeriesChangeDetector detector{
            trendModel, residualModel, 6 * core::constants::HOUR,
            24 * core::constants::HOUR, 14.0};
        for (std::size_t i = 950u; i < samples.size(); ++i) {
            addSampleToModel(time, samples[i]);
            detector.addSamples({{time, samples[i]}}, maths_t::CUnitWeights::SINGLE_UNIT);
            if (detector.stopTesting()) {
                break;
            }

            CPPUNIT_ASSERT(!detector.change());

            time += BUCKET_LENGTH;
        }
    }
}

void CTimeSeriesChangeDetectorTest::testLevelShift() {
    TGeneratorVec trends{constant, ramp, smoothDaily, weekends, spikeyDaily};
    this->testChange(
        trends, maths::SChangeDescription::E_LevelShift,
        [](TGenerator trend, core_t::TTime time) { return trend(time) + 0.5; },
        5.0, 0.0, 15.0);
}

void CTimeSeriesChangeDetectorTest::testLinearScale() {
    TGeneratorVec trends{smoothDaily, spikeyDaily};
    this->testChange(
        trends, maths::SChangeDescription::E_LinearScale,
        [](TGenerator trend, core_t::TTime time) { return 3.0 * trend(time); },
        3.0, 0.0, 15.0);
}

void CTimeSeriesChangeDetectorTest::testTimeShift() {
    TGeneratorVec trends{smoothDaily, spikeyDaily};
    this->testChange(trends, maths::SChangeDescription::E_TimeShift,
                     [](TGenerator trend, core_t::TTime time) {
                         return trend(time - core::constants::HOUR);
                     },
                     -static_cast<double>(core::constants::HOUR), 0.05, 23.0);
    this->testChange(trends, maths::SChangeDescription::E_TimeShift,
                     [](TGenerator trend, core_t::TTime time) {
                         return trend(time + core::constants::HOUR);
                     },
                     +static_cast<double>(core::constants::HOUR), 0.05, 23.0);
}

void CTimeSeriesChangeDetectorTest::testPersist() {
    test::CRandomNumbers rng;

    TDoubleVec samples;
    rng.generateNormalSamples(10.0, 10.0, 1000, samples);

    TDecompositionPtr trendModel(new maths::CTimeSeriesDecomposition{DECAY_RATE, BUCKET_LENGTH});
    TPriorPtr residualModel(makeResidualModel());

    auto addSampleToModel = [&trendModel, &residualModel](core_t::TTime time, double x) {
        trendModel->addPoint(time, x);
        double detrended{trendModel->detrend(time, x, 0.0)};
        residualModel->addSamples({detrended}, maths_t::CUnitWeights::SINGLE_UNIT);
        residualModel->propagateForwardsByTime(1.0);
    };

    core_t::TTime time{0};
    for (std::size_t i = 0u; i < 990; ++i) {
        addSampleToModel(time, samples[i]);
        time += BUCKET_LENGTH;
    }

    maths::CUnivariateTimeSeriesChangeDetector origDetector{
        trendModel, residualModel, 6 * core::constants::HOUR,
        24 * core::constants::HOUR, 12.0};

    maths::CModelParams modelParams{
        BUCKET_LENGTH, 1.0, 0.0, 1.0, 6 * core::constants::HOUR, 24 * core::constants::HOUR};
    maths::SDistributionRestoreParams distributionParams{maths_t::E_ContinuousData, DECAY_RATE};
    maths::STimeSeriesDecompositionRestoreParams decompositionParams{
        DECAY_RATE, BUCKET_LENGTH, distributionParams};
    maths::SModelRestoreParams params{modelParams, decompositionParams, distributionParams};

    for (std::size_t i = 990u; i < samples.size(); ++i) {
        addSampleToModel(time, samples[i]);
        std::string origXml;
        {
            ml::core::CRapidXmlStatePersistInserter inserter{"root"};
            origDetector.acceptPersistInserter(inserter);
            inserter.toXml(origXml);
        }

        maths::CUnivariateTimeSeriesChangeDetector restoredDetector{
            trendModel, residualModel, 6 * core::constants::HOUR,
            24 * core::constants::HOUR, 12.0};
        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);
        traverser.traverseSubLevel(boost::bind(
            &maths::CUnivariateTimeSeriesChangeDetector::acceptRestoreTraverser,
            &restoredDetector, boost::cref(params), _1));

        LOG_DEBUG(<< "expected " << origDetector.checksum() << " got "
                  << restoredDetector.checksum());
        CPPUNIT_ASSERT_EQUAL(origDetector.checksum(), restoredDetector.checksum());
    }
}

CppUnit::Test* CTimeSeriesChangeDetectorTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CTimeSeriesChangeDetectorTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesChangeDetectorTest>(
        "CTimeSeriesChangeDetectorTest::testNoChange",
        &CTimeSeriesChangeDetectorTest::testNoChange));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesChangeDetectorTest>(
        "CTimeSeriesChangeDetectorTest::testLevelShift",
        &CTimeSeriesChangeDetectorTest::testLevelShift));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesChangeDetectorTest>(
        "CTimeSeriesChangeDetectorTest::testLinearScale",
        &CTimeSeriesChangeDetectorTest::testLinearScale));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesChangeDetectorTest>(
        "CTimeSeriesChangeDetectorTest::testTimeShift",
        &CTimeSeriesChangeDetectorTest::testTimeShift));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTimeSeriesChangeDetectorTest>(
        "CTimeSeriesChangeDetectorTest::testPersist",
        &CTimeSeriesChangeDetectorTest::testPersist));

    return suiteOfTests;
}

void CTimeSeriesChangeDetectorTest::testChange(const TGeneratorVec& trends,
                                               maths::SChangeDescription::EDescription description,
                                               TChange applyChange,
                                               double expectedChange,
                                               double maximumFalseNegatives,
                                               double maximumMeanBucketsToDetectChange) {
    using TOptionalSize = boost::optional<std::size_t>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

    test::CRandomNumbers rng;

    double falseNegatives{0.0};
    TMeanAccumulator meanBucketsToDetect;

    TDoubleVec samples;
    for (std::size_t t = 0u; t < 100; ++t) {
        if (t % 10 == 0) {
            LOG_DEBUG(<< t << "%");
        }

        rng.generateNormalSamples(0.0, 1.0, 1000, samples);

        TDecompositionPtr trendModel(
            new maths::CTimeSeriesDecomposition{DECAY_RATE, BUCKET_LENGTH});
        TPriorPtr residualModel(makeResidualModel());

        auto addSampleToModel = [&trendModel, &residualModel](
                                    core_t::TTime time, double x, double weight) {
            trendModel->addPoint(time, x, maths_t::countWeight(weight));
            double detrended{trendModel->detrend(time, x, 0.0)};
            residualModel->addSamples({detrended}, {maths_t::countWeight(weight)});
            residualModel->propagateForwardsByTime(1.0);
        };

        core_t::TTime time{0};
        for (std::size_t i = 0u; i < 950; ++i) {
            double x{10.0 * trends[t % trends.size()](time) + samples[i]};
            addSampleToModel(time, x, 1.0);
            time += BUCKET_LENGTH;
        }

        maths::CUnivariateTimeSeriesChangeDetector detector{
            trendModel, residualModel, 6 * core::constants::HOUR,
            24 * core::constants::HOUR, 14.0};

        TOptionalSize bucketsToDetect;
        for (std::size_t i = 950u; i < samples.size(); ++i) {
            double x{10.0 * applyChange(trends[t % trends.size()], time) + samples[i]};

            addSampleToModel(time, x, 0.5);
            detector.addSamples({{time, x}}, maths_t::CUnitWeights::SINGLE_UNIT);

            auto change = detector.change();
            if (change) {
                if (!bucketsToDetect) {
                    bucketsToDetect.reset(i - 949);
                }
                CPPUNIT_ASSERT_EQUAL(change->s_Description, description);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedChange, change->s_Value[0],
                                             0.5 * std::fabs(expectedChange));
                break;
            }
            if (detector.stopTesting()) {
                break;
            }

            time += BUCKET_LENGTH;
        }
        if (bucketsToDetect == boost::none) {
            falseNegatives += 0.01;
        } else {
            meanBucketsToDetect.add(static_cast<double>(*bucketsToDetect));
        }
    }

    LOG_DEBUG(<< "false negatives = " << falseNegatives);
    LOG_DEBUG(<< "buckets to detect = " << maths::CBasicStatistics::mean(meanBucketsToDetect));
    CPPUNIT_ASSERT(falseNegatives <= maximumFalseNegatives);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanBucketsToDetect) <
                   maximumMeanBucketsToDetectChange);
}
