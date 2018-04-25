/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CTrendComponentTest.h"

#include <core/CLogger.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CDecayRateController.h>
#include <maths/CTrendComponent.h>

#include <test/CRandomNumbers.h>

#include <boost/tuple/tuple.hpp>

#include <cmath>
#include <numeric>

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble3Vec = core::CSmallVector<double, 3>;
using TDouble3VecVec = std::vector<TDouble3Vec>;
using TGenerator = TDoubleVec (*)(test::CRandomNumbers&, core_t::TTime, core_t::TTime, core_t::TTime);
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TRegression = maths::CRegression::CLeastSquaresOnline<2, double>;

TDoubleVec multiscaleRandomWalk(test::CRandomNumbers& rng,
                                core_t::TTime bucketLength,
                                core_t::TTime start,
                                core_t::TTime end) {
    TDoubleVecVec noise(4);

    core_t::TTime buckets{(end - start) / bucketLength + 1};
    rng.generateNormalSamples(0.0, 0.2, buckets, noise[0]);
    rng.generateNormalSamples(0.0, 0.5, buckets, noise[1]);
    rng.generateNormalSamples(0.0, 1.0, buckets, noise[2]);
    rng.generateNormalSamples(0.0, 5.0, buckets, noise[3]);
    for (core_t::TTime i = 1; i < buckets; ++i) {
        noise[0][i] = 0.998 * noise[0][i - 1] + 0.002 * noise[0][i];
        noise[1][i] = 0.99 * noise[1][i - 1] + 0.01 * noise[1][i];
        noise[2][i] = 0.9 * noise[2][i - 1] + 0.1 * noise[2][i];
    }

    TDoubleVec result;
    result.reserve(buckets);

    TDoubleVec rw{0.0, 0.0, 0.0};
    for (core_t::TTime i = 0; i < buckets; ++i) {
        rw[0] = rw[0] + noise[0][i];
        rw[1] = rw[1] + noise[1][i];
        rw[2] = rw[2] + noise[2][i];
        double ramp{0.05 * static_cast<double>(i)};
        result.push_back(ramp + rw[0] + rw[1] + rw[2] + noise[3][i]);
    }

    return result;
}

TDoubleVec piecewiseLinear(test::CRandomNumbers& rng,
                           core_t::TTime bucketLength,
                           core_t::TTime start,
                           core_t::TTime end) {
    core_t::TTime buckets{(end - start) / bucketLength + 1};

    TDoubleVec knots;
    rng.generateUniformSamples(100.0, 500.0, buckets / 200, knots);
    knots.insert(knots.begin(), 0.0);
    std::partial_sum(knots.begin(), knots.end(), knots.begin());

    TDoubleVec slopes;
    rng.generateUniformSamples(-1.0, 2.0, knots.size(), slopes);

    TDoubleVec result;
    result.reserve(buckets);

    double value{0.0};

    auto knot = knots.begin();
    auto slope = slopes.begin();
    for (core_t::TTime time = start; time < end; time += bucketLength) {
        if (time > start + static_cast<core_t::TTime>(bucketLength * *knot)) {
            ++knot;
            ++slope;
        }
        value += *slope;
        result.push_back(value);
    }

    return result;
}

TDoubleVec staircase(test::CRandomNumbers& rng,
                     core_t::TTime bucketLength,
                     core_t::TTime start,
                     core_t::TTime end) {
    core_t::TTime buckets{(end - start) / bucketLength + 1};

    TDoubleVec knots;
    rng.generateUniformSamples(200.0, 400.0, buckets / 200, knots);
    knots.insert(knots.begin(), 0.0);
    std::partial_sum(knots.begin(), knots.end(), knots.begin());

    TDoubleVec steps;
    rng.generateUniformSamples(1.0, 20.0, knots.size(), steps);

    TDoubleVec result;
    result.reserve(buckets);

    double value{0.0};

    auto knot = knots.begin();
    auto step = steps.begin();
    for (core_t::TTime time = start; time < end; time += bucketLength) {
        if (time > start + static_cast<core_t::TTime>(bucketLength * *knot)) {
            value += *step;
            ++knot;
            ++step;
        }
        result.push_back(value);
    }

    return result;
}

TDoubleVec switching(test::CRandomNumbers& rng,
                     core_t::TTime bucketLength,
                     core_t::TTime start,
                     core_t::TTime end) {
    core_t::TTime buckets{(end - start) / bucketLength + 1};

    TDoubleVec knots;
    rng.generateUniformSamples(400.0, 800.0, buckets / 400, knots);
    knots.insert(knots.begin(), 0.0);
    std::partial_sum(knots.begin(), knots.end(), knots.begin());

    TDoubleVec steps;
    rng.generateUniformSamples(-10.0, 10.0, knots.size(), steps);

    TDoubleVec result;
    result.reserve(buckets);

    double value{0.0};

    auto knot = knots.begin();
    auto step = steps.begin();
    for (core_t::TTime time = start; time < end; time += bucketLength) {
        if (time > start + static_cast<core_t::TTime>(bucketLength * *knot)) {
            value += *step;
            ++knot;
            ++step;
        }
        result.push_back(value);
    }

    return result;
}
}

void CTrendComponentTest::testValueAndVariance() {
    // Check that the prediction bias is small in the long run
    // and that the predicted variance approximately matches the
    // variance observed in prediction errors.

    test::CRandomNumbers rng;

    core_t::TTime bucketLength{600};
    core_t::TTime start{1000000};
    core_t::TTime end{3000000};

    TDoubleVec values(multiscaleRandomWalk(rng, bucketLength, start, end));

    maths::CTrendComponent component{0.012};
    maths::CDecayRateController controller(maths::CDecayRateController::E_PredictionBias |
                                               maths::CDecayRateController::E_PredictionErrorIncrease,
                                           1);

    TMeanVarAccumulator normalisedResiduals;
    for (core_t::TTime time = start; time < end; time += bucketLength) {
        double value{values[(time - start) / bucketLength]};
        double prediction{maths::CBasicStatistics::mean(component.value(time, 0.0))};

        if (time > start + bucketLength) {
            double variance{maths::CBasicStatistics::mean(component.variance(0.0))};
            normalisedResiduals.add((value - prediction) / std::sqrt(variance));
        }

        component.add(time, value);
        controller.multiplier({prediction},
                              {{values[(time - start) / bucketLength] - prediction}},
                              bucketLength, 1.0, 0.012);
        component.decayRate(0.012 * controller.multiplier());
        component.propagateForwardsByTime(bucketLength);
    }

    LOG_DEBUG(<< "normalised error moments = " << normalisedResiduals);
    CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::mean(normalisedResiduals)) < 0.5);
    CPPUNIT_ASSERT(std::fabs(maths::CBasicStatistics::variance(normalisedResiduals) - 1.0) < 0.2);
}

void CTrendComponentTest::testDecayRate() {
    // Test that the trend short range predictions approximately
    // match a regression model with the same decay rate.

    test::CRandomNumbers rng;

    core_t::TTime bucketLength{600};
    core_t::TTime start{0};
    core_t::TTime end{3000000};

    //std::ofstream file;
    //file.open("results.m");
    //TDoubleVec predictions;
    //TDoubleVec expectedPredictions;

    TDoubleVec values(multiscaleRandomWalk(rng, bucketLength, start, end));

    maths::CTrendComponent component{0.012};
    TRegression regression;
    maths::CDecayRateController controller(maths::CDecayRateController::E_PredictionBias |
                                               maths::CDecayRateController::E_PredictionErrorIncrease,
                                           1);

    TMeanAccumulator error;
    TMeanAccumulator level;
    for (core_t::TTime time = start; time < end; time += bucketLength) {
        double value{values[(time - start) / bucketLength]};
        component.add(time, value);
        regression.add(time / 604800.0, value);

        double expectedPrediction{regression.predict(time / 604800.0)};
        double prediction{maths::CBasicStatistics::mean(component.value(time, 0.0))};
        error.add(std::fabs(prediction - expectedPrediction));
        level.add(value);

        controller.multiplier({prediction},
                              {{values[(time - start) / bucketLength] - prediction}},
                              bucketLength, 1.0, 0.012);
        component.decayRate(0.012 * controller.multiplier());
        component.propagateForwardsByTime(bucketLength);
        regression.age(std::exp(-0.012 * controller.multiplier() * 600.0 / 86400.0));

        //predictions.push_back(prediction);
        //expectedPredictions.push_back(expectedPrediction);
    }

    double relativeError{maths::CBasicStatistics::mean(error) /
                         std::fabs(maths::CBasicStatistics::mean(level))};
    LOG_DEBUG(<< "relative error = " << relativeError);

    //file << "f  = " << core::CContainerPrinter::print(values) << ";" << std::endl;
    //file << "p  = " << core::CContainerPrinter::print(predictions) << ";" << std::endl;
    //file << "pe = " << core::CContainerPrinter::print(expectedPredictions) << ";" << std::endl;
}

void CTrendComponentTest::testForecast() {
    // Check the forecast errors for a variety of signals.

    test::CRandomNumbers rng;

    auto testForecast = [&rng](TGenerator generate, core_t::TTime start, core_t::TTime end) {
        //std::ofstream file;
        //file.open("results.m");
        //TDoubleVec predictions;
        //TDoubleVec forecastPredictions;
        //TDoubleVec forecastLower;
        //TDoubleVec forecastUpper;

        core_t::TTime bucketLength{600};
        TDoubleVec values(generate(rng, bucketLength, start, end + 1000 * bucketLength));

        maths::CTrendComponent component{0.012};
        maths::CDecayRateController controller(
            maths::CDecayRateController::E_PredictionBias |
                maths::CDecayRateController::E_PredictionErrorIncrease,
            1);

        core_t::TTime time{0};
        for (/**/; time < end; time += bucketLength) {
            component.add(time, values[time / bucketLength]);
            component.propagateForwardsByTime(bucketLength);

            double prediction{maths::CBasicStatistics::mean(component.value(time, 0.0))};
            controller.multiplier({prediction},
                                  {{values[time / bucketLength] - prediction}},
                                  bucketLength, 0.3, 0.012);
            component.decayRate(0.012 * controller.multiplier());
            //predictions.push_back(prediction);
        }

        component.shiftOrigin(time);

        TDouble3VecVec forecast;
        component.forecast(time, time + 1000 * bucketLength, 3600, 95.0, forecast);

        TMeanAccumulator meanError;
        TMeanAccumulator meanErrorAt95;
        for (auto& errorbar : forecast) {
            core_t::TTime bucket{(time - start) / bucketLength};
            meanError.add(std::fabs((values[bucket] - errorbar[1]) /
                                    std::fabs(values[bucket])));
            meanErrorAt95.add(std::max(std::max(values[bucket] - errorbar[2],
                                                errorbar[0] - values[bucket]),
                                       0.0) /
                              std::fabs(values[bucket]));
            //forecastLower.push_back(errorbar[0]);
            //forecastPredictions.push_back(errorbar[1]);
            //forecastUpper.push_back(errorbar[2]);
        }

        //file << "f  = " << core::CContainerPrinter::print(values) << ";" << std::endl;
        //file << "p  = " << core::CContainerPrinter::print(predictions) << ";" << std::endl;
        //file << "fl = " << core::CContainerPrinter::print(forecastLower) << ";" << std::endl;
        //file << "fm = " << core::CContainerPrinter::print(forecastPredictions) << ";" << std::endl;
        //file << "fu = " << core::CContainerPrinter::print(forecastUpper) << ";" << std::endl;

        LOG_DEBUG(<< "error       = " << maths::CBasicStatistics::mean(meanError));
        LOG_DEBUG(<< "error @ 95% = " << maths::CBasicStatistics::mean(meanErrorAt95));

        return std::make_pair(maths::CBasicStatistics::mean(meanError),
                              maths::CBasicStatistics::mean(meanErrorAt95));
    };

    double error;
    double errorAt95;

    LOG_DEBUG(<< "Random Walk");
    {
        boost::tie(error, errorAt95) = testForecast(multiscaleRandomWalk, 0, 3000000);
        CPPUNIT_ASSERT(error < 0.16);
        CPPUNIT_ASSERT(errorAt95 < 0.001);
    }

    LOG_DEBUG(<< "Piecewise Linear");
    {
        boost::tie(error, errorAt95) = testForecast(piecewiseLinear, 0, 3200000);
        CPPUNIT_ASSERT(error < 0.17);
        CPPUNIT_ASSERT(errorAt95 < 0.07);
    }

    LOG_DEBUG(<< "Staircase");
    {
        boost::tie(error, errorAt95) = testForecast(staircase, 0, 2000000);
        CPPUNIT_ASSERT(error < 0.03);
        CPPUNIT_ASSERT(errorAt95 < 0.01);
    }

    LOG_DEBUG(<< "Switching");
    {
        boost::tie(error, errorAt95) = testForecast(switching, 0, 3000000);
        CPPUNIT_ASSERT(error < 0.06);
        CPPUNIT_ASSERT(errorAt95 < 0.01);
    }
}

void CTrendComponentTest::testPersist() {
    // Check that serialization is idempotent.

    test::CRandomNumbers rng;

    core_t::TTime bucketLength{600};
    core_t::TTime start{1200};
    core_t::TTime end{200000};

    TDoubleVec values(multiscaleRandomWalk(rng, bucketLength, start, end));

    maths::CTrendComponent origComponent{0.012};

    for (core_t::TTime time = start; time < end; time += bucketLength) {
        double value{values[(time - start) / bucketLength]};
        origComponent.add(time, value);
        origComponent.propagateForwardsByTime(bucketLength);
    }

    std::string origXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        origComponent.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_DEBUG(<< "decomposition XML representation:\n" << origXml);

    // Restore the XML into a new filter
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::CTrendComponent restoredComponent{0.1};
    traverser.traverseSubLevel(boost::bind(
        &maths::CTrendComponent::acceptRestoreTraverser, &restoredComponent, _1));

    CPPUNIT_ASSERT_EQUAL(origComponent.checksum(), restoredComponent.checksum());

    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredComponent.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

CppUnit::Test* CTrendComponentTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CTrendComponentTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CTrendComponentTest>(
        "CTrendComponentTest::testValueAndVariance", &CTrendComponentTest::testValueAndVariance));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTrendComponentTest>(
        "CTrendComponentTest::testDecayRate", &CTrendComponentTest::testDecayRate));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTrendComponentTest>(
        "CTrendComponentTest::testForecast", &CTrendComponentTest::testForecast));
    suiteOfTests->addTest(new CppUnit::TestCaller<CTrendComponentTest>(
        "CTrendComponentTest::testPersist", &CTrendComponentTest::testPersist));

    return suiteOfTests;
}
