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

#include <core/CJsonStatePersistInserter.h>
#include <core/CJsonStateRestoreTraverser.h>
#include <core/CLogger.h>
#include <core/CoreTypes.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CLeastSquaresOnlineRegression.h>
#include <maths/common/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/common/CRestoreParams.h>

#include <maths/time_series/CDecayRateController.h>
#include <maths/time_series/CTrendComponent.h>

#include <test/CRandomNumbers.h>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace {
BOOST_AUTO_TEST_SUITE(CTrendComponentTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TDouble1Vec = core::CSmallVector<double, 1>;
using TDouble3Vec = core::CSmallVector<double, 3>;
using TDouble3VecVec = std::vector<TDouble3Vec>;
using TGenerator = TDoubleVec (*)(test::CRandomNumbers&, core_t::TTime, core_t::TTime, core_t::TTime);
using TMeanAccumulator = maths::common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = maths::common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TRegression = maths::common::CLeastSquaresOnlineRegression<2, double>;

const core_t::TTime BUCKET_LENGTH{600};

TDoubleVec multiscaleRandomWalk(test::CRandomNumbers& rng, core_t::TTime start, core_t::TTime end) {
    TDoubleVecVec noise(4);

    core_t::TTime buckets{(end - start) / BUCKET_LENGTH + 1};
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

TDoubleVec piecewiseLinear(test::CRandomNumbers& rng, core_t::TTime start, core_t::TTime end) {
    core_t::TTime buckets{(end - start) / BUCKET_LENGTH + 1};

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
    for (core_t::TTime time = start; time < end; time += BUCKET_LENGTH) {
        if (time > start + static_cast<core_t::TTime>(BUCKET_LENGTH * *knot)) {
            ++knot;
            ++slope;
        }
        value += *slope;
        result.push_back(value);
    }

    return result;
}

TDoubleVec staircase(test::CRandomNumbers& rng, core_t::TTime start, core_t::TTime end) {
    core_t::TTime buckets{(end - start) / BUCKET_LENGTH + 1};

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
    for (core_t::TTime time = start; time < end; time += BUCKET_LENGTH) {
        if (time > start + static_cast<core_t::TTime>(BUCKET_LENGTH * *knot)) {
            value += *step;
            ++knot;
            ++step;
        }
        result.push_back(value);
    }

    return result;
}

TDoubleVec switching(test::CRandomNumbers& rng, core_t::TTime start, core_t::TTime end) {
    core_t::TTime buckets{(end - start) / BUCKET_LENGTH + 1};

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
    for (core_t::TTime time = start; time < end; time += BUCKET_LENGTH) {
        if (time > start + static_cast<core_t::TTime>(BUCKET_LENGTH * *knot)) {
            value += *step;
            ++knot;
            ++step;
        }
        result.push_back(value);
    }

    return result;
}

template<typename ITR>
auto trainModel(ITR beginValues, ITR endValues) {

    maths::time_series::CTrendComponent component{0.012};
    maths::time_series::CDecayRateController controller(
        maths::time_series::CDecayRateController::E_PredictionBias |
            maths::time_series::CDecayRateController::E_PredictionErrorIncrease,
        1);

    core_t::TTime time{0};

    for (ITR value = beginValues; value != endValues; ++value, time += BUCKET_LENGTH) {
        component.add(time, *value);
        component.propagateForwardsByTime(BUCKET_LENGTH);

        double prediction{component.value(time, 0.0).mean()};
        controller.multiplier({prediction}, {{*value - prediction}},
                              BUCKET_LENGTH, 0.3, 0.012);
        component.decayRate(0.012 * controller.multiplier());
    }

    component.shiftOrigin(time);

    return std::make_pair(component, time);
}

template<typename ITR>
auto forecastErrors(ITR actual,
                    ITR endActual,
                    core_t::TTime time,
                    const maths::time_series::CTrendComponent& component) {

    core_t::TTime interval(std::distance(actual, endActual) * BUCKET_LENGTH);

    TDouble3VecVec forecast;
    component.forecast(time, time + interval, BUCKET_LENGTH, 95.0, false,
                       [](core_t::TTime) { return TDouble3Vec(3, 0.0); },
                       [&forecast](core_t::TTime, const TDouble3Vec& value) {
                           forecast.push_back(value);
                       });

    TMeanAccumulator meanError;
    TMeanAccumulator meanErrorAt95;

    for (auto errorbar = forecast.begin(); errorbar != forecast.end(); ++errorbar, ++actual) {
        meanError.add(std::fabs((*actual - (*errorbar)[1]) / std::fabs(*actual)));
        meanErrorAt95.add(
            std::max(std::max(*actual - (*errorbar)[2], (*errorbar)[0] - *actual), 0.0) /
            std::fabs(*actual));
    }

    LOG_DEBUG(<< "error       = " << maths::common::CBasicStatistics::mean(meanError));
    LOG_DEBUG(<< "error @ 95% = " << maths::common::CBasicStatistics::mean(meanErrorAt95));

    return std::make_pair(maths::common::CBasicStatistics::mean(meanError),
                          maths::common::CBasicStatistics::mean(meanErrorAt95));
}

BOOST_AUTO_TEST_CASE(testValueAndVariance) {
    // Check that the prediction bias is small in the long run
    // and that the predicted variance approximately matches the
    // variance observed in prediction errors.

    test::CRandomNumbers rng;

    core_t::TTime start{1000000};
    core_t::TTime end{3000000};

    TDoubleVec values(multiscaleRandomWalk(rng, start, end));

    maths::time_series::CTrendComponent component{0.012};
    maths::time_series::CDecayRateController controller(
        maths::time_series::CDecayRateController::E_PredictionBias |
            maths::time_series::CDecayRateController::E_PredictionErrorIncrease,
        1);

    TMeanVarAccumulator normalisedResiduals;
    for (core_t::TTime time = start; time < end; time += BUCKET_LENGTH) {
        double value{values[(time - start) / BUCKET_LENGTH]};
        double prediction{component.value(time, 0.0).mean()};

        if (time > start + BUCKET_LENGTH) {
            double variance{component.variance(0.0).mean()};
            normalisedResiduals.add((value - prediction) / std::sqrt(variance));
        }

        component.add(time, value);
        controller.multiplier({prediction},
                              {{values[(time - start) / BUCKET_LENGTH] - prediction}},
                              BUCKET_LENGTH, 1.0, 0.012);
        component.decayRate(0.012 * controller.multiplier());
        component.propagateForwardsByTime(BUCKET_LENGTH);
    }

    LOG_DEBUG(<< "normalised error moments = " << normalisedResiduals);
    BOOST_TEST_REQUIRE(
        std::fabs(maths::common::CBasicStatistics::mean(normalisedResiduals)) < 0.5);
    BOOST_TEST_REQUIRE(std::fabs(maths::common::CBasicStatistics::variance(normalisedResiduals) -
                                 1.0) < 0.2);
}

BOOST_AUTO_TEST_CASE(testDecayRate) {
    // Test that the trend short range predictions approximately
    // match a regression model with the same decay rate.

    test::CRandomNumbers rng;

    core_t::TTime start{0};
    core_t::TTime end{3000000};

    TDoubleVec values(multiscaleRandomWalk(rng, start, end));

    maths::time_series::CTrendComponent component{0.012};
    TRegression regression;
    maths::time_series::CDecayRateController controller(
        maths::time_series::CDecayRateController::E_PredictionBias |
            maths::time_series::CDecayRateController::E_PredictionErrorIncrease,
        1);

    TMeanAccumulator error;
    TMeanAccumulator level;
    for (core_t::TTime time = start; time < end; time += BUCKET_LENGTH) {
        double value{values[(time - start) / BUCKET_LENGTH]};
        component.add(time, value);
        regression.add(static_cast<double>(time) / 604800.0, value);

        double expectedPrediction{regression.predict(static_cast<double>(time) / 604800.0)};
        double prediction{component.value(time, 0.0).mean()};
        error.add(std::fabs(prediction - expectedPrediction));
        level.add(value);

        controller.multiplier({prediction},
                              {{values[(time - start) / BUCKET_LENGTH] - prediction}},
                              BUCKET_LENGTH, 1.0, 0.012);
        component.decayRate(0.012 * controller.multiplier());
        component.propagateForwardsByTime(BUCKET_LENGTH);
        regression.age(std::exp(-0.012 * controller.multiplier() * 600.0 / 86400.0));
    }

    double relativeError{maths::common::CBasicStatistics::mean(error) /
                         std::fabs(maths::common::CBasicStatistics::mean(level))};
    LOG_DEBUG(<< "relative error = " << relativeError);
    BOOST_REQUIRE_SMALL(relativeError, 0.5);
}

BOOST_AUTO_TEST_CASE(testForecast) {
    // Check the forecast errors for a variety of signals.

    test::CRandomNumbers rng;

    maths::time_series::CTrendComponent component{0.012};
    TDoubleVec values;
    core_t::TTime startForecast;
    double error;
    double errorAt95;

    LOG_DEBUG(<< "Random Walk");
    values = multiscaleRandomWalk(rng, 0, 3000000 + 1000 * BUCKET_LENGTH);
    std::tie(component, startForecast) =
        trainModel(values.begin(), values.begin() + 3000000 / BUCKET_LENGTH);
    std::tie(error, errorAt95) = forecastErrors(values.begin() + 3000000 / BUCKET_LENGTH,
                                                values.end(), startForecast, component);
    BOOST_TEST_REQUIRE(error < 0.17);
    BOOST_TEST_REQUIRE(errorAt95 < 0.001);

    LOG_DEBUG(<< "Piecewise Linear");
    values = piecewiseLinear(rng, 0, 3200000 + 1000 * BUCKET_LENGTH);
    std::tie(component, startForecast) =
        trainModel(values.begin(), values.begin() + 3200000 / BUCKET_LENGTH);
    std::tie(error, errorAt95) = forecastErrors(values.begin() + 3200000 / BUCKET_LENGTH,
                                                values.end(), startForecast, component);
    BOOST_TEST_REQUIRE(error < 0.03);
    BOOST_TEST_REQUIRE(errorAt95 < 0.001);

    LOG_DEBUG(<< "Staircase");
    values = staircase(rng, 0, 2000000 + 1000 * BUCKET_LENGTH);
    std::tie(component, startForecast) =
        trainModel(values.begin(), values.begin() + 2000000 / BUCKET_LENGTH);
    std::tie(error, errorAt95) = forecastErrors(values.begin() + 2000000 / BUCKET_LENGTH,
                                                values.end(), startForecast, component);
    BOOST_TEST_REQUIRE(error < 0.15);
    BOOST_TEST_REQUIRE(errorAt95 < 0.08);

    LOG_DEBUG(<< "Switching");
    values = switching(rng, 0, 3000000 + 1000 * BUCKET_LENGTH);
    std::tie(component, startForecast) =
        trainModel(values.begin(), values.begin() + 3000000 / BUCKET_LENGTH);
    std::tie(error, errorAt95) = forecastErrors(values.begin() + 3000000 / BUCKET_LENGTH,
                                                values.end(), startForecast, component);
    BOOST_TEST_REQUIRE(error < 0.14);
    BOOST_TEST_REQUIRE(errorAt95 < 0.001);
}

BOOST_AUTO_TEST_CASE(testStepChangeForecasting) {
    // A randomized test that forecasts of time series with step changes
    // don't explode. We previously sometimes ran into issues when we
    // extrapolated the feature distributions we use to predict steps.
    // In such cases we would predict far too many steps leading to
    // overly wide forecast bounds and unrealistic predictions.

    using TSizeVec = std::vector<std::size_t>;

    test::CRandomNumbers rng;
    double interval{20.0};

    maths::time_series::CTrendComponent::TFloatMeanAccumulatorVec values;

    for (std::size_t t = 0; t < 100; ++t) {
        TSizeVec changePoints;
        rng.generateUniformSamples(0, 1000, 6, changePoints);
        std::sort(changePoints.begin(), changePoints.end());
        changePoints.push_back(1000);
        TDoubleVec levels;
        rng.generateUniformSamples(-0.5 * interval, 0.5 * interval, 7, levels);

        maths::time_series::CTrendComponent trendModel{0.012};

        TDoubleVec noise;
        auto level = levels.begin();
        auto changePoint = changePoints.begin();
        core_t::TTime time{1672531200};
        for (std::size_t i = 0; i < 1000; ++i, time += BUCKET_LENGTH) {
            rng.generateNormalSamples(0.0, 0.25, 1, noise);
            double value{*level + noise[0]};
            trendModel.add(time, value);
            values.emplace_back().add(value);
            if (i == *changePoint) {
                ++level;
                ++changePoint;
                double shift{*level - *(level - 1)};
                core_t::TTime valuesStartTime{
                    time - static_cast<core_t::TTime>(values.size()) * BUCKET_LENGTH};
                TSizeVec segments{0, *changePoint - *(changePoint - 1) - 1,
                                  *changePoint - *(changePoint - 1)};
                TDoubleVec shifts{0.0, *level - *(level - 1)};
                trendModel.shiftLevel(shift, valuesStartTime, BUCKET_LENGTH,
                                      values, segments, shifts);
                values.clear();
            } else {
                trendModel.dontShiftLevel(time, value);
            }
        }

        TDouble3VecVec forecast;
        trendModel.forecast(time, time + 200 * BUCKET_LENGTH, BUCKET_LENGTH, 90.0, false,
                            [](core_t::TTime) { return TDouble3Vec(3, 0.0); },
                            [&forecast](core_t::TTime, const TDouble3Vec& value) {
                                forecast.push_back(value);
                            });

        // Check that the prediction is in the switching interval and
        // the forecast confidence interval isn't too wide.
        BOOST_TEST_REQUIRE(forecast.back()[1] > -0.75 * interval);
        BOOST_TEST_REQUIRE(forecast.back()[1] < 0.75 * interval);
        BOOST_TEST_REQUIRE(forecast.back()[2] - forecast.back()[0] < 3.5 * interval);
    }
}

BOOST_AUTO_TEST_CASE(testPersist) {
    // Check that serialization is idempotent.

    test::CRandomNumbers rng;

    core_t::TTime start{1200};
    core_t::TTime end{200000};

    TDoubleVec values(multiscaleRandomWalk(rng, start, end));

    maths::time_series::CTrendComponent origComponent{0.012};

    for (core_t::TTime time = start; time < end; time += BUCKET_LENGTH) {
        double value{values[(time - start) / BUCKET_LENGTH]};
        origComponent.add(time, value);
        origComponent.propagateForwardsByTime(BUCKET_LENGTH);
    }

    std::ostringstream origJson;
    core::CJsonStatePersistInserter::persist(
        origJson, std::bind(&maths::time_series::CTrendComponent::acceptPersistInserter,
                            &origComponent, std::placeholders::_1));

    LOG_DEBUG(<< "decomposition JSON representation:\n" << origJson.str());

    std::istringstream origJsonStrm{"{\"topLevel\" : " + origJson.str() + "}"};
    core::CJsonStateRestoreTraverser traverser(origJsonStrm);
    maths::common::SDistributionRestoreParams params{maths_t::E_ContinuousData, 0.1};

    maths::time_series::CTrendComponent restoredComponent{0.1};
    traverser.traverseSubLevel([&](auto& traverser_) {
        return restoredComponent.acceptRestoreTraverser(params, traverser_);
    });

    BOOST_REQUIRE_EQUAL(origComponent.checksum(), restoredComponent.checksum());

    std::ostringstream newJson;
    core::CJsonStatePersistInserter::persist(
        newJson, std::bind(&maths::time_series::CTrendComponent::acceptPersistInserter,
                           &restoredComponent, std::placeholders::_1));
    BOOST_REQUIRE_EQUAL(origJson.str(), newJson.str());
}

BOOST_AUTO_TEST_SUITE_END()
}
