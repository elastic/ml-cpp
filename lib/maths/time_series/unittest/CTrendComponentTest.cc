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
#include <core/Constants.h>
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

TDouble3VecVec forecastValues(const maths::time_series::CTrendComponent& component,
                              core_t::TTime start,
                              core_t::TTime interval) {
    TDouble3VecVec result;
    component.forecast(start, start + interval, BUCKET_LENGTH, 95.0, false,
                       [](core_t::TTime) { return TDouble3Vec(3, 0.0); },
                       [&result](core_t::TTime, const TDouble3Vec& value) {
                           result.push_back(value);
                       });
    return result;
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

BOOST_AUTO_TEST_CASE(testForecastAfterTemporaryDrop) {
    // Regression for elastic/ml-cpp#2772. This is the isolated trend state for
    // the affected percentage metric; the full customer model state is not
    // needed to reproduce the runaway forecast.
    const std::string state{R"({"a":"0.431100","b":"1719821532","c":"1724549149","d":"1724284800","e":{"7.1":"","a":"250.479599:0.147902533","b":{"a":"8.06102753:1,0.37328211047068233,0.14105012085650631,0.053787408052162582,0.020661131796887847,0.85000006431035102,0.31715800105897174,0.11979260726141326"},"c":"21.1509151:0.0005455305792922573,0.00054112481310964833,0.00053581286394443475"},"e":{"7.1":"","a":"250.479599:0.295805037","b":{"a":"21.1509151:1,0.33215533650146561,0.11703279402852568,0.042270298480796595,0.015581801328682397,0.85148315534083596,0.28270303236397781,0.09957293288043953"},"c":"47.9369965:0.00029250961809147485,0.00028017050058775522,0.00027538121287162256"},"e":{"7.1":"","a":"250.479599:0.526017249","b":{"a":"47.9369965:1,0.25512903552999755,0.089623992769172545,0.027458695894191861,0.011810625854069556,0.85171140682578828,0.21770040684058509,0.076054678533887018"},"c":"99.096405:0.00038084258443694632,0.00025448879813878768,0.0002027623237618853"},"e":{"7.1":"","a":"250.479599:0.545107722","b":{"a":"149.5289:1,-0.067857738998839554,0.26265235133845866,-0.31958018108907954,0.66184621665902565,0.82949782905180303,-0.031475171584892062,0.18032231216902042"},"c":"304.828003:0.0067217006731924078,0.0014471298922769367,0.00093666246244155239"},"e":{"7.1":"","a":"250.479599:0.318530947","b":{"a":"460.967804:1,-1.0519092806127064,3.0049065013656149,-11.027589371621847,49.311893119082001,0.68395954956407923,-0.37799222292729512,0.53565179375735417"},"c":"871.070251:0.067334228268369989,0.0080346444347579263,0.0059031585621117186"},"e":{"7.1":"","a":"250.479599:0.081665419","b":{"a":"1378.00476:1,-2.6519394382590891,11.587463535185339,-59.335884960678762,333.05381708093898,0.38507297594864454,-0.178276008012313,-0.66295740146638527"},"c":"1815.88391:0.15143886325568157,0.015523240531493367,0.012581268734747606"},"e":{"7.1":"","a":"250.479599:0.0204163548","b":{"a":"2111.20581:1,-3.2540533955795321,15.529479742310103,-84.156038814723601,490.41583001013078,0.27535778764295005,-0.011875762868824316,-1.4452510597651793"},"c":"2283.84741:0.17050434863857433,0.019148988230895836,0.014572939799648845"},"e":{"7.1":"","a":"250.479599:0.00408327067","b":{"a":"2396.6521:1,-3.4206889516152468,16.67502492985405,-91.579499722634083,538.43923018928899,0.24570856625052842,0.039130784809345238,-1.6837936619345601"},"c":"2435.90601:0.1744073726021757,0.020301972603092477,0.015104627372535553"},"f":"0.00044498652947708639","g":"2476.01978:0.690303862:0.0145720355","h":"1722770824","i":{"b":"0","c":{"e":"1715.63477","f":{"a":{"d":{"h":"0.00120000006","a":"813662.312","b":"1694.42847","c":"848.214172","d":"189591364491369.41","e":"1694.42847"}}},"f":{"a":{"d":{"h":"0.00120000006","a":"0.69859004","b":"1694.42847","c":"848.214172","d":"19.469362799280617","e":"1694.42847"}}}},"b":"1","c":{"e":"0.990120709","f":{"a":{"d":{"h":"0.00120000006","a":"1310413","b":"1","c":"1.5","d":"0","e":"1"}}},"f":{"a":{"d":{"h":"0.00120000006","a":"0.864572227","b":"1","c":"1.5","d":"0","e":"1"}}}}},"j":{"h":"0.0240000002","a":"-0.31222561","b":"2.13724494","c":"2.06862259","d":"0.094579347425700844","e":"1.04676807"}})"};
    std::istringstream stateStream{"{\"topLevel\":" + state + "}"};
    core::CJsonStateRestoreTraverser traverser{stateStream};
    maths::common::SDistributionRestoreParams params{maths_t::E_ContinuousData, 0.1};
    maths::time_series::CTrendComponent component{0.024};

    BOOST_REQUIRE(traverser.traverseSubLevel([&](auto& traverser_) {
        return component.acceptRestoreTraverser(params, traverser_);
    }));

    core_t::TTime startTime{1724549400};
    core_t::TTime endTime{startTime + 30 * core::constants::DAY};
    TDouble3VecVec forecast;
    component.forecast(startTime, endTime, 30 * core::constants::MINUTE, 95.0,
                       true, [](core_t::TTime) { return TDouble3Vec(3, 0.0); },
                       [&forecast](core_t::TTime, const TDouble3Vec& value) {
                           forecast.push_back(value);
                       });

    BOOST_REQUIRE(!forecast.empty());
    LOG_DEBUG(<< "First forecast = " << forecast.front()[1]
              << ", last forecast = " << forecast.back()[1]);
    for (const auto& value : forecast) {
        BOOST_TEST_REQUIRE(std::isfinite(value[0]));
        BOOST_TEST_REQUIRE(std::isfinite(value[1]));
        BOOST_TEST_REQUIRE(std::isfinite(value[2]));
        BOOST_TEST_REQUIRE(value[0] <= value[1]);
        BOOST_TEST_REQUIRE(value[1] <= value[2]);
    }
    BOOST_TEST_REQUIRE(std::fabs(forecast.back()[1] - forecast.front()[1]) <
                       0.2 * std::max(std::fabs(forecast.front()[1]), 1.0));
}

BOOST_AUTO_TEST_CASE(testForecastPreservesSupportedLinearTrendAndPrefix) {
    // A supported linear signal should keep its short, useful extrapolation;
    // splitting a request must not change the common prefix.
    for (double slope : {0.05, 0.2}) {
        TDoubleVec values;
        for (std::size_t i = 0; i < 2000; ++i) {
            values.push_back(10.0 + slope * static_cast<double>(i));
        }
        auto[component, start] = trainModel(values.begin(), values.end());
        auto day = forecastValues(component, start, core::constants::DAY);
        auto week = forecastValues(component, start, 7 * core::constants::DAY);
        auto month = forecastValues(component, start, 30 * core::constants::DAY);

        BOOST_REQUIRE_EQUAL(week.size(), 7 * core::constants::DAY / BUCKET_LENGTH);
        BOOST_REQUIRE(month.size() >= week.size());
        for (std::size_t i = 0; i < week.size(); ++i) {
            BOOST_REQUIRE_CLOSE(week[i][1], month[i][1], 1e-10);
        }

        double expectedChange{slope * static_cast<double>(day.size() - 1)};
        BOOST_TEST_REQUIRE(day.back()[1] - day.front()[1] > 0.5 * expectedChange);
    }
}

BOOST_AUTO_TEST_CASE(testForecastPreservesExactLinearTrendWithZeroUncertainty) {
    // An exact line with zero extrapolation uncertainty must still extrapolate.
    maths::time_series::CTrendComponent component{0.1};
    core_t::TTime time{0};
    for (std::size_t i = 0; i < 2000; ++i, time += BUCKET_LENGTH) {
        component.add(time, 10.0 + 0.2 * static_cast<double>(i));
        component.propagateForwardsByTime(BUCKET_LENGTH);
    }
    component.shiftOrigin(time);

    std::ostringstream state;
    core::CJsonStatePersistInserter::persist(
        state, std::bind_front(&maths::time_series::CTrendComponent::acceptPersistInserter,
                               &component));
    std::string zeroVarianceState{state.str()};
    std::size_t position{0};
    for (std::size_t i = 0; i < 8; ++i) {
        position = zeroVarianceState.find("\"e\":{\"7.1\"", position);
        BOOST_REQUIRE_NE(position, std::string::npos);
        position = zeroVarianceState.find("\"c\":\"", position);
        BOOST_REQUIRE_NE(position, std::string::npos);
        std::size_t end{zeroVarianceState.find('"', position + 5)};
        BOOST_REQUIRE_NE(end, std::string::npos);
        zeroVarianceState.replace(position + 5, end - position - 5, "250:1,0,0");
        position += 5 + std::string{"250:1,0,0"}.size();
    }

    std::istringstream stateStream{"{\"topLevel\":" + zeroVarianceState + "}"};
    core::CJsonStateRestoreTraverser traverser{stateStream};
    maths::common::SDistributionRestoreParams params{maths_t::E_ContinuousData, 0.1};
    maths::time_series::CTrendComponent restored{0.1};
    BOOST_REQUIRE(traverser.traverseSubLevel([&](auto& traverser_) {
        return restored.acceptRestoreTraverser(params, traverser_);
    }));

    auto forecast = forecastValues(restored, time, core::constants::DAY);

    BOOST_REQUIRE_EQUAL(forecast.size(), core::constants::DAY / BUCKET_LENGTH);
    BOOST_TEST_REQUIRE(forecast.back()[1] - forecast.front()[1] >
                       0.5 * 0.2 * static_cast<double>(forecast.size() - 1));
}

BOOST_AUTO_TEST_CASE(testForecastIsAffineInvariant) {
    TDoubleVec values;
    TDoubleVec transformed;
    for (std::size_t i = 0; i < 2000; ++i) {
        double value{4.0 + 0.1 * static_cast<double>(i) +
                     0.00001 * static_cast<double>(i * i)};
        values.push_back(value);
        transformed.push_back(100.0 * value - 37.0);
    }
    auto[component, start] = trainModel(values.begin(), values.end());
    auto[transformedComponent, transformedStart] =
        trainModel(transformed.begin(), transformed.end());
    auto forecast = forecastValues(component, start, 7 * core::constants::DAY);
    auto transformedForecast = forecastValues(transformedComponent, transformedStart,
                                              7 * core::constants::DAY);

    BOOST_REQUIRE_EQUAL(forecast.size(), transformedForecast.size());
    for (std::size_t i = 0; i < forecast.size(); ++i) {
        BOOST_REQUIRE_CLOSE(transformedForecast[i][1], 100.0 * forecast[i][1] - 37.0, 1e-8);
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
        origJson, std::bind_front(&maths::time_series::CTrendComponent::acceptPersistInserter,
                                  &origComponent));

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
        newJson, std::bind_front(&maths::time_series::CTrendComponent::acceptPersistInserter,
                                 &restoredComponent));
    BOOST_REQUIRE_EQUAL(origJson.str(), newJson.str());
}

BOOST_AUTO_TEST_SUITE_END()
}
