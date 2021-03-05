/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CoreTypes.h>

#include <maths/CDecayRateController.h>
#include <maths/CIntegerTools.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CModel.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CSpline.h>
#include <maths/CTimeSeriesDecomposition.h>
#include <maths/CTimeSeriesModel.h>

#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include "TestUtils.h"

#include <boost/numeric/conversion/bounds.hpp>
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <functional>
#include <memory>

BOOST_AUTO_TEST_SUITE(CForecastTest)

using namespace ml;
using namespace handy_typedefs;

namespace {
using TTrend = std::function<double(core_t::TTime, double)>;
using TDoubleVec = std::vector<double>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble2VecWeightsAryVec = std::vector<maths_t::TDouble2VecWeightsAry>;
using TTimeDouble2VecSizeTr = core::CTriple<core_t::TTime, TDouble2Vec, std::size_t>;
using TTimeDouble2VecSizeTrVec = std::vector<TTimeDouble2VecSizeTr>;
using TErrorBarVec = std::vector<maths::SErrorBar>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TModelPtr = std::shared_ptr<maths::CModel>;

class CDebugGenerator {
public:
    static const bool ENABLED{false};

public:
    ~CDebugGenerator() {
        if (ENABLED) {
            std::ofstream file;
            file.open("results.py");
            file << "import matplotlib.pyplot as plt;\n";
            file << "t = " << core::CContainerPrinter::print(m_ValueTimes) << ";\n";
            file << "f = " << core::CContainerPrinter::print(m_Values) << ";\n";
            file << "tp = " << core::CContainerPrinter::print(m_PredictionTimes) << ";\n";
            file << "fp = " << core::CContainerPrinter::print(m_Predictions) << ";\n";
            file << "tf = " << core::CContainerPrinter::print(m_ForecastTimes) << ";\n";
            file << "fl = " << core::CContainerPrinter::print(m_ForecastLower) << ";\n";
            file << "fm = " << core::CContainerPrinter::print(m_ForecastMean) << ";\n";
            file << "fu = " << core::CContainerPrinter::print(m_ForecastUpper) << ";\n";
            file << "plt.plot(t, f);\n";
            file << "plt.plot(tp, fp, 'k');\n";
            file << "plt.plot(tf, fl, 'r');\n";
            file << "plt.plot(tf, fm, 'k');\n";
            file << "plt.plot(tf, fu, 'r');\n";
            file << "plt.show();\n";
        }
    }
    void addValue(core_t::TTime time, double value) {
        if (ENABLED) {
            m_ValueTimes.push_back(time);
            m_Values.push_back(value);
        }
    }
    void addPrediction(core_t::TTime time, double prediction) {
        if (ENABLED) {
            m_PredictionTimes.push_back(time);
            m_Predictions.push_back(prediction);
        }
    }
    void addForecast(core_t::TTime time, const maths::SErrorBar& forecast) {
        if (ENABLED) {
            m_ForecastTimes.push_back(time);
            m_ForecastLower.push_back(forecast.s_LowerBound);
            m_ForecastMean.push_back(forecast.s_Predicted);
            m_ForecastUpper.push_back(forecast.s_UpperBound);
        }
    }

private:
    TTimeVec m_ValueTimes;
    TDoubleVec m_Values;
    TTimeVec m_PredictionTimes;
    TDoubleVec m_Predictions;
    TTimeVec m_ForecastTimes;
    TDoubleVec m_ForecastLower;
    TDoubleVec m_ForecastMean;
    TDoubleVec m_ForecastUpper;
};

const double DECAY_RATE{0.0005};
const std::size_t TAG{0};
const TDouble2Vec MINIMUM_VALUE{boost::numeric::bounds<double>::lowest()};
const TDouble2Vec MAXIMUM_VALUE{boost::numeric::bounds<double>::highest()};

maths::CModelParams params(core_t::TTime bucketLength) {
    using TTimeDoubleMap = std::map<core_t::TTime, double>;
    static TTimeDoubleMap learnRates;
    learnRates[bucketLength] = static_cast<double>(bucketLength) / 1800.0;
    double minimumSeasonalVarianceScale{0.25};
    return maths::CModelParams{bucketLength,
                               learnRates[bucketLength],
                               DECAY_RATE,
                               minimumSeasonalVarianceScale,
                               6 * core::constants::HOUR,
                               core::constants::DAY};
}

maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary decayRateControllers() {
    return {{maths::CDecayRateController(maths::CDecayRateController::E_PredictionBias |
                                             maths::CDecayRateController::E_PredictionErrorIncrease,
                                         1),
             maths::CDecayRateController(maths::CDecayRateController::E_PredictionBias |
                                             maths::CDecayRateController::E_PredictionErrorIncrease |
                                             maths::CDecayRateController::E_PredictionErrorDecrease,
                                         1)}};
}

void mockSink(maths::SErrorBar errorBar, TErrorBarVec& prediction) {
    prediction.push_back(errorBar);
}

class CTest {
public:
    CTest& bucketLength(core_t::TTime value) {
        m_BucketLength = value;
        return *this;
    }
    CTest& daysToLearn(std::size_t value) {
        m_DaysToLearn = value;
        return *this;
    }
    CTest& noiseVariance(double value) {
        m_NoiseVariance = value;
        return *this;
    }
    // Note that if the confidence intervals are perfectly calibrated we expect 20%
    // out-of-bounds since we ask for 80% confidence interval. In practice, we tend
    // to get somewhat lower % since we allow for the parameters to change over the
    // forecast interval.
    CTest& maximumPercentageOutOfBounds(double value) {
        m_MaximumPercentageOutOfBounds = value;
        return *this;
    }
    CTest& maximumError(double value) {
        m_MaximumError = value;
        return *this;
    }

    void run(TTrend trend) const {
        LOG_DEBUG(<< "*** learn ***");

        test::CRandomNumbers rng;
        maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary controllers{
            decayRateControllers()};
        maths::CUnivariateTimeSeriesModel model(
            params(m_BucketLength), TAG,
            maths::CTimeSeriesDecomposition(0.012, m_BucketLength),
            maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, DECAY_RATE),
            &controllers);
        CDebugGenerator debug;

        core_t::TTime time{0};
        TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
        for (std::size_t d = 0; d < m_DaysToLearn; ++d) {
            TDoubleVec noise;
            rng.generateNormalSamples(0.0, m_NoiseVariance,
                                      core::constants::DAY / m_BucketLength, noise);

            for (std::size_t i = 0; i < noise.size(); ++i, time += m_BucketLength) {
                maths::CModelAddSamplesParams params;
                params.integer(false)
                    .propagationInterval(1.0)
                    .trendWeights(weights)
                    .priorWeights(weights);
                double yi{trend(time, noise[i])};
                model.addSamples(params, {core::make_triple(time, TDouble2Vec{yi}, TAG)});
                debug.addValue(time, yi);
                debug.addPrediction(
                    time, maths::CBasicStatistics::mean(model.predict(time)));
            }
        }

        LOG_DEBUG(<< "*** forecast ***");

        TErrorBarVec prediction;
        core_t::TTime start{time};
        core_t::TTime end{time + 2 * core::constants::WEEK};
        TModelPtr forecastModel(model.cloneForForecast());
        std::string m;
        forecastModel->forecast(
            0, start, start, end, 80.0, MINIMUM_VALUE, MAXIMUM_VALUE,
            std::bind(&mockSink, std::placeholders::_1, std::ref(prediction)), m);

        std::size_t outOfBounds{0};
        std::size_t count{0};
        TMeanAccumulator error;

        for (std::size_t i = 0; i < prediction.size(); /**/) {
            TDoubleVec noise;
            rng.generateNormalSamples(0.0, m_NoiseVariance,
                                      core::constants::DAY / m_BucketLength, noise);
            TDoubleVec day;
            for (std::size_t j = 0; i < prediction.size() && j < noise.size();
                 ++i, ++j, time += m_BucketLength) {
                double yj{trend(time, noise[j])};
                day.push_back(yj);
                outOfBounds +=
                    (yj < prediction[i].s_LowerBound || yj > prediction[i].s_UpperBound ? 1 : 0);
                ++count;
                error.add(std::fabs(yj - prediction[i].s_Predicted) / std::fabs(yj));
                debug.addValue(time, yj);
                debug.addForecast(time, prediction[i]);
            }
        }

        double percentageOutOfBounds{100.0 * static_cast<double>(outOfBounds) /
                                     static_cast<double>(count)};
        LOG_DEBUG(<< "% out of bounds = " << percentageOutOfBounds);
        LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));

        BOOST_TEST_REQUIRE(percentageOutOfBounds < m_MaximumPercentageOutOfBounds);
        BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < m_MaximumError);
    }

private:
    core_t::TTime m_BucketLength = 0;
    std::size_t m_DaysToLearn = 0;
    double m_NoiseVariance = 0.0;
    double m_MaximumPercentageOutOfBounds = 20.0;
    double m_MaximumError = 0.0;
};
}

BOOST_AUTO_TEST_CASE(testDailyNoLongTermTrend) {
    core_t::TTime bucketLength{600};
    TDoubleVec y{0.0,   2.0,   2.0,   4.0,   8.0,  10.0,  15.0,  20.0,
                 120.0, 120.0, 110.0, 100.0, 90.0, 100.0, 130.0, 80.0,
                 30.0,  15.0,  10.0,  8.0,   5.0,  3.0,   2.0,   0.0};

    test::CRandomNumbers rng;

    TTrend trend = [&y, bucketLength](core_t::TTime time, double noise) {
        core_t::TTime i{(time % core::constants::DAY) / bucketLength};
        double alpha{static_cast<double>(i % 6) / 6.0};
        double beta{1.0 - alpha};
        return 40.0 + alpha * y[i / 6] + beta * y[(i / 6 + 1) % y.size()] + noise;
    };

    CTest test;
    test.bucketLength(bucketLength)
        .daysToLearn(63)
        .noiseVariance(64.0)
        .maximumPercentageOutOfBounds(12.0)
        .maximumError(0.14)
        .run(trend);
}

BOOST_AUTO_TEST_CASE(testDailyConstantLongTermTrend) {
    core_t::TTime bucketLength{3600};
    TDoubleVec y{0.0,  2.0,   2.0,   4.0,   8.0,   10.0,  15.0, 20.0,
                 80.0, 100.0, 110.0, 120.0, 110.0, 100.0, 90.0, 80.0,
                 30.0, 15.0,  10.0,  8.0,   5.0,   3.0,   2.0,  0.0};

    TTrend trend = [&y, bucketLength](core_t::TTime time, double noise) {
        core_t::TTime i{(time % core::constants::DAY) / bucketLength};
        return 0.25 * static_cast<double>(time) / static_cast<double>(bucketLength) +
               y[i] + noise;
    };

    CTest test;
    test.bucketLength(bucketLength)
        .daysToLearn(63)
        .noiseVariance(64.0)
        .maximumPercentageOutOfBounds(13.0)
        .maximumError(0.016)
        .run(trend);
}

BOOST_AUTO_TEST_CASE(testDailyVaryingLongTermTrend) {
    core_t::TTime bucketLength{1800};
    double day{static_cast<double>(core::constants::DAY)};
    TDoubleVec times{0.0,         5.0 * day,   10.0 * day,  15.0 * day,
                     20.0 * day,  25.0 * day,  30.0 * day,  35.0 * day,
                     40.0 * day,  45.0 * day,  50.0 * day,  55.0 * day,
                     60.0 * day,  65.0 * day,  70.0 * day,  75.0 * day,
                     80.0 * day,  85.0 * day,  90.0 * day,  95.0 * day,
                     100.0 * day, 105.0 * day, 110.0 * day, 115.0 * day};
    TDoubleVec values{20.0, 30.0, 25.0, 35.0, 45.0, 40.0,  38.0,  36.0,
                      35.0, 25.0, 35.0, 45.0, 55.0, 62.0,  70.0,  76.0,
                      79.0, 82.0, 86.0, 90.0, 95.0, 100.0, 106.0, 112.0};

    maths::CSpline<> trend_(maths::CSplineTypes::E_Cubic);
    trend_.interpolate(times, values, maths::CSplineTypes::E_Natural);

    TTrend trend = [&trend_](core_t::TTime time, double noise) {
        double time_{static_cast<double>(time)};
        return trend_.value(time_) +
               8.0 * std::sin(boost::math::double_constants::two_pi * time_ / 43200.0) + noise;
    };

    CTest test;
    test.bucketLength(bucketLength)
        .daysToLearn(98)
        .noiseVariance(9.0)
        .maximumPercentageOutOfBounds(5.0)
        .maximumError(0.04)
        .run(trend);
}

BOOST_AUTO_TEST_CASE(testComplexNoLongTermTrend) {
    core_t::TTime bucketLength{3600};
    TDoubleVec y{0.0,  10.0,  20.0,  20.0,  30.0,  40.0,  50.0, 60.0,
                 80.0, 100.0, 110.0, 120.0, 110.0, 100.0, 90.0, 80.0,
                 60.0, 40.0,  30.0,  20.0,  10.0,  10.0,  5.0,  0.0};
    TDoubleVec scale{1.0, 1.1, 1.05, 0.95, 0.9, 0.3, 0.2};

    TTrend trend = [&y, &scale, bucketLength](core_t::TTime time, double noise) {
        core_t::TTime d{(time % core::constants::WEEK) / core::constants::DAY};
        core_t::TTime h{(time % core::constants::DAY) / bucketLength};
        return scale[d] * (20.0 + y[h] + noise);
    };

    CTest test;
    test.bucketLength(bucketLength)
        .daysToLearn(63)
        .noiseVariance(4.0)
        .maximumPercentageOutOfBounds(9.0)
        .maximumError(0.06)
        .run(trend);
}

BOOST_AUTO_TEST_CASE(testComplexConstantLongTermTrend) {
    core_t::TTime bucketLength{3600};
    TDoubleVec y{0.0,  10.0,  20.0,  20.0,  30.0,  40.0,  50.0, 60.0,
                 80.0, 100.0, 110.0, 120.0, 110.0, 100.0, 90.0, 80.0,
                 60.0, 40.0,  30.0,  20.0,  10.0,  10.0,  5.0,  0.0};
    TDoubleVec scale{1.0, 1.1, 1.05, 0.95, 0.9, 0.3, 0.2};

    TTrend trend = [&y, &scale, bucketLength](core_t::TTime time, double noise) {
        core_t::TTime d{(time % core::constants::WEEK) / core::constants::DAY};
        core_t::TTime h{(time % core::constants::DAY) / bucketLength};
        return 0.25 * static_cast<double>(time) / static_cast<double>(bucketLength) +
               scale[d] * (20.0 + y[h] + noise);
    };

    CTest test;
    test.bucketLength(bucketLength)
        .daysToLearn(63)
        .noiseVariance(4.0)
        .maximumPercentageOutOfBounds(7.5)
        .maximumError(0.02)
        .run(trend);
}

BOOST_AUTO_TEST_CASE(testComplexVaryingLongTermTrend) {
    core_t::TTime bucketLength{1800};
    double day{static_cast<double>(core::constants::DAY)};
    TDoubleVec times{0.0,         5.0 * day,   10.0 * day,  15.0 * day,
                     20.0 * day,  25.0 * day,  30.0 * day,  35.0 * day,
                     40.0 * day,  45.0 * day,  50.0 * day,  55.0 * day,
                     60.0 * day,  65.0 * day,  70.0 * day,  75.0 * day,
                     80.0 * day,  85.0 * day,  90.0 * day,  95.0 * day,
                     100.0 * day, 105.0 * day, 110.0 * day, 115.0 * day};
    TDoubleVec values{20.0, 30.0, 25.0, 35.0, 45.0, 40.0,  38.0,  36.0,
                      35.0, 34.0, 35.0, 40.0, 48.0, 55.0,  65.0,  76.0,
                      79.0, 82.0, 86.0, 90.0, 95.0, 100.0, 106.0, 112.0};
    TDoubleVec y{0.0, 1.0,  2.0,  2.0,  3.0,  4.0,  5.0, 6.0,
                 8.0, 10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 8.0,
                 6.0, 4.0,  3.0,  2.0,  1.0,  1.0,  0.5, 0.0};
    TDoubleVec scale{1.0, 1.1, 1.05, 0.95, 0.9, 0.3, 0.2};

    maths::CSpline<> trend_(maths::CSplineTypes::E_Cubic);
    trend_.interpolate(times, values, maths::CSplineTypes::E_Natural);

    TTrend trend = [&trend_, &y, &scale](core_t::TTime time, double noise) {
        core_t::TTime d{(time % core::constants::WEEK) / core::constants::DAY};
        core_t::TTime h{(time % core::constants::DAY) / core::constants::HOUR};
        double time_{static_cast<double>(time)};
        return trend_.value(time_) + scale[d] * (20.0 + y[h] + noise);
    };

    CTest test;
    test.bucketLength(bucketLength)
        .daysToLearn(98)
        .noiseVariance(4.0)
        .maximumPercentageOutOfBounds(28.0)
        .maximumError(0.06)
        .run(trend);
}

BOOST_AUTO_TEST_CASE(testNonNegative) {
    core_t::TTime bucketLength{1800};

    test::CRandomNumbers rng;

    maths::CTimeSeriesDecomposition trend(0.012, bucketLength);
    maths::CNormalMeanPrecConjugate prior = maths::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, DECAY_RATE);
    maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary controllers{
        decayRateControllers()};
    maths::CUnivariateTimeSeriesModel model(params(bucketLength), TAG, trend,
                                            prior, &controllers);
    CDebugGenerator debug;

    LOG_DEBUG(<< "*** learn ***");

    core_t::TTime time{0};
    std::vector<maths_t::TDouble2VecWeightsAry> weights{
        maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
    for (std::size_t d = 0; d < 20; ++d) {
        TDoubleVec noise;
        rng.generateNormalSamples(2.0, 3.0, 48, noise);
        for (auto value = noise.begin(); value != noise.end(); ++value, time += bucketLength) {
            maths::CModelAddSamplesParams params;
            params.integer(false)
                .nonNegative(true)
                .propagationInterval(1.0)
                .trendWeights(weights)
                .priorWeights(weights);
            double y{std::max(*value, 0.0)};
            model.addSamples(params, {core::make_triple(time, TDouble2Vec{y}, TAG)});
            debug.addValue(time, y);
            debug.addPrediction(time, maths::CBasicStatistics::mean(model.predict(time)));
        }
    }

    LOG_DEBUG(<< "*** forecast ***");

    TErrorBarVec prediction;
    core_t::TTime start{time};
    core_t::TTime end{time + 20 * core::constants::DAY};
    std::string m;
    TModelPtr forecastModel(model.cloneForForecast());
    forecastModel->forecast(
        0, start, start, end, maths::CModel::DEFAULT_BOUNDS_PERCENTILE,
        MINIMUM_VALUE, MAXIMUM_VALUE,
        std::bind(&mockSink, std::placeholders::_1, std::ref(prediction)), m);

    std::size_t outOfBounds{0};
    std::size_t count{0};

    for (std::size_t i = 0; i < prediction.size(); ++i) {
        TDoubleVec noise;
        rng.generateNormalSamples(2.0, 3.0, 48, noise);
        for (auto value = noise.begin(); i < prediction.size() && value != noise.end();
             ++i, ++value, time += bucketLength) {
            BOOST_TEST_REQUIRE(prediction[i].s_LowerBound >= 0);
            BOOST_TEST_REQUIRE(prediction[i].s_Predicted >= 0);
            BOOST_TEST_REQUIRE(prediction[i].s_UpperBound >= 0);

            double y{std::max(*value, 0.0)};
            outOfBounds +=
                (y < prediction[i].s_LowerBound || y > prediction[i].s_UpperBound ? 1 : 0);
            ++count;
            debug.addValue(time, y);
            debug.addForecast(time, prediction[i]);
        }
    }

    double percentageOutOfBounds{100.0 * static_cast<double>(outOfBounds) /
                                 static_cast<double>(count)};
    LOG_DEBUG(<< "% out of bounds = " << percentageOutOfBounds);

    BOOST_TEST_REQUIRE(percentageOutOfBounds < 4.0);
}

BOOST_AUTO_TEST_CASE(testFinancialIndex) {
    core_t::TTime bucketLength{1800};

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    BOOST_TEST_REQUIRE(test::CTimeSeriesTestData::parse(
        "testfiles/financial_index.csv", timeseries, startTime, endTime, "^([0-9]+),([0-9\\.]+)"));
    BOOST_TEST_REQUIRE(timeseries.empty() == false);

    LOG_DEBUG(<< "timeseries = "
              << core::CContainerPrinter::print(timeseries.begin(), timeseries.begin() + 10)
              << " ...");

    maths::CTimeSeriesDecomposition trend(0.012, bucketLength);
    maths::CNormalMeanPrecConjugate prior = maths::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, DECAY_RATE);
    maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary controllers{
        decayRateControllers()};
    maths::CUnivariateTimeSeriesModel model(params(bucketLength), TAG, trend,
                                            prior, &controllers);
    CDebugGenerator debug;

    LOG_DEBUG(<< "*** learn ***");

    std::size_t n{5 * timeseries.size() / 6};

    TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};
    for (std::size_t i = 0; i < n; ++i) {
        maths::CModelAddSamplesParams params;
        params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
        model.addSamples(
            params, {core::make_triple(timeseries[i].first,
                                       TDouble2Vec{timeseries[i].second}, TAG)});
        debug.addValue(timeseries[i].first, timeseries[i].second);
        debug.addPrediction(
            timeseries[i].first,
            maths::CBasicStatistics::mean(model.predict(timeseries[i].first)));
    }

    LOG_DEBUG(<< "*** forecast ***");

    TErrorBarVec prediction;
    core_t::TTime start{timeseries[n].first};
    core_t::TTime end{timeseries[timeseries.size() - 1].first};
    std::string m;
    TModelPtr forecastModel(model.cloneForForecast());
    forecastModel->forecast(
        startTime, start, start, end, 99.0, MINIMUM_VALUE, MAXIMUM_VALUE,
        std::bind(&mockSink, std::placeholders::_1, std::ref(prediction)), m);

    std::size_t outOfBounds{0};
    std::size_t count{0};
    TMeanAccumulator error;

    for (std::size_t i = n, j = 0;
         i < timeseries.size() && j < prediction.size(); ++i, ++j) {
        double yi{timeseries[i].second};
        outOfBounds +=
            (yi < prediction[j].s_LowerBound || yi > prediction[j].s_UpperBound ? 1 : 0);
        ++count;
        error.add(std::fabs(yi - prediction[j].s_Predicted) / std::fabs(yi));
        debug.addValue(timeseries[i].first, timeseries[i].second);
        debug.addForecast(timeseries[i].first, prediction[j]);
    }

    double percentageOutOfBounds{100.0 * static_cast<double>(outOfBounds) /
                                 static_cast<double>(count)};
    LOG_DEBUG(<< "% out of bounds = " << percentageOutOfBounds);
    LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));

    BOOST_TEST_REQUIRE(percentageOutOfBounds < 40.0);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(error) < 0.1);
}

BOOST_AUTO_TEST_CASE(testTruncation) {

    core_t::TTime bucketLength{1800};
    TDouble2VecWeightsAryVec weights{maths_t::CUnitWeights::unit<TDouble2Vec>(1)};

    for (auto dataEndTime : {core::constants::DAY, 20 * core::constants::DAY}) {

        maths::CTimeSeriesDecomposition trend(0.012, bucketLength);
        maths::CNormalMeanPrecConjugate prior = maths::CNormalMeanPrecConjugate::nonInformativePrior(
            maths_t::E_ContinuousData, DECAY_RATE);
        maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary controllers{
            decayRateControllers()};
        maths::CUnivariateTimeSeriesModel model(params(bucketLength), TAG,
                                                trend, prior, &controllers);

        for (core_t::TTime time = 0; time < dataEndTime; time += bucketLength) {
            maths::CModelAddSamplesParams params;
            params.integer(false).propagationInterval(1.0).trendWeights(weights).priorWeights(weights);
            double yi{static_cast<double>(time)};
            model.addSamples(params, {core::make_triple(time, TDouble2Vec{yi}, TAG)});
        }

        // Check truncation

        TErrorBarVec prediction;
        std::string m1;
        model.forecast(
            0, dataEndTime, dataEndTime, dataEndTime + 2 * core::constants::DAY,
            90.0, MINIMUM_VALUE, MAXIMUM_VALUE,
            std::bind(&mockSink, std::placeholders::_1, std::ref(prediction)), m1);
        LOG_DEBUG(<< "response = '" << m1 << "'");
        BOOST_TEST_REQUIRE((m1.size() > 0) == (dataEndTime < 2 * core::constants::DAY));
        BOOST_TEST_REQUIRE(prediction.size() > 0);
        BOOST_TEST_REQUIRE(prediction.back().s_Time < 2 * dataEndTime);

        // Check forecast range out-of-bounds

        prediction.clear();
        std::string m2;
        model.forecast(
            0, dataEndTime, dataEndTime + 30 * core::constants::DAY,
            dataEndTime + 40 * core::constants::DAY, 90.0, MINIMUM_VALUE, MAXIMUM_VALUE,
            std::bind(&mockSink, std::placeholders::_1, std::ref(prediction)), m2);
        LOG_DEBUG(<< "response = '" << m2 << "'");
        BOOST_TEST_REQUIRE(m2.empty() == false);
        BOOST_TEST_REQUIRE(m1 != m2);
        BOOST_TEST_REQUIRE(prediction.empty());
    }
}

BOOST_AUTO_TEST_SUITE_END()
