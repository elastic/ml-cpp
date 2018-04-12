/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
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

#include "CForecastTest.h"

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

#include <boost/bind.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>

using namespace ml;
using namespace handy_typedefs;

namespace {
using TDoubleVec = std::vector<double>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
using TDouble2Vec = core::CSmallVector<double, 2>;
using TDouble2Vec4Vec = core::CSmallVector<TDouble2Vec, 4>;
using TDouble2Vec4VecVec = std::vector<TDouble2Vec4Vec>;
using TTimeDouble2VecSizeTr = core::CTriple<core_t::TTime, TDouble2Vec, std::size_t>;
using TTimeDouble2VecSizeTrVec = std::vector<TTimeDouble2VecSizeTr>;
using TErrorBarVec = std::vector<maths::SErrorBar>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TModelPtr = boost::shared_ptr<maths::CModel>;

const double DECAY_RATE{0.0005};
const std::size_t TAG{0u};
const TDouble2Vec MINIMUM_VALUE{boost::numeric::bounds<double>::lowest()};
const TDouble2Vec MAXIMUM_VALUE{boost::numeric::bounds<double>::highest()};

maths::CModelParams params(core_t::TTime bucketLength) {
    using TTimeDoubleMap = std::map<core_t::TTime, double>;
    static TTimeDoubleMap learnRates;
    learnRates[bucketLength] = static_cast<double>(bucketLength) / 1800.0;
    double minimumSeasonalVarianceScale{0.25};
    return maths::CModelParams{bucketLength, learnRates[bucketLength],
                               DECAY_RATE, minimumSeasonalVarianceScale};
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
}

void mockSink(maths::SErrorBar errorBar, TErrorBarVec& prediction) {
    prediction.push_back(errorBar);
}

void CForecastTest::testDailyNoLongTermTrend() {
    LOG_DEBUG(<< "+-------------------------------------------+");
    LOG_DEBUG(<< "|  CForecastTest::testDailyNoLongTermTrend  |");
    LOG_DEBUG(<< "+-------------------------------------------+");

    core_t::TTime bucketLength{600};
    TDoubleVec y{0.0,   2.0,   2.0,   4.0,   8.0,  10.0,  15.0,  20.0,
                 120.0, 120.0, 110.0, 100.0, 90.0, 100.0, 130.0, 80.0,
                 30.0,  15.0,  10.0,  8.0,   5.0,  3.0,   2.0,   0.0};

    test::CRandomNumbers rng;

    auto trend = [&y, bucketLength](core_t::TTime time, double noise) {
        core_t::TTime i{(time % 86400) / bucketLength};
        double alpha{static_cast<double>(i % 6) / 6.0};
        double beta{1.0 - alpha};
        return 40.0 + alpha * y[i / 6] + beta * y[(i / 6 + 1) % y.size()] + noise;
    };

    this->test(trend, bucketLength, 60, 64.0, 4.0, 0.13);
}

void CForecastTest::testDailyConstantLongTermTrend() {
    LOG_DEBUG(<< "+-------------------------------------------------+");
    LOG_DEBUG(<< "|  CForecastTest::testDailyConstantLongTermTrend  |");
    LOG_DEBUG(<< "+-------------------------------------------------+");

    core_t::TTime bucketLength{3600};
    TDoubleVec y{0.0,  2.0,   2.0,   4.0,   8.0,   10.0,  15.0, 20.0,
                 80.0, 100.0, 110.0, 120.0, 110.0, 100.0, 90.0, 80.0,
                 30.0, 15.0,  10.0,  8.0,   5.0,   3.0,   2.0,  0.0};

    auto trend = [&y, bucketLength](core_t::TTime time, double noise) {
        core_t::TTime i{(time % 86400) / bucketLength};
        return 0.25 * static_cast<double>(time) / static_cast<double>(bucketLength) +
               y[i] + noise;
    };

    this->test(trend, bucketLength, 60, 64.0, 15.0, 0.02);
}

void CForecastTest::testDailyVaryingLongTermTrend() {
    LOG_DEBUG(<< "+------------------------------------------------+");
    LOG_DEBUG(<< "|  CForecastTest::testDailyVaryingLongTermTrend  |");
    LOG_DEBUG(<< "+------------------------------------------------+");

    core_t::TTime bucketLength{3600};
    double day{86400.0};
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

    auto trend = [&trend_](core_t::TTime time, double noise) {
        double time_{static_cast<double>(time)};
        return trend_.value(time_) +
               8.0 * std::sin(boost::math::double_constants::two_pi * time_ / 43200.0) + noise;
    };

    this->test(trend, bucketLength, 100, 9.0, 13.0, 0.04);
}

void CForecastTest::testComplexNoLongTermTrend() {
    LOG_DEBUG(<< "+---------------------------------------------+");
    LOG_DEBUG(<< "|  CForecastTest::testComplexNoLongTermTrend  |");
    LOG_DEBUG(<< "+---------------------------------------------+");

    core_t::TTime bucketLength{3600};
    TDoubleVec y{0.0,  10.0,  20.0,  20.0,  30.0,  40.0,  50.0, 60.0,
                 80.0, 100.0, 110.0, 120.0, 110.0, 100.0, 90.0, 80.0,
                 60.0, 40.0,  30.0,  20.0,  10.0,  10.0,  5.0,  0.0};
    TDoubleVec scale{1.0, 1.1, 1.05, 0.95, 0.9, 0.3, 0.2};

    auto trend = [&y, &scale, bucketLength](core_t::TTime time, double noise) {
        core_t::TTime d{(time % 604800) / 86400};
        core_t::TTime h{(time % 86400) / bucketLength};
        return scale[d] * (20.0 + y[h] + noise);
    };

    this->test(trend, bucketLength, 60, 24.0, 34.0, 0.13);
}

void CForecastTest::testComplexConstantLongTermTrend() {
    LOG_DEBUG(<< "+---------------------------------------------------+");
    LOG_DEBUG(<< "|  CForecastTest::testComplexConstantLongTermTrend  |");
    LOG_DEBUG(<< "+---------------------------------------------------+");

    core_t::TTime bucketLength{3600};
    TDoubleVec y{0.0,  10.0,  20.0,  20.0,  30.0,  40.0,  50.0, 60.0,
                 80.0, 100.0, 110.0, 120.0, 110.0, 100.0, 90.0, 80.0,
                 60.0, 40.0,  30.0,  20.0,  10.0,  10.0,  5.0,  0.0};
    TDoubleVec scale{1.0, 1.1, 1.05, 0.95, 0.9, 0.3, 0.2};

    auto trend = [&y, &scale, bucketLength](core_t::TTime time, double noise) {
        core_t::TTime d{(time % 604800) / 86400};
        core_t::TTime h{(time % 86400) / bucketLength};
        return 0.25 * static_cast<double>(time) / static_cast<double>(bucketLength) +
               scale[d] * (20.0 + y[h] + noise);
    };

    this->test(trend, bucketLength, 60, 24.0, 17.0, 0.04);
}

void CForecastTest::testComplexVaryingLongTermTrend() {
    LOG_DEBUG(<< "+--------------------------------------------------+");
    LOG_DEBUG(<< "|  CForecastTest::testComplexVaryingLongTermTrend  |");
    LOG_DEBUG(<< "+--------------------------------------------------+");

    core_t::TTime bucketLength{3600};
    double day{86400.0};
    TDoubleVec times{0.0,         5.0 * day,   10.0 * day,  15.0 * day,
                     20.0 * day,  25.0 * day,  30.0 * day,  35.0 * day,
                     40.0 * day,  45.0 * day,  50.0 * day,  55.0 * day,
                     60.0 * day,  65.0 * day,  70.0 * day,  75.0 * day,
                     80.0 * day,  85.0 * day,  90.0 * day,  95.0 * day,
                     100.0 * day, 105.0 * day, 110.0 * day, 115.0 * day};
    TDoubleVec values{20.0, 30.0, 25.0, 35.0, 45.0, 40.0,  38.0,  36.0,
                      35.0, 25.0, 35.0, 45.0, 55.0, 62.0,  70.0,  76.0,
                      79.0, 82.0, 86.0, 90.0, 95.0, 100.0, 106.0, 112.0};
    TDoubleVec y{0.0, 1.0,  2.0,  2.0,  3.0,  4.0,  5.0, 6.0,
                 8.0, 10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 8.0,
                 6.0, 4.0,  3.0,  2.0,  1.0,  1.0,  0.5, 0.0};
    TDoubleVec scale{1.0, 1.1, 1.05, 0.95, 0.9, 0.3, 0.2};

    maths::CSpline<> trend_(maths::CSplineTypes::E_Cubic);
    trend_.interpolate(times, values, maths::CSplineTypes::E_Natural);

    auto trend = [&trend_, &y, &scale, bucketLength](core_t::TTime time, double noise) {
        core_t::TTime d{(time % 604800) / 86400};
        core_t::TTime h{(time % 86400) / bucketLength};
        double time_{static_cast<double>(time)};
        return trend_.value(time_) + scale[d] * (20.0 + y[h] + noise);
    };

    this->test(trend, bucketLength, 60, 4.0, 23.0, 0.05);
}

void CForecastTest::testNonNegative() {
    LOG_DEBUG(<< "+----------------------------------+");
    LOG_DEBUG(<< "|  CForecastTest::testNonNegative  |");
    LOG_DEBUG(<< "+----------------------------------+");

    core_t::TTime bucketLength{1800};

    test::CRandomNumbers rng;

    maths::CTimeSeriesDecomposition trend(0.012, bucketLength);
    maths::CNormalMeanPrecConjugate prior = maths::CNormalMeanPrecConjugate::nonInformativePrior(
        maths_t::E_ContinuousData, DECAY_RATE);
    maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary controllers{
        decayRateControllers()};
    maths::CUnivariateTimeSeriesModel model(params(bucketLength), TAG, trend,
                                            prior, &controllers);

    LOG_DEBUG(<< "*** learn ***");

    //std::ofstream file;
    //file.open("results.m");
    //TDoubleVec actual;
    //TDoubleVec ly;
    //TDoubleVec my;
    //TDoubleVec uy;

    core_t::TTime time{0};
    TDouble2Vec4VecVec weights{{{1.0}}};
    for (std::size_t d = 0u; d < 20; ++d) {
        TDoubleVec noise;
        rng.generateNormalSamples(2.0, 3.0, 48, noise);
        for (auto value = noise.begin(); value != noise.end(); ++value, time += bucketLength) {
            maths::CModelAddSamplesParams params;
            params.integer(false)
                .nonNegative(true)
                .propagationInterval(1.0)
                .weightStyles(maths::CConstantWeights::COUNT)
                .trendWeights(weights)
                .priorWeights(weights);
            double y{std::max(*value, 0.0)};
            model.addSamples(params, {core::make_triple(time, TDouble2Vec{y}, TAG)});
            //actual.push_back(y);
        }
    }

    LOG_DEBUG(<< "*** forecast ***");

    TErrorBarVec prediction;
    core_t::TTime start{time};
    core_t::TTime end{time + 20 * core::constants::DAY};
    std::string m;
    TModelPtr forecastModel(model.cloneForForecast());
    forecastModel->forecast(start, end, 95.0, MINIMUM_VALUE, MAXIMUM_VALUE,
                            boost::bind(&mockSink, _1, boost::ref(prediction)), m);

    std::size_t outOfBounds{0};
    std::size_t count{0};

    for (std::size_t i = 0u; i < prediction.size(); ++i) {
        TDoubleVec noise;
        rng.generateNormalSamples(2.0, 3.0, 48, noise);
        for (auto value = noise.begin(); i < prediction.size() && value != noise.end();
             ++i, ++value, time += bucketLength) {
            CPPUNIT_ASSERT(prediction[i].s_LowerBound >= 0);
            CPPUNIT_ASSERT(prediction[i].s_Predicted >= 0);
            CPPUNIT_ASSERT(prediction[i].s_UpperBound >= 0);

            double y{std::max(*value, 0.0)};
            outOfBounds +=
                (y < prediction[i].s_LowerBound || y > prediction[i].s_UpperBound ? 1 : 0);
            ++count;
            //actual.push_back(y);
            //ly.push_back(prediction[i].s_LowerBound);
            //my.push_back(prediction[i].s_Predicted);
            //uy.push_back(prediction[i].s_UpperBound);
        }
    }

    double percentageOutOfBounds{100.0 * static_cast<double>(outOfBounds) /
                                 static_cast<double>(count)};
    LOG_DEBUG(<< "% out of bounds = " << percentageOutOfBounds);

    //file << "actual = " << core::CContainerPrinter::print(actual) << ";\n";
    //file << "ly = " << core::CContainerPrinter::print(ly) << ";\n";
    //file << "my = " << core::CContainerPrinter::print(my) << ";\n";
    //file << "uy = " << core::CContainerPrinter::print(uy) << ";\n";

    CPPUNIT_ASSERT(percentageOutOfBounds < 8.0);
}

void CForecastTest::testFinancialIndex() {
    LOG_DEBUG(<< "+-------------------------------------+");
    LOG_DEBUG(<< "|  CForecastTest::testFinancialIndex  |");
    LOG_DEBUG(<< "+-------------------------------------+");

    core_t::TTime bucketLength{1800};

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse("testfiles/financial_index.csv",
                                                    timeseries, startTime, endTime,
                                                    "^([0-9]+),([0-9\\.]+)"));
    CPPUNIT_ASSERT(!timeseries.empty());

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

    LOG_DEBUG(<< "*** learn ***");

    //std::ofstream file;
    //file.open("results.m");
    //TDoubleVec actual;
    //TDoubleVec ly;
    //TDoubleVec my;
    //TDoubleVec uy;

    std::size_t n{5 * timeseries.size() / 6};

    TDouble2Vec4VecVec weights{{{1.0}}};
    for (std::size_t i = 0u; i < n; ++i) {
        maths::CModelAddSamplesParams params;
        params.integer(false)
            .propagationInterval(1.0)
            .weightStyles(maths::CConstantWeights::COUNT)
            .trendWeights(weights)
            .priorWeights(weights);
        model.addSamples(
            params, {core::make_triple(timeseries[i].first,
                                       TDouble2Vec{timeseries[i].second}, TAG)});
        //actual.push_back(timeseries[i].second);
    }

    LOG_DEBUG(<< "*** forecast ***");

    TErrorBarVec prediction;
    core_t::TTime start{timeseries[n].first};
    core_t::TTime end{timeseries[timeseries.size() - 1].first};
    std::string m;
    TModelPtr forecastModel(model.cloneForForecast());
    forecastModel->forecast(start, end, 99.0, MINIMUM_VALUE, MAXIMUM_VALUE,
                            boost::bind(&mockSink, _1, boost::ref(prediction)), m);

    std::size_t outOfBounds{0};
    std::size_t count{0};
    TMeanAccumulator error;

    for (std::size_t i = n, j = 0u;
         i < timeseries.size() && j < prediction.size(); ++i, ++j) {
        double yi{timeseries[i].second};
        outOfBounds +=
            (yi < prediction[j].s_LowerBound || yi > prediction[j].s_UpperBound ? 1 : 0);
        ++count;
        error.add(std::fabs(yi - prediction[j].s_Predicted) / std::fabs(yi));
        //actual.push_back(yi);
        //ly.push_back(prediction[j].s_LowerBound);
        //my.push_back(prediction[j].s_Predicted);
        //uy.push_back(prediction[j].s_UpperBound);
    }

    double percentageOutOfBounds{100.0 * static_cast<double>(outOfBounds) /
                                 static_cast<double>(count)};
    LOG_DEBUG(<< "% out of bounds = " << percentageOutOfBounds);
    LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));

    //file << "actual = " << core::CContainerPrinter::print(actual) << ";\n";
    //file << "ly = " << core::CContainerPrinter::print(ly) << ";\n";
    //file << "my = " << core::CContainerPrinter::print(my) << ";\n";
    //file << "uy = " << core::CContainerPrinter::print(uy) << ";\n";

    CPPUNIT_ASSERT(percentageOutOfBounds < 53.0);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.1);
}

CppUnit::Test* CForecastTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CForecastTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastTest>(
        "CForecastTest::testDailyNoLongTermTrend", &CForecastTest::testDailyNoLongTermTrend));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastTest>(
        "CForecastTest::testDailyConstantLongTermTrend",
        &CForecastTest::testDailyConstantLongTermTrend));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastTest>(
        "CForecastTest::testDailyVaryingLongTermTrend",
        &CForecastTest::testDailyVaryingLongTermTrend));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastTest>(
        "CForecastTest::testComplexNoLongTermTrend", &CForecastTest::testComplexNoLongTermTrend));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastTest>(
        "CForecastTest::testComplexConstantLongTermTrend",
        &CForecastTest::testComplexConstantLongTermTrend));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastTest>(
        "CForecastTest::testComplexVaryingLongTermTrend",
        &CForecastTest::testComplexVaryingLongTermTrend));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastTest>(
        "CForecastTest::testNonNegative", &CForecastTest::testNonNegative));
    suiteOfTests->addTest(new CppUnit::TestCaller<CForecastTest>(
        "CForecastTest::testFinancialIndex", &CForecastTest::testFinancialIndex));

    return suiteOfTests;
}

void CForecastTest::test(TTrend trend,
                         core_t::TTime bucketLength,
                         std::size_t daysToLearn,
                         double noiseVariance,
                         double maximumPercentageOutOfBounds,
                         double maximumError) {

    //std::ofstream file;
    //file.open("results.m");
    //TDoubleVec actual;
    //TDoubleVec ly;
    //TDoubleVec my;
    //TDoubleVec uy;

    LOG_DEBUG(<< "*** learn ***");

    test::CRandomNumbers rng;

    maths::CUnivariateTimeSeriesModel::TDecayRateController2Ary controllers{
        decayRateControllers()};
    maths::CUnivariateTimeSeriesModel model(
        params(bucketLength), TAG, maths::CTimeSeriesDecomposition(0.012, bucketLength),
        maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, DECAY_RATE),
        &controllers);

    core_t::TTime time{0};
    TDouble2Vec4VecVec weights{{{1.0}}};
    for (std::size_t d = 0u; d < daysToLearn; ++d) {
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, noiseVariance, 86400 / bucketLength, noise);

        for (std::size_t i = 0u; i < noise.size(); ++i, time += bucketLength) {
            maths::CModelAddSamplesParams params;
            params.integer(false)
                .propagationInterval(1.0)
                .weightStyles(maths::CConstantWeights::COUNT)
                .trendWeights(weights)
                .priorWeights(weights);
            double yi{trend(time, noise[i])};
            model.addSamples(params, {core::make_triple(time, TDouble2Vec{yi}, TAG)});
            //actual.push_back(yi);
        }
    }

    LOG_DEBUG(<< "*** forecast ***");

    TErrorBarVec prediction;
    core_t::TTime start{time};
    core_t::TTime end{time + 2 * core::constants::WEEK};
    TModelPtr forecastModel(model.cloneForForecast());
    std::string m;
    forecastModel->forecast(start, end, 80.0, MINIMUM_VALUE, MAXIMUM_VALUE,
                            boost::bind(&mockSink, _1, boost::ref(prediction)), m);

    std::size_t outOfBounds{0};
    std::size_t count{0};
    TMeanAccumulator error;

    for (std::size_t i = 0u; i < prediction.size(); /**/) {
        TDoubleVec noise;
        rng.generateNormalSamples(0.0, noiseVariance, 86400 / bucketLength, noise);
        TDoubleVec day;
        for (std::size_t j = 0u; i < prediction.size() && j < noise.size();
             ++i, ++j, time += bucketLength) {
            double yj{trend(time, noise[j])};
            day.push_back(yj);
            outOfBounds +=
                (yj < prediction[i].s_LowerBound || yj > prediction[i].s_UpperBound ? 1 : 0);
            ++count;
            error.add(std::fabs(yj - prediction[i].s_Predicted) / std::fabs(yj));
            //actual.push_back(yj);
            //ly.push_back(prediction[i].s_LowerBound);
            //my.push_back(prediction[i].s_Predicted);
            //uy.push_back(prediction[i].s_UpperBound);
        }
    }

    double percentageOutOfBounds{100.0 * static_cast<double>(outOfBounds) /
                                 static_cast<double>(count)};
    LOG_DEBUG(<< "% out of bounds = " << percentageOutOfBounds);
    LOG_DEBUG(<< "error = " << maths::CBasicStatistics::mean(error));

    //file << "actual = " << core::CContainerPrinter::print(actual) << ";\n";
    //file << "ly = " << core::CContainerPrinter::print(ly) << ";\n";
    //file << "my = " << core::CContainerPrinter::print(my) << ";\n";
    //file << "uy = " << core::CContainerPrinter::print(uy) << ";\n";

    CPPUNIT_ASSERT(percentageOutOfBounds < maximumPercentageOutOfBounds);
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < maximumError);
}
