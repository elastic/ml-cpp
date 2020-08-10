/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CSignal.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>

#include <maths/CBasicStatistics.h>
#include <maths/CIntegerTools.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTimeSeriesSegmentation.h>

#include <boost/circular_buffer.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

namespace ml {
namespace maths {

namespace {

using TSizeVec = std::vector<std::size_t>;
using TComplex = std::complex<double>;
using TComplexVec = std::vector<TComplex>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

constexpr double LOG_TWO{0.6931471805599453};
constexpr double LOG_SIXTEEN{2.772588722239781};

//! Scale \p f by \p scale.
void scale(double scale, TComplexVec& f) {
    for (std::size_t i = 0; i < f.size(); ++i) {
        f[i] *= scale;
    }
}

//! Compute the radix 2 FFT of \p f in-place.
void radix2fft(TComplexVec& f) {
    // Perform the appropriate permutation of f(x) by swapping
    // each i in [0, N] with its bit reversal.

    std::uint64_t bits = CIntegerTools::nextPow2(f.size()) - 1;
    for (std::uint64_t i = 0; i < f.size(); ++i) {
        std::uint64_t j{CIntegerTools::reverseBits(i) >> (64 - bits)};
        if (j > i) {
            LOG_TRACE(<< j << " -> " << i);
            std::swap(f[i], f[j]);
        }
    }

    // Apply the twiddle factors.

    for (std::size_t stride = 1; stride < f.size(); stride <<= 1) {
        for (std::size_t k = 0; k < stride; ++k) {
            double t{boost::math::double_constants::pi *
                     static_cast<double>(k) / static_cast<double>(stride)};
            TComplex w(std::cos(t), std::sin(t));
            for (std::size_t start = k; start + stride < f.size(); start += 2 * stride) {
                TComplex fs{f[start]};
                TComplex tw{w * f[start + stride]};
                f[start] = fs + tw;
                f[start + stride] = fs - tw;
            }
        }
    }

    std::reverse(f.begin() + 1, f.end());
}
}

void CSignal::conj(TComplexVec& f) {
    for (std::size_t i = 0; i < f.size(); ++i) {
        f[i] = std::conj(f[i]);
    }
}

void CSignal::hadamard(const TComplexVec& fx, TComplexVec& fy) {
    for (std::size_t i = 0; i < fx.size(); ++i) {
        fy[i] *= fx[i];
    }
}

void CSignal::fft(TComplexVec& f) {
    std::size_t n{f.size()};
    std::size_t p{CIntegerTools::nextPow2(n)};
    std::size_t m{std::size_t{1} << p};

    if ((m >> 1) == n) {
        radix2fft(f);
    } else {
        // We use Bluestein's trick to reformulate as a convolution
        // which can be computed by padding to a power of 2.

        LOG_TRACE(<< "Using Bluestein's trick");

        m = 2 * n - 1;
        p = CIntegerTools::nextPow2(m);
        m = std::size_t{1} << p;

        TComplexVec chirp;
        chirp.reserve(n);
        TComplexVec a(m, TComplex{0.0, 0.0});
        TComplexVec b(m, TComplex{0.0, 0.0});

        chirp.emplace_back(1.0, 0.0);
        a[0] = f[0] * chirp[0];
        b[0] = chirp[0];
        for (std::size_t i = 1; i < n; ++i) {
            double t = boost::math::double_constants::pi *
                       static_cast<double>(i * i) / static_cast<double>(n);
            chirp.emplace_back(std::cos(t), std::sin(t));
            a[i] = f[i] * std::conj(chirp[i]);
            b[i] = b[m - i] = chirp[i];
        }

        fft(a);
        fft(b);
        hadamard(a, b);
        ifft(b);

        for (std::size_t i = 0; i < n; ++i) {
            f[i] = std::conj(chirp[i]) * b[i];
        }
    }
}

void CSignal::ifft(TComplexVec& f) {
    conj(f);
    fft(f);
    conj(f);
    scale(1.0 / static_cast<double>(f.size()), f);
}

double CSignal::cyclicAutocorrelation(std::size_t offset,
                                      const TFloatMeanAccumulatorVec& values,
                                      const TMomentTransformFunc& tranform,
                                      const TMomentWeightFunc& weight) {
    return cyclicAutocorrelation(
        offset, TFloatMeanAccumulatorCRng(values, 0, values.size()), tranform, weight);
}

double CSignal::cyclicAutocorrelation(std::size_t offset,
                                      const TFloatMeanAccumulatorCRng& values,
                                      const TMomentTransformFunc& transform,
                                      const TMomentWeightFunc& weight) {
    std::size_t n{values.size()};

    TMeanVarAccumulator moments;
    for (const auto& value : values) {
        if (CBasicStatistics::count(value) > 0.0) {
            moments.add(transform(value), weight(value));
        }
    }

    double mean{CBasicStatistics::mean(moments)};

    TMeanAccumulator autocorrelation;
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::size_t j{(i + offset) % n};
        if (CBasicStatistics::count(values[i]) > 0.0 &&
            CBasicStatistics::count(values[j]) > 0.0) {
            double avgWeight{std::sqrt(weight(values[i]) * weight(values[j]))};
            autocorrelation.add(
                (transform(values[i]) - mean) * (transform(values[j]) - mean), avgWeight);
        }
    }

    double a{CBasicStatistics::mean(autocorrelation)};
    double v{CBasicStatistics::maximumLikelihoodVariance(moments)};

    return a == v ? 1.0 : a / v;
}

void CSignal::autocorrelations(const TFloatMeanAccumulatorVec& values,
                               TComplexVec& f,
                               TDoubleVec& result) {

    result.clear();

    if (values.empty()) {
        return;
    }

    std::size_t n{values.size()};

    TMeanVarAccumulator moments;
    for (const auto& value : values) {
        if (CBasicStatistics::count(value) > 0.0) {
            moments.add(CBasicStatistics::mean(value));
        }
    }
    double mean{CBasicStatistics::mean(moments)};
    double variance{CBasicStatistics::maximumLikelihoodVariance(moments)};

    if (variance == 0.0) {
        // The autocorrelation of a constant is zero.
        result.resize(n, 0.0);
        return;
    }

    f.assign(n, TComplex{0.0, 0.0});
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j{i};
        while (j < n && CBasicStatistics::count(values[j]) == 0.0) {
            ++j;
        }
        if (i < j) {
            // Infer missing values by linearly interpolating.
            if (j == n) {
                break;
            } else if (i > 0) {
                for (std::size_t k = i; k < j; ++k) {
                    double alpha{static_cast<double>(k - i + 1) /
                                 static_cast<double>(j - i + 1)};
                    double real{CBasicStatistics::mean(values[j]) - mean};
                    f[k] = (1.0 - alpha) * f[i - 1] + alpha * TComplex{real, 0.0};
                }
            }
            i = j;
        }
        f[i] = TComplex{CBasicStatistics::mean(values[i]) - mean, 0.0};
    }

    fft(f);
    TComplexVec fConj(f);
    conj(fConj);
    hadamard(fConj, f);
    ifft(f);

    result.reserve(n);
    for (std::size_t i = 1; i < n; ++i) {
        result.push_back(f[i].real() / variance / static_cast<double>(n));
    }
}

void CSignal::removeLinearTrend(TFloatMeanAccumulatorVec& values) {
    using TRegression = CLeastSquaresOnlineRegression<1, double>;
    TRegression trend;
    double dt{10.0 / static_cast<double>(values.size())};
    double time{0.0};
    for (const auto& value : values) {
        trend.add(time, CBasicStatistics::mean(value), CBasicStatistics::count(value));
        time += dt;
    }
    time = 0.0;
    for (auto& value : values) {
        if (CBasicStatistics::count(value) > 0.0) {
            CBasicStatistics::moment<0>(value) -= trend.predict(time);
        }
        time += dt;
    }
}

CSignal::SSeasonalComponentSummary CSignal::seasonalComponentSummary(std::size_t period) {
    return {period, 0, period, TSizeSizePr{0, period}};
}

void CSignal::appendSeasonalComponentSummary(std::size_t period,
                                             TSeasonalComponentVec& periods) {
    periods.emplace_back(period, 0, period, TSizeSizePr{0, period});
}

CSignal::TSeasonalComponentVec
CSignal::seasonalDecomposition(TFloatMeanAccumulatorVec& values,
                               double outlierFraction,
                               const TSizeSizeSizeTr& diurnal,
                               const TPeriodWeightFunc& weight,
                               TOptionalSize startOfWeekOverride) {

    std::size_t n{values.size()};
    if (CSignal::countNotMissing(values) < 10) {
        return {};
    }

    auto count = [](const TFloatMeanAccumulatorVec& valuesToTest) {
        return std::accumulate(valuesToTest.begin(), valuesToTest.end(), 0.0,
                               [](double result, const TFloatMeanAccumulator& value) {
                                   return result + CBasicStatistics::count(value);
                               });
    };
    auto residualMoments = [&](TFloatMeanAccumulatorVec& valuesToTest,
                               const TSeasonalComponentVec& periods,
                               const TMeanAccumulatorVec1Vec& components) {
        TMeanVarAccumulator result;
        for (std::size_t i = 0; i < valuesToTest.size(); ++i) {
            if (CBasicStatistics::count(valuesToTest[i]) > 0.0) {
                for (std::size_t j = 0; j < components.size(); ++j) {
                    CBasicStatistics::moment<0>(valuesToTest[i]) -=
                        periods[j].value(components[j], i);
                }
                result.add(CBasicStatistics::mean(valuesToTest[i]));
            }
        }
        return result;
    };

    TSeasonalComponentVec result;

    std::size_t day, week, year;
    std::tie(day, week, year) = diurnal;

    std::size_t missingAtEnd(std::find_if(values.rbegin(), values.rend(),
                                          [](const auto& value) {
                                              return CBasicStatistics::count(value) > 0.0;
                                          }) -
                             values.rbegin());
    n -= missingAtEnd;

    std::size_t pad{n / 4};
    TSizeVec periods;
    periods.reserve(pad + 2);
    periods.resize(pad - 1);
    std::iota(periods.begin(), periods.end(), 2);
    for (auto period : {day, week, year}) {
        if (periods.back() < period && n >= 2 * period) {
            periods.push_back(period);
        }
    }

    TDoubleVec correlations;
    TComplexVec placeholder;
    TFloatMeanAccumulatorVec valuesToTest{values.begin(), values.begin() + n};
    TSeasonalComponentVec decomposition;
    TMeanAccumulatorVec1Vec components;
    TSizeVec selected;
    std::size_t withoutComponent{0};
    double varianceWithComponent{0.0};
    double varianceWithoutComponent{0.0};
    double params{0.0};
    double scale{1.0};
    double points{static_cast<double>(countNotMissing(valuesToTest))};
    double pValueThreshold{0.05};
    double pValue{0.005};
    double eps{static_cast<double>(std::numeric_limits<float>::epsilon())};

    result.push_back(seasonalComponentSummary(1));
    fitSeasonalComponentsRobust(result, outlierFraction, valuesToTest, components);
    TMeanVarAccumulator momentsWithoutComponent{
        residualMoments(valuesToTest, result, components)};
    result.clear();

    do {
        // Compute the serial autocorrelations padding to the maximum offset
        // to avoid windowing effects.
        valuesToTest.resize(n + 3 * pad);
        autocorrelations(valuesToTest, placeholder, correlations);
        valuesToTest.resize(n);
        correlations.resize(3 * pad);

        // Average the serial correlations of each component over offets P, 2P,
        // ..., mP for mP < n. Note that we need to correct the correlation for
        // longer offsets for the zero pad we append.
        for (std::size_t i = 0; i < periods.size(); ++i) {
            std::size_t period{periods[i]};
            TMeanAccumulator meanCorrelation;
            for (std::size_t offset = period; offset < 3 * pad; offset += period) {
                meanCorrelation.add(correlations[offset - 1] * static_cast<double>(n) /
                                    static_cast<double>(n - offset));
            }
            correlations[period - 1] = weight(period) * CBasicStatistics::mean(meanCorrelation);
            LOG_TRACE(<< "correlation(" << period << ") = " << correlations[period - 1]);
        }

        auto correlationLess = [&](std::size_t lhs, std::size_t rhs) {
            return correlations[lhs - 1] < correlations[rhs - 1];
        };

        std::size_t maxCorrelationPeriod{
            *std::max_element(periods.begin(), periods.end(), correlationLess)};
        std::size_t maxCorrelationIndex{maxCorrelationPeriod - 1};
        LOG_TRACE(<< "max correlation(" << maxCorrelationPeriod
                  << ") = " << correlations[maxCorrelationIndex]);

        std::size_t selectedPeriod{maxCorrelationPeriod};

        // We prefer shorter periods if the decision is close because the model
        // will be more accurate. In particular, if we have a divisor of the best
        // period whose autocorrelation is within epsilon we'll select that one.
        double cutoff{0.8 * correlations[maxCorrelationIndex]};
        LOG_TRACE(<< "cutoff = " << cutoff);
        for (auto period : periods) {
            if (period <= selectedPeriod / 2) {
                if (maxCorrelationPeriod % period == 0 && correlations[period - 1] > cutoff) {
                    selectedPeriod = period;
                }
            } else {
                break;
            }
        }
        LOG_TRACE(<< "selected period = " << selectedPeriod);

        withoutComponent = result.size();

        if (std::find(selected.begin(), selected.end(), selectedPeriod) !=
                selected.end() ||
            varianceWithComponent < eps * varianceWithoutComponent) {
            break;
        }
        selected.push_back(selectedPeriod);

        valuesToTest.assign(values.begin(), values.begin() + n);

        decomposition.clear();
        if (selectedPeriod == day || selectedPeriod == week) {
            decomposition = tradingDayDecomposition(valuesToTest, outlierFraction,
                                                    week, startOfWeekOverride);
            valuesToTest.assign(values.begin(), values.begin() + n);
        }

        if (decomposition.empty()) {
            appendSeasonalComponentSummary(selectedPeriod, result);
        } else {
            result.insert(result.end(), decomposition.begin(), decomposition.end());
        }

        fitSeasonalComponents(result, valuesToTest, components);
        scale = 1.0 / count(valuesToTest);
        reweightOutliers(result, components, outlierFraction, valuesToTest);
        scale *= count(valuesToTest);

        params = 0.0;
        for (std::size_t j = withoutComponent; j < components.size(); ++j) {
            params += static_cast<double>(countNotMissing(components[j]));
        }

        TMeanVarAccumulator momentsWithComponent{
            residualMoments(valuesToTest, result, components)};
        varianceWithComponent = CBasicStatistics::variance(momentsWithComponent);
        varianceWithoutComponent = CBasicStatistics::variance(momentsWithoutComponent);
        momentsWithoutComponent = momentsWithComponent;

        pValueThreshold = std::pow(
            0.05, std::max((0.2 - correlations[selectedPeriod - 1]) / 0.02, 1.0));
        double F{varianceWithoutComponent == varianceWithComponent
                     ? 1.0
                     : varianceWithoutComponent / varianceWithComponent};
        pValue = CTools::oneMinusPowOneMinusX(
            CStatisticalTests::rightTailFTest(F, scale * points - 1.0, scale * points - params),
            static_cast<double>(pad - 4));
        LOG_TRACE(<< "variance without component = " << varianceWithoutComponent
                  << ", variance with component = " << varianceWithComponent
                  << ", correlation = " << correlations[selectedPeriod - 1]
                  << ", number points = " << scale * points << ", scale = " << scale
                  << ", number parameters = " << params << " p-value = " << pValue
                  << ", threshold to accept = " << pValueThreshold);

    } while (pValue < pValueThreshold);

    result.resize(withoutComponent);
    fitSeasonalComponents(result, values, components);
    reweightOutliers(result, components, outlierFraction, values);

    return result;
}

CSignal::TSeasonalComponentVec
CSignal::tradingDayDecomposition(TFloatMeanAccumulatorVec& values,
                                 double outlierFraction,
                                 std::size_t week,
                                 TOptionalSize startOfWeekOverride) {

    using TSizeAry = std::array<std::size_t, 4>;
    using TSizeSizePr2VecAry = std::array<TSizeSizePr2Vec, 4>;
    using TMeanAccumulatorAry = std::array<TMeanAccumulator, 4>;
    using TMeanVarAccumulatorBuffer = boost::circular_buffer<TMeanVarAccumulator>;
    using TMeanVarAccumulatorBufferAry = std::array<TMeanVarAccumulatorBuffer, 4>;
    using TMeanVarAccumulatorBuffer1Vec = core::CSmallVector<TMeanVarAccumulatorBuffer, 1>;

    if (values.size() < 2 * week) {
        return {};
    }

    std::size_t day{(week + 3) / 7};
    std::size_t weekend{(2 * week + 3) / 7};
    std::size_t weekday{(5 * week + 3) / 7};

    // Work on the largest subset of the values which is a multiple week.
    std::size_t remainder{values.size() % week};
    TFloatMeanAccumulatorCRng valuesToTest{values, 0, values.size() - remainder};
    std::size_t n{valuesToTest.size()};
    LOG_TRACE(<< "number values = " << n);

    std::size_t startOfWeek{startOfWeekOverride != boost::none ? *startOfWeekOverride : 0};

    double weeklyVariance{[&] {
        TSeasonalComponentVec weeklyPeriod{seasonalComponentSummary(week)};
        TMeanAccumulatorVec1Vec weeklyComponent;
        fitSeasonalComponentsRobust(weeklyPeriod, outlierFraction, values, weeklyComponent);
        return residualVariance(values, weeklyPeriod, weeklyComponent).second;
    }()};
    double dailyVariance{[&] {
        TSeasonalComponentVec dailyPeriod{seasonalComponentSummary(day)};
        TMeanAccumulatorVec1Vec dailyComponent;
        doFitSeasonalComponents(dailyPeriod, valuesToTest, dailyComponent);
        return residualVariance(values, dailyPeriod, dailyComponent).second;
    }()};

    if (dailyVariance == 0.0) {
        return {};
    }

    TSizeSizePr2Vec weekends;
    TSizeSizePr2Vec weekdays;
    for (std::size_t i = 0; i < n; i += week) {
        weekends.emplace_back(i, i + weekend);
        weekdays.emplace_back(i + weekend, i + week);
    }
    LOG_TRACE(<< "day = " << day << ", weekend = " << weekend << ", weekday = " << weekday);
    LOG_TRACE(<< "weekends = " << core::CContainerPrinter::print(weekends)
              << ", weekdays = " << core::CContainerPrinter::print(weekdays));

    constexpr std::size_t WEEKEND_DAILY{0};
    constexpr std::size_t WEEKEND_WEEKLY{1};
    constexpr std::size_t WEEKDAY_DAILY{2};
    constexpr std::size_t WEEKDAY_WEEKLY{3};
    TSizeAry strides{day, weekend, day, weekday};
    TSizeSizePr2VecAry partitions{weekends, weekends, weekdays, weekdays};
    TMeanVarAccumulatorBufferAry components{
        TMeanVarAccumulatorBuffer{day, TMeanVarAccumulator{}},
        TMeanVarAccumulatorBuffer{weekend, TMeanVarAccumulator{}},
        TMeanVarAccumulatorBuffer{day, TMeanVarAccumulator{}},
        TMeanVarAccumulatorBuffer{weekday, TMeanVarAccumulator{}}};
    LOG_TRACE(<< "strides = " << core::CContainerPrinter::print(strides));
    LOG_TRACE(<< "partitions = " << core::CContainerPrinter::print(partitions));

    // Initialize the components.
    TMeanVarAccumulatorBuffer1Vec component(1);
    auto initialize = [&](const TSizeSizePr& window, TMeanVarAccumulatorBuffer& component_) {
        component[0].swap(component_);
        std::size_t period{component[0].size()};
        doFitSeasonalComponents({{period, startOfWeek, week, window}}, valuesToTest, component);
        component[0].swap(component_);
    };
    initialize({0, weekend}, components[WEEKEND_DAILY]);
    initialize({0, weekend}, components[WEEKEND_WEEKLY]);
    initialize({weekend, week}, components[WEEKDAY_DAILY]);
    initialize({weekend, week}, components[WEEKDAY_WEEKLY]);
    LOG_TRACE(<< "components = " << core::CContainerPrinter::print(components));

    TMeanAccumulatorAry variances;
    for (std::size_t i = 0; i < components.size(); ++i) {
        variances[i] = std::accumulate(
            components[i].begin(), components[i].end(), TMeanAccumulator{},
            [](TMeanAccumulator variance_, const TMeanVarAccumulator& value) {
                variance_.add(CBasicStatistics::variance(value),
                              CBasicStatistics::count(value));
                return variance_;
            });
    }

    // We consider three possibilities:
    //   1) Daily periodity for weekdays and weekends,
    //   2) Daily periodity for weekdays only,
    //   3) Daily periodity for weekends only.

    constexpr std::size_t WEEKEND_DAILY_WEEKDAY_DAILY{0};
    constexpr std::size_t WEEKEND_WEEKLY_WEEKDAY_DAILY{1};
    constexpr std::size_t WEEKEND_DAILY_WEEKDAY_WEEKLY{2};
    constexpr std::size_t WEEKLY{3};
    constexpr std::size_t NO_TRADING_DAY{4};
    TDoubleVec candidates(3 * week, 0.0);

    auto captureVarianceAtStartOfWeek = [&](std::size_t i) {
        candidates[3 * i + WEEKEND_DAILY_WEEKDAY_DAILY] = CBasicStatistics::mean(
            variances[WEEKEND_DAILY] + variances[WEEKDAY_DAILY]);
        candidates[3 * i + WEEKEND_WEEKLY_WEEKDAY_DAILY] = CBasicStatistics::mean(
            variances[WEEKEND_WEEKLY] + variances[WEEKDAY_DAILY]);
        candidates[3 * i + WEEKEND_DAILY_WEEKDAY_WEEKLY] = CBasicStatistics::mean(
            variances[WEEKEND_DAILY] + variances[WEEKDAY_WEEKLY]);
    };

    if (startOfWeekOverride != boost::none) {
        captureVarianceAtStartOfWeek(startOfWeek);
        startOfWeek *= 3;
    } else {
        // Compute the variances for each candidate partition.
        captureVarianceAtStartOfWeek(0);
        for (std::size_t i = 0; i + 1 < week; ++i) {
            for (std::size_t j = 0; j < components.size(); ++j) {
                TMeanVarAccumulator next;
                for (const auto& subset : partitions[j]) {
                    for (std::size_t k = i + subset.first + strides[j];
                         k <= i + subset.second; k += strides[j]) {
                        next.add(CBasicStatistics::mean(valuesToTest[k % n]),
                                 CBasicStatistics::count(valuesToTest[k % n]));
                    }
                }
                auto last = components[j].front();
                components[j].push_back(next);
                variances[j] += CBasicStatistics::momentsAccumulator(
                    CBasicStatistics::count(next), CBasicStatistics::variance(next));
                variances[j] -= CBasicStatistics::momentsAccumulator(
                    CBasicStatistics::count(last), CBasicStatistics::variance(last));
            }
            captureVarianceAtStartOfWeek(i + 1);
        }

        double minCost{std::numeric_limits<double>::max()};
        startOfWeek = 3 * week + 3;

        // For each possibility extract the best explanation. We seek to partition
        // where the time series value is absolutely small and the total difference
        // between values either side of the partition times is small.
        double threshold{candidates[0]};
        for (std::size_t i = 3; i < candidates.size(); i += 3) {
            threshold = std::min(threshold, candidates[i]);
        }
        threshold *= 1.05;
        for (std::size_t i = 0; i < candidates.size(); i += 3) {
            for (std::size_t j = 0; j < 3; ++j) {
                if (candidates[i] < threshold) {
                    double cost{0.0};
                    for (std::size_t k = i / 3; k < i / 3 + n; k += week) {
                        double knots[]{
                            CBasicStatistics::mean(valuesToTest[(k + n - 1) % n]),
                            CBasicStatistics::mean(valuesToTest[(k + n + 0) % n]),
                            CBasicStatistics::mean(valuesToTest[(k + n + weekend - 1) % n]),
                            CBasicStatistics::mean(valuesToTest[(k + n + weekend + 0) % n])};
                        cost += std::fabs(knots[0]) + std::fabs(knots[1]) +
                                std::fabs(knots[2]) + std::fabs(knots[3]) +
                                std::fabs(knots[1] - knots[0]) +
                                std::fabs(knots[3] - knots[2]);
                    }
                    LOG_TRACE(<< "cost(" << i / 3 << "," << j << ") = " << cost);
                    std::tie(minCost, startOfWeek) = std::min(
                        std::make_pair(minCost, startOfWeek), std::make_pair(cost, i));
                }
            }
        }
    }
    LOG_TRACE(<< "start of week = " << startOfWeek / 3);

    double degreesFreedom[]{static_cast<double>(n - 2 * day),
                            static_cast<double>(n - (weekend + day)),
                            static_cast<double>(n - (weekday + day)),
                            static_cast<double>(n - week), static_cast<double>(n - day)};
    double minimumVariances[]{candidates[startOfWeek + WEEKEND_DAILY_WEEKDAY_DAILY],
                              candidates[startOfWeek + WEEKEND_WEEKLY_WEEKDAY_DAILY],
                              candidates[startOfWeek + WEEKEND_DAILY_WEEKDAY_WEEKLY],
                              weeklyVariance, dailyVariance};
    LOG_TRACE(<< "degrees freedom = " << core::CContainerPrinter::print(degreesFreedom));
    LOG_TRACE(<< "variances = " << core::CContainerPrinter::print(minimumVariances));

    std::size_t h0{NO_TRADING_DAY};
    auto test = [&](std::size_t h1, double significantPValue = 0.05) {
        double pValue{CStatisticalTests::rightTailFTest(
            minimumVariances[h0] == minimumVariances[h1]
                ? 1.0
                : minimumVariances[h0] / minimumVariances[h1],
            degreesFreedom[h0], degreesFreedom[h1])};
        if (pValue < significantPValue) {
            h0 = h1;
            return std::make_pair(true, pValue);
        }
        return std::make_pair(false, pValue);
    };

    TSeasonalComponentVec result;
    double referencePValue{1.0};
    for (auto alternative : {WEEKEND_DAILY_WEEKDAY_DAILY, WEEKEND_WEEKLY_WEEKDAY_DAILY,
                             WEEKEND_DAILY_WEEKDAY_WEEKLY}) {
        bool accept;
        double pValue;
        std::tie(accept, pValue) = test(alternative);
        if (accept) {
            switch (alternative) {
            case WEEKEND_DAILY_WEEKDAY_DAILY:
                result.emplace_back(day, startOfWeek / 3, week,
                                    TSizeSizePr{0 * day, 2 * day});
                result.emplace_back(day, startOfWeek / 3, week,
                                    TSizeSizePr{2 * day, 7 * day});
                referencePValue = pValue;
                break;
            case WEEKEND_WEEKLY_WEEKDAY_DAILY:
                result.emplace_back(week, startOfWeek / 3, week,
                                    TSizeSizePr{0 * day, 2 * day});
                result.emplace_back(day, startOfWeek / 3, week,
                                    TSizeSizePr{2 * day, 7 * day});
                break;
            case WEEKEND_DAILY_WEEKDAY_WEEKLY:
                result.emplace_back(day, startOfWeek / 3, week,
                                    TSizeSizePr{0 * day, 2 * day});
                result.emplace_back(week, startOfWeek / 3, week,
                                    TSizeSizePr{2 * day, 7 * day});
                break;
            default:
                break;
            }
        }
    }

    // Check if the evidence for weekly is stronger than for weekday/weekend split.
    LOG_TRACE(<< "reference p-value = " << referencePValue);
    h0 = WEEKEND_DAILY_WEEKDAY_DAILY;
    if (result.size() > 0 && test(WEEKLY, 10.0 * referencePValue).first) {
        result.clear();
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

    return result;
}

void CSignal::fitSeasonalComponents(const TSeasonalComponentVec& periods,
                                    const TFloatMeanAccumulatorVec& values,
                                    TMeanAccumulatorVec1Vec& components) {
    doFitSeasonalComponents(periods, values, components);
}

void CSignal::fitSeasonalComponentsRobust(const TSeasonalComponentVec& periods,
                                          double outlierFraction,
                                          TFloatMeanAccumulatorVec& values,
                                          TMeanAccumulatorVec1Vec& components) {
    fitSeasonalComponents(periods, values, components);
    if (outlierFraction > 0.0) {
        reweightOutliers(periods, components, outlierFraction, values);
        fitSeasonalComponents(periods, values, components);
    }
}

void CSignal::reweightOutliers(const TSeasonalComponentVec& periods,
                               const TMeanAccumulatorVec1Vec& components,
                               double fraction,
                               TFloatMeanAccumulatorVec& values) {
    auto predictor = [&](std::size_t index) {
        double value{0.0};
        for (std::size_t i = 0; i < components.size(); ++i) {
            value += periods[i].value(components[i], index);
        }
        return value;
    };
    reweightOutliers(predictor, fraction, values);
}

void CSignal::reweightOutliers(const TPredictor& predictor,
                               double fraction,
                               TFloatMeanAccumulatorVec& values) {

    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr, std::greater<TDoubleSizePr>>;

    std::size_t numberOutliers{static_cast<std::size_t>([&] {
        return fraction * static_cast<double>(countNotMissing(values));
    }())};
    LOG_TRACE(<< "number outliers = " << numberOutliers);

    if (numberOutliers == 0) {
        return;
    }

    TMaxAccumulator outliers{2 * numberOutliers};
    TMeanAccumulator meanAbs;
    TMeanAccumulator meanDifference;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (CBasicStatistics::count(values[i]) > 0.0) {
            double difference{std::fabs(CBasicStatistics::mean(values[i]) - predictor(i))};
            outliers.add({difference, i});
            meanAbs.add(std::fabs(CBasicStatistics::mean(values[i])));
            meanDifference.add(difference);
        }
    }
    if (CBasicStatistics::mean(meanDifference) == 0.0) {
        return;
    }

    outliers.sort();
    LOG_TRACE(<< "outliers = " << core::CContainerPrinter::print(outliers));

    TMeanAccumulator meanDifferenceOfOutliers;
    for (std::size_t i = 0; 4 * i < outliers.count(); ++i) {
        meanDifferenceOfOutliers.add(outliers[i].first);
    }
    meanDifference -= meanDifferenceOfOutliers;
    double threshold{std::max(3.0 * CBasicStatistics::mean(meanDifference),
                              std::numeric_limits<double>::epsilon() *
                                  CBasicStatistics::mean(meanAbs))};
    LOG_TRACE(<< "threshold = " << CBasicStatistics::mean(meanDifference));

    double logThreshold{std::log(threshold)};
    for (const auto& outlier : outliers) {
        double logDifference{std::log(outlier.first)};
        CBasicStatistics::count(values[outlier.second]) *=
            CTools::linearlyInterpolate(logThreshold - LOG_TWO, logThreshold,
                                        1.0, 0.1, logDifference) *
            CTools::linearlyInterpolate(logThreshold, logThreshold + LOG_SIXTEEN,
                                        1.0, 0.1, logDifference);
    }
    LOG_TRACE(<< "values - outliers = " << core::CContainerPrinter::print(values));
}

double CSignal::meanNumberRepeatedValues(const TFloatMeanAccumulatorVec& values,
                                         const SSeasonalComponentSummary& period) {
    TDoubleVec counts(period.period(), 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (CBasicStatistics::count(values[i]) > 0.0 && period.contains(i)) {
            counts[period.offset(i)] += 1.0;
        }
    }
    return CBasicStatistics::mean(
        std::accumulate(counts.begin(), counts.end(), TMeanAccumulator{},
                        [](TMeanAccumulator mean, double count) {
                            if (count > 0.0) {
                                mean.add(count);
                            }
                            return mean;
                        }));
}

CSignal::TDoubleDoublePr
CSignal::residualVariance(const TFloatMeanAccumulatorVec& values,
                          const TSeasonalComponentVec& periods,
                          const TMeanAccumulatorVec1Vec& components) {
    TMeanVarAccumulator moments[2];
    for (std::size_t i = 0; i < values.size(); ++i) {
        bool skip{true};
        double prediction{0.0};
        for (std::size_t j = 0; j < periods.size(); ++j) {
            if (periods[j].contains(i)) {
                skip = false;
                prediction += CBasicStatistics::mean(components[j][periods[j].offset(i)]);
            }
        }
        if (skip == false) {
            moments[0].add(CBasicStatistics::mean(values[i]),
                           CBasicStatistics::count(values[i]));
            moments[1].add(CBasicStatistics::mean(values[i]) - prediction,
                           CBasicStatistics::count(values[i]));
        }
    }
    return {CBasicStatistics::variance(moments[0]),
            CBasicStatistics::variance(moments[1])};
}

std::size_t CSignal::selectComponentSize(const TFloatMeanAccumulatorVec& values,
                                         std::size_t period) {

    auto interpolate = [&](std::size_t i, const std::size_t* adjacent,
                           const TMeanAccumulatorVec& centres,
                           const TMeanAccumulatorVec& model) {
        double prediction{0.0};
        double adjacentCentres[]{CBasicStatistics::mean(centres[adjacent[0]]),
                                 CBasicStatistics::mean(centres[adjacent[1]]),
                                 CBasicStatistics::mean(centres[adjacent[2]])};
        double distances[]{std::fabs(adjacentCentres[0] - static_cast<double>(i)),
                           std::fabs(adjacentCentres[1] - static_cast<double>(i)),
                           std::fabs(adjacentCentres[2] - static_cast<double>(i))};
        double Z{0.0};
        for (std::size_t j = 0; j < 3; ++j) {
            distances[j] = std::min(distances[j],
                                    static_cast<double>(period) - distances[j]);
            double weight{std::exp(-distances[j])};
            prediction += weight * CBasicStatistics::mean(model[adjacent[j]]);
            Z += weight;
        }
        return prediction / Z;
    };

    auto residualVariance = [&](const TMeanAccumulatorVec& centres,
                                const TMeanAccumulatorVec& model) {
        TMeanVarAccumulator moments;
        std::size_t compression{period / model.size()};
        for (std::size_t i = 0; i < values.size(); ++i) {
            std::size_t bucket{(i % period) / compression};
            std::size_t indices[]{(bucket + model.size() - 1) % model.size(),
                                  (bucket + model.size() + 0) % model.size(),
                                  (bucket + model.size() + 1) % model.size()};
            double prediction{interpolate(i % period, indices, centres, model)};
            moments.add(CBasicStatistics::mean(values[i]) - prediction,
                        CBasicStatistics::count(values[i]));
        }
        return CBasicStatistics::variance(moments);
    };

    TMeanAccumulatorVec centres(period);
    for (std::size_t i = 0; i < centres.size(); ++i) {
        centres[i].add(static_cast<double>(i % period));
    }

    TMeanAccumulatorVec1Vec component;
    fitSeasonalComponents({seasonalComponentSummary(period)}, values, component);
    TMeanAccumulatorVec compressedComponent{std::move(component[0])};

    std::size_t size{period};

    std::size_t h0{1};
    double degreesFreedom[]{static_cast<double>(values.size() - period), 0.0};
    double variances[]{residualVariance(centres, compressedComponent), 0.0};

    for (std::size_t i = 2; i <= period / 2; ++i) {
        if (period % i == 0) {
            LOG_TRACE(<< "size = " << period / i);

            centres.assign(period / i, TMeanAccumulator{});
            compressedComponent.assign(period / i, TMeanAccumulator{});
            for (std::size_t j = 0; j < values.size(); ++j) {
                std::size_t bucket{(j % period) / i};
                centres[bucket].add(static_cast<double>(j % period),
                                    CBasicStatistics::count(values[j]));
                compressedComponent[bucket].add(CBasicStatistics::mean(values[j]),
                                                CBasicStatistics::count(values[j]));
            }
            LOG_TRACE(<< "centres = " << core::CContainerPrinter::print(centres));

            degreesFreedom[h0] = static_cast<double>(values.size() - period / i);
            variances[h0] = residualVariance(centres, compressedComponent);
            LOG_TRACE(<< "degrees freedom = "
                      << core::CContainerPrinter::print(degreesFreedom));
            LOG_TRACE(<< "variances = " << core::CContainerPrinter::print(variances));

            if (variances[h0] != variances[1 - h0] &&
                CStatisticalTests::rightTailFTest(variances[h0] / variances[1 - h0],
                                                  degreesFreedom[h0],
                                                  degreesFreedom[1 - h0]) < 0.1) {
                break;
            }
            if (CStatisticalTests::rightTailFTest(variances[1 - h0] / variances[h0],
                                                  degreesFreedom[1 - h0],
                                                  degreesFreedom[h0]) < 0.1) {
                h0 = 1 - h0;
            }
            size = compressedComponent.size();
        }
    }

    return size;
}

template<typename VALUES, typename COMPONENT>
void CSignal::doFitSeasonalComponents(const TSeasonalComponentVec& periods,
                                      const VALUES& values,
                                      core::CSmallVector<COMPONENT, 1>& components) {
    if (periods.empty()) {
        return;
    }

    LOG_TRACE(<< "periods = " << core::CContainerPrinter::print(periods));

    components.resize(periods.size());
    for (std::size_t i = 0; i < periods.size(); ++i) {
        components[i].assign(periods[i].period(), typename COMPONENT::value_type{});
    }

    // The iterative scheme is as follows:
    //   1. Minimize MSE for period p(i) w.r.t. {values - components(j != i)}
    //      holding other components fixed,
    //   2. Set i = i + 1 % "number components".
    //
    // Note that the iterative refinements are only necessary if the number
    // of values isn't a multiple of the least common multiple of the periods.
    // We could check this but the iterations are really cheap so just always
    // iterating is fast enough.

    std::size_t iterations{components.size() == 1 ? 1 : 2 * components.size()};

    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {

        std::size_t i{iteration % components.size()};

        auto predictor = [&](std::size_t index) {
            double value{0.0};
            for (std::size_t j = 0; j < i; ++j) {
                value += periods[j].value(components[j], index);
            }
            for (std::size_t j = i + 1; j < components.size(); ++j) {
                value += periods[j].value(components[j], index);
            }
            return value;
        };

        fitSeasonalComponentsMinusPrediction(periods[i], predictor, values, components[i]);
    }
}

template<typename PREDICTOR, typename VALUES, typename COMPONENT>
void CSignal::fitSeasonalComponentsMinusPrediction(const SSeasonalComponentSummary& period,
                                                   const PREDICTOR& predictor,
                                                   const VALUES& values,
                                                   COMPONENT& component) {
    if (period.period() > 0) {
        component.assign(period.period(), typename COMPONENT::value_type{});
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (period.contains(i)) {
                component[period.offset(i)].add(
                    CBasicStatistics::mean(values[i]) - predictor(i),
                    CBasicStatistics::count(values[i]));
            }
        }
    }
}
}
}
