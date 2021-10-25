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

#include <maths/time_series/CSignal.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CIntegerTools.h>
#include <maths/common/CLeastSquaresOnlineRegression.h>
#include <maths/common/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/common/CStatisticalTests.h>

#include <maths/time_series/CTimeSeriesSegmentation.h>

#include <boost/circular_buffer.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <tuple>

namespace ml {
namespace maths {
namespace time_series {

namespace {

using TSizeVec = std::vector<std::size_t>;
using TComplex = std::complex<double>;
using TComplexVec = std::vector<TComplex>;
using TMeanAccumulator = common::CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = common::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;

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
    // Perform the appropriate permutation of f(x) by swapping each i in [0, N]
    // with its bit reversal.

    std::uint64_t bits = common::CIntegerTools::nextPow2(f.size()) - 1;
    for (std::uint64_t i = 0; i < f.size(); ++i) {
        std::uint64_t j{common::CIntegerTools::reverseBits(i) >> (64 - bits)};
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
    std::size_t p{common::CIntegerTools::nextPow2(n)};
    std::size_t m{std::size_t{1} << p};

    if ((m >> 1) == n) {
        radix2fft(f);
    } else {
        // We use Bluestein's trick to reformulate as a convolution which can be
        // computed by padding to a power of 2.

        LOG_TRACE(<< "Using Bluestein's trick");

        m = 2 * n - 1;
        p = common::CIntegerTools::nextPow2(m);
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

double CSignal::cyclicAutocorrelation(const SSeasonalComponentSummary& period,
                                      const TFloatMeanAccumulatorVec& values,
                                      const TMomentTransformFunc& tranform,
                                      const TMomentWeightFunc& weight,
                                      double eps) {
    return cyclicAutocorrelation(period,
                                 TFloatMeanAccumulatorCRng(values, 0, values.size()),
                                 tranform, weight, eps);
}

double CSignal::cyclicAutocorrelation(const SSeasonalComponentSummary& period,
                                      const TFloatMeanAccumulatorCRng& values,
                                      const TMomentTransformFunc& transform,
                                      const TMomentWeightFunc& weight,
                                      double eps) {
    TMeanVarAccumulator moments;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (period.contains(i) && common::CBasicStatistics::count(values[i]) > 0.0) {
            moments.add(transform(values[i]), weight(values[i]));
        }
    }

    double mean{common::CBasicStatistics::mean(moments)};

    TMeanAccumulator autocorrelation;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (period.contains(i)) {
            std::size_t j{period.nextRepeat(i) % values.size()};
            if (common::CBasicStatistics::count(values[i]) > 0.0 &&
                common::CBasicStatistics::count(values[j]) > 0.0) {
                double avgWeight{std::sqrt(weight(values[i]) * weight(values[j]))};
                autocorrelation.add(
                    (transform(values[i]) - mean) * (transform(values[j]) - mean), avgWeight);
            }
        }
    }

    double a{common::CBasicStatistics::mean(autocorrelation)};
    double v{common::CBasicStatistics::maximumLikelihoodVariance(moments) + eps};

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
        if (common::CBasicStatistics::count(value) > 0.0) {
            moments.add(common::CBasicStatistics::mean(value));
        }
    }
    double mean{common::CBasicStatistics::mean(moments)};
    double variance{common::CBasicStatistics::maximumLikelihoodVariance(moments)};

    if (variance == 0.0) {
        // The autocorrelation of a constant is zero.
        result.resize(n, 0.0);
        return;
    }

    f.assign(n, TComplex{0.0, 0.0});
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j{i};
        while (j < n && common::CBasicStatistics::count(values[j]) == 0.0) {
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
                    double real{common::CBasicStatistics::mean(values[j]) - mean};
                    f[k] = (1.0 - alpha) * f[i - 1] + alpha * TComplex{real, 0.0};
                }
            }
            i = j;
        }
        f[i] = TComplex{common::CBasicStatistics::mean(values[i]) - mean, 0.0};
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

CSignal::SSeasonalComponentSummary CSignal::seasonalComponentSummary(std::size_t period) {
    return {period, 0, period, TSizeSizePr{0, period}};
}

void CSignal::appendSeasonalComponentSummary(std::size_t period,
                                             TSeasonalComponentVec& periods) {
    periods.emplace_back(period, 0, period, TSizeSizePr{0, period});
}

CSignal::TSeasonalComponentVec
CSignal::seasonalDecomposition(const TFloatMeanAccumulatorVec& values,
                               double outlierFraction,
                               const TSizeSizeSizeTr& diurnal,
                               TOptionalSize startOfWeekOverride,
                               double significantPValue,
                               std::size_t maxComponents) {

    if (CSignal::countNotMissing(values) < 10) {
        return {};
    }

    TSeasonalComponentVec result;

    std::size_t day, week, year;
    std::tie(day, week, year) = diurnal;

    std::size_t n{values.size()};
    std::size_t missingAtEnd(
        std::find_if(values.rbegin(), values.rend(),
                     [](const auto& value) {
                         return common::CBasicStatistics::count(value) > 0.0;
                     }) -
        values.rbegin());
    n -= missingAtEnd;

    std::size_t pad{n / 4};
    TSizeVec periods;
    periods.reserve(pad + 2);
    periods.resize(pad - 1);
    std::iota(periods.begin(), periods.end(), 2);
    for (auto period : {day, week, year}) {
        if (periods.back() < period && 100 * n > 190 * period) {
            periods.push_back(period);
        }
    }

    TDoubleVec correlations;
    TComplexVec placeholder;
    TFloatMeanAccumulatorVec valuesToTest{values.begin(), values.begin() + n};
    TSeasonalComponentVec decomposition;
    TMeanAccumulatorVecVec components;
    TSizeVec candidatePeriods;
    TSizeVec selectedPeriods;
    std::size_t sizeWithoutComponent{0};
    double varianceWithComponent{0.0};
    double varianceWithoutComponent{0.0};
    double pValue{1.0};
    double eps{common::CTools::pow2(
        1000.0 * static_cast<double>(std::numeric_limits<float>::epsilon()))};

    result.push_back(seasonalComponentSummary(1));
    fitSeasonalComponentsRobust(result, outlierFraction, valuesToTest, components);
    auto H0 = residualVarianceStats(valuesToTest, result, components);
    result.clear();

    do {
        // Centre the data.
        double mean{common::CBasicStatistics::mean(std::accumulate(
            valuesToTest.begin(), valuesToTest.end(), TMeanAccumulator{},
            [](auto partialMean, const auto& value) {
                partialMean.add(common::CBasicStatistics::mean(value));
                return partialMean;
            }))};
        for (auto& value : valuesToTest) {
            common::CBasicStatistics::moment<0>(value) -= mean;
        }

        // Compute the serial autocorrelations padding to the maximum offset to
        // avoid windowing effects.
        valuesToTest.resize(n + 3 * pad);
        autocorrelations(valuesToTest, placeholder, correlations);
        valuesToTest.resize(n);
        correlations.resize(std::max(3 * pad, periods.back()));

        // In order to handle with smooth varying functions whose autocorrelation
        // is high for small offsets we perform an average the serial correlations
        // of each component over offets P, 2P, ..., mP for mP < n. Note that we
        // need to correct the correlation for longer offsets for the zero pad we
        // append.
        for (auto period : periods) {
            TMeanAccumulator meanCorrelation;
            for (std::size_t offset = period; offset < 3 * pad; offset += period) {
                meanCorrelation.add(static_cast<double>(n) / static_cast<double>(n - offset) *
                                    correlations[offset - 1]);
            }
            correlations[period - 1] = common::CBasicStatistics::mean(meanCorrelation);
            LOG_TRACE(<< "correlation(" << period << ") = " << correlations[period - 1]);
        }

        auto correlationLess = [&](std::size_t lhs, std::size_t rhs) {
            return correlations[lhs - 1] < correlations[rhs - 1];
        };
        std::size_t maxCorrelationPeriod{
            *std::max_element(periods.begin(), periods.end(), correlationLess)};
        double maxCorrelation{correlations[maxCorrelationPeriod - 1]};
        LOG_TRACE(<< "max correlation(" << maxCorrelationPeriod << ") = " << maxCorrelation);

        sizeWithoutComponent = result.size();

        // Exclude if there is a not too much longer period that has significantly
        // higher autocorrelation for which we haven't seen enough repeats.
        if (maxCorrelation < 0.8 * *std::max_element(correlations.begin() + pad,
                                                     correlations.begin() + 3 * pad / 2)) {
            break;
        }

        candidatePeriods.clear();

        // Prefer shorter periods if the decision is close because the model will
        // be more accurate. In particular, if we have a divisor of the selected
        // period whose autocorrelation is within epsilon we'll select that one.
        double cutoff{0.8 * maxCorrelation};
        LOG_TRACE(<< "cutoff = " << cutoff);
        for (std::size_t i = maxCorrelationPeriod / 2; i >= 2; --i) {
            std::size_t period{maxCorrelationPeriod / i};
            if (maxCorrelationPeriod % period == 0 && correlations[period - 1] > cutoff) {
                candidatePeriods.push_back(period);
                break;
            }
        }

        // Check if the selected seasonal component with period p is really an
        // additive combination of two or more shorter seasonal components, i.e.
        // if p mod p_i = 0 for all p_i and each component with period p_i has
        // reasonable autocorrelation.
        if (candidatePeriods.empty()) {
            checkForSeasonalDecomposition(correlations, maxCorrelationPeriod,
                                          cutoff, maxComponents, candidatePeriods);
        }
        LOG_TRACE(<< "candidate periods = "
                  << core::CContainerPrinter::print(candidatePeriods));

        // If we've already selected the candidate components or we've explained
        // nearly all the variance then stop.
        if (std::all_of(candidatePeriods.begin(), candidatePeriods.end(),
                        [&](std::size_t period) {
                            return std::find(selectedPeriods.begin(),
                                             selectedPeriods.end(),
                                             period) != selectedPeriods.end();
                        }) ||
            varianceWithComponent < eps * varianceWithoutComponent) {
            break;
        }

        selectedPeriods.insert(selectedPeriods.end(), candidatePeriods.begin(),
                               candidatePeriods.end());
        valuesToTest.assign(values.begin(), values.begin() + n);

        // Check for weekend/weekday decomposition if the candidate seasonality
        // is daily or weekly.
        if (checkForTradingDayDecomposition(
                valuesToTest, outlierFraction, day, week, decomposition, components,
                candidatePeriods, result, startOfWeekOverride, significantPValue)) {
            valuesToTest.assign(values.begin(), values.begin() + n);
        } else {
            for (auto period : candidatePeriods) {
                appendSeasonalComponentSummary(period, result);
            }
        }
        LOG_TRACE(<< "selected periods = " << core::CContainerPrinter::print(result));

        fitSeasonalComponentsRobust(result, outlierFraction, valuesToTest, components);

        // Here we use a test of the explained variance vs a null hypothesis
        // which doesn't include the components. Note that since we find the
        // maximum the autocorrelation over n = |periods| we expect the chance
        // of seeing the smallest p-value to be chance that n p-values are all
        // greater than the observed p-value or 1 - (1 - p)^n. In practice,
        // this only holds if the p-values are independent, which they clearly
        // aren't. For any correlation we have morally 1 - (1 - p)^{k n} for
        // k < 1. We use k = 0.5.
        auto H1 = residualVarianceStats(valuesToTest, result, components);
        varianceWithoutComponent = H0.s_ResidualVariance;
        varianceWithComponent = H1.s_ResidualVariance;
        pValue = common::CTools::oneMinusPowOneMinusX(
            nestedDecompositionPValue(H0, H1), 0.5 * static_cast<double>(periods.size()));
        LOG_TRACE(<< H1.print() << " vs " << H0.print());
        LOG_TRACE(<< "p-value = " << pValue << ", p-value to accept = " << significantPValue);

        H0 = H1;
        removeComponents(result, components, valuesToTest);

    } while (pValue < significantPValue && result.size() < maxComponents);

    result.resize(sizeWithoutComponent);

    return result;
}

CSignal::TSeasonalComponentVec
CSignal::tradingDayDecomposition(const TFloatMeanAccumulatorVec& values,
                                 double outlierFraction,
                                 std::size_t week,
                                 TOptionalSize startOfWeekOverride,
                                 double significantPValue) {

    using TMeanVarAccumulatorBuffer = boost::circular_buffer<TMeanVarAccumulator>;
    using TMeanVarAccumulatorBufferVec = std::vector<TMeanVarAccumulatorBuffer>;

    constexpr std::size_t WEEKEND_DAILY{0};
    constexpr std::size_t WEEKDAY_DAILY{1};
    if (100 * values.size() < 190 * week || week < 14) {
        return {};
    }

    std::size_t day{(week + 3) / 7};
    std::size_t weekend{(2 * week + 3) / 7};
    std::size_t weekday{(5 * week + 3) / 7};

    TFloatMeanAccumulatorVec temporaryValues{values};

    // Work on the largest subset of the values which is a multiple week.
    std::size_t remainder{temporaryValues.size() % week};
    TFloatMeanAccumulatorCRng valuesToTest{temporaryValues, 0,
                                           temporaryValues.size() - remainder};
    std::size_t n{valuesToTest.size()};
    LOG_TRACE(<< "number values = " << n << "/" << values.size());

    std::size_t startOfWeek{startOfWeekOverride != boost::none && *startOfWeekOverride < week
                                ? *startOfWeekOverride
                                : 0};

    TSeasonalComponentVec dailyPeriod{seasonalComponentSummary(day)};
    TMeanAccumulatorVecVec temporaryComponents;
    fitSeasonalComponentsRobust(dailyPeriod, outlierFraction, temporaryValues,
                                temporaryComponents);
    auto dailyHypothesis =
        residualVarianceStats(temporaryValues, dailyPeriod, temporaryComponents);

    double epsVariance{
        common::CTools::pow2(1000.0 * std::numeric_limits<double>::epsilon()) * [&] {
            TMeanVarAccumulator moments;
            for (const auto& value : values) {
                if (common::CBasicStatistics::count(value) > 0.0) {
                    moments.add(common::CBasicStatistics::mean(value));
                }
            }
            return common::CBasicStatistics::maximumLikelihoodVariance(moments);
        }()};
    LOG_TRACE(<< "daily variance = " << dailyHypothesis.s_ResidualVariance
              << " threshold to test = " << epsVariance);

    if (dailyHypothesis.s_ResidualVariance <= epsVariance) {
        return {};
    }

    temporaryValues = values;
    TSeasonalComponentVec weeklyPeriod{seasonalComponentSummary(week)};
    TMeanAccumulatorVecVec weeklyComponent;
    fitSeasonalComponents(weeklyPeriod, values, weeklyComponent);
    reweightOutliers(weeklyPeriod, weeklyComponent, outlierFraction, temporaryValues);

    // Compute the partitions' index windows.
    TSizeSizePr2Vec weekends;
    TSizeSizePr2Vec weekdays;
    for (std::size_t i = 0; i < n; i += week) {
        weekends.emplace_back(i, i + weekend);
        weekdays.emplace_back(i + weekend, i + week);
    }
    std::size_t strides[]{day, day};
    TSizeSizePr2Vec partitions[]{weekends, weekdays};
    LOG_TRACE(<< "day = " << day << ", weekend = " << weekend << ", weekday = " << weekday);
    LOG_TRACE(<< "weekends = " << core::CContainerPrinter::print(weekends)
              << ", weekdays = " << core::CContainerPrinter::print(weekdays));
    LOG_TRACE(<< "strides = " << core::CContainerPrinter::print(strides));
    LOG_TRACE(<< "partitions = " << core::CContainerPrinter::print(partitions));

    // Initialize the components.
    TMeanVarAccumulatorBuffer components[]{
        TMeanVarAccumulatorBuffer{day, TMeanVarAccumulator{}},
        TMeanVarAccumulatorBuffer{day, TMeanVarAccumulator{}}};
    TMeanVarAccumulatorBufferVec placeholder(1);
    auto initialize = [&](const TSizeSizePr& window, TMeanVarAccumulatorBuffer& component) {
        placeholder[0].swap(component);
        std::size_t period{placeholder[0].size()};
        doFitSeasonalComponents({{period, startOfWeek, week, window}},
                                valuesToTest, placeholder);
        placeholder[0].swap(component);
    };
    initialize({0, weekend}, components[WEEKEND_DAILY]);
    initialize({weekend, week}, components[WEEKDAY_DAILY]);
    LOG_TRACE(<< "components = " << core::CContainerPrinter::print(components));

    TMeanAccumulator variances[std::size(components)];
    for (std::size_t i = 0; i < std::size(components); ++i) {
        variances[i] = std::accumulate(
            components[i].begin(), components[i].end(), TMeanAccumulator{},
            [](auto variance_, const auto& value) {
                variance_.add(common::CBasicStatistics::maximumLikelihoodVariance(value),
                              common::CBasicStatistics::count(value));
                return variance_;
            });
    }

    TDoubleVec candidateVariances(week, 0.0);
    auto captureVarianceAtStartOfWeek = [&](std::size_t i) {
        candidateVariances[i] = common::CBasicStatistics::mean(
            variances[WEEKEND_DAILY] + variances[WEEKDAY_DAILY]);
    };

    // Choose the best partition.

    if (startOfWeekOverride != boost::none && *startOfWeekOverride < week) {
        captureVarianceAtStartOfWeek(startOfWeek);
    } else {
        // Compute the variances for each candidate partition.
        captureVarianceAtStartOfWeek(0);
        for (std::size_t i = 0; i + 1 < week; ++i) {
            for (std::size_t j = 0; j < std::size(components); ++j) {
                TMeanVarAccumulator next;
                for (const auto& subset : partitions[j]) {
                    for (std::size_t k = i + subset.first + strides[j];
                         k <= i + subset.second; k += strides[j]) {
                        next.add(common::CBasicStatistics::mean(valuesToTest[k % n]),
                                 common::CBasicStatistics::count(valuesToTest[k % n]));
                    }
                }
                auto last = components[j].front();
                components[j].push_back(next);
                variances[j] += common::CBasicStatistics::momentsAccumulator(
                    common::CBasicStatistics::count(next),
                    common::CBasicStatistics::maximumLikelihoodVariance(next));
                variances[j] -= common::CBasicStatistics::momentsAccumulator(
                    common::CBasicStatistics::count(last),
                    common::CBasicStatistics::maximumLikelihoodVariance(last));
            }
            captureVarianceAtStartOfWeek(i + 1);
        }

        double minCost{std::numeric_limits<double>::max()};
        startOfWeek = week + 1;

        // If the choice is marginal, we seek to partition where the time series
        // value is absolutely small.
        double varianceThreshold{1.05 * *std::min_element(candidateVariances.begin(),
                                                          candidateVariances.end())};
        for (std::size_t i = 0; i < candidateVariances.size(); ++i) {
            if (candidateVariances[i] < varianceThreshold) {
                double cost{0.0};
                for (std::size_t k = i; k < i + n; k += week) {
                    double knots[]{
                        common::CBasicStatistics::mean(valuesToTest[(k + n - 1) % n]),
                        common::CBasicStatistics::mean(valuesToTest[(k + n + 0) % n]),
                        common::CBasicStatistics::mean(
                            valuesToTest[(k + n + weekend - 1) % n]),
                        common::CBasicStatistics::mean(
                            valuesToTest[(k + n + weekend + 0) % n])};
                    cost += std::fabs(knots[0]) + std::fabs(knots[1]) +
                            std::fabs(knots[2]) + std::fabs(knots[3]) +
                            std::fabs(knots[1] - knots[0]) +
                            std::fabs(knots[3] - knots[2]);
                }
                LOG_TRACE(<< "cost(" << i << ") = " << cost);
                std::tie(minCost, startOfWeek) = std::min(
                    std::make_pair(minCost, startOfWeek), std::make_pair(cost, i));
            }
        }
    }
    LOG_TRACE(<< "start of week = " << startOfWeek << "/" << week);

    // Check if there is reasonable evidence for weekday/weekend partition.
    TSeasonalComponentVec weekendPeriods{
        {week, startOfWeek, week, TSizeSizePr{0 * day, 2 * day}},
        {day, startOfWeek, week, TSizeSizePr{2 * day, 7 * day}}};
    temporaryValues = values;
    fitSeasonalComponentsRobust(weekendPeriods, outlierFraction,
                                temporaryValues, temporaryComponents);
    auto weekendHypothesis = residualVarianceStats(temporaryValues, weekendPeriods,
                                                   temporaryComponents);
    double pValue{common::CTools::oneMinusPowOneMinusX(
        nestedDecompositionPValue(dailyHypothesis, weekendHypothesis),
        0.5 * static_cast<double>(week))};
    weekendPeriods.emplace_back(day, startOfWeek, week, TSizeSizePr{0 * day, 2 * day});
    weekendPeriods.emplace_back(week, startOfWeek, week, TSizeSizePr{2 * day, 7 * day});
    std::sort(weekendPeriods.begin(), weekendPeriods.end());
    LOG_TRACE(<< "p-value = " << pValue);
    return pValue >= significantPValue ? TSeasonalComponentVec{} : weekendPeriods;
}

void CSignal::fitSeasonalComponents(const TSeasonalComponentVec& periods,
                                    const TFloatMeanAccumulatorVec& values,
                                    TMeanAccumulatorVecVec& components) {
    doFitSeasonalComponents(periods, values, components);
}

void CSignal::fitSeasonalComponentsRobust(const TSeasonalComponentVec& periods,
                                          double outlierFraction,
                                          TFloatMeanAccumulatorVec& values,
                                          TMeanAccumulatorVecVec& components) {
    fitSeasonalComponents(periods, values, components);
    if (outlierFraction > 0.0) {
        reweightOutliers(periods, components, outlierFraction, values);
        fitSeasonalComponents(periods, values, components);
    }
}

bool CSignal::reweightOutliers(const TSeasonalComponentVec& periods,
                               const TMeanAccumulatorVecVec& components,
                               double fraction,
                               TFloatMeanAccumulatorVec& values) {
    auto predictor = [&](std::size_t index) {
        double value{0.0};
        for (std::size_t i = 0; i < components.size(); ++i) {
            value += periods[i].value(components[i], index);
        }
        return value;
    };
    return reweightOutliers(predictor, fraction, values);
}

bool CSignal::reweightOutliers(const TIndexPredictor& predictor,
                               double fraction,
                               TFloatMeanAccumulatorVec& values) {

    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TMaxAccumulator =
        common::CBasicStatistics::COrderStatisticsHeap<TDoubleSizePr, std::greater<TDoubleSizePr>>;

    std::size_t numberOutliers{static_cast<std::size_t>([&] {
        return fraction * static_cast<double>(countNotMissing(values));
    }())};
    LOG_TRACE(<< "number outliers = " << numberOutliers);

    if (numberOutliers == 0) {
        return false;
    }

    TMaxAccumulator outliers{2 * numberOutliers};
    TMeanAccumulator meanDifference;
    TMeanVarAccumulator predictionMoments;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (common::CBasicStatistics::count(values[i]) > 0.0) {
            double prediction{predictor(i)};
            double difference{std::fabs(common::CBasicStatistics::mean(values[i]) - prediction)};
            outliers.add({difference, i});
            meanDifference.add(difference);
            predictionMoments.add(std::fabs(common::CBasicStatistics::mean(values[i])));
        }
    }
    if (common::CBasicStatistics::mean(meanDifference) == 0.0) {
        return false;
    }

    outliers.sort();
    LOG_TRACE(<< "outliers = " << core::CContainerPrinter::print(outliers));

    TMeanAccumulator meanDifferenceOfOutliers;
    for (std::size_t i = 0; 4 * i < outliers.count(); ++i) {
        meanDifferenceOfOutliers.add(outliers[i].first);
    }
    meanDifference -= meanDifferenceOfOutliers;
    double threshold{std::max(
        3.0 * common::CBasicStatistics::mean(meanDifference),
        0.05 * std::sqrt(common::CBasicStatistics::variance(predictionMoments)))};
    LOG_TRACE(<< "threshold = " << common::CBasicStatistics::mean(meanDifference));

    bool result{false};

    double logThreshold{std::log(threshold)};
    for (const auto& outlier : outliers) {
        double logDifference{std::log(outlier.first)};
        double weight{common::CTools::linearlyInterpolate(logThreshold - LOG_TWO, logThreshold,
                                                          1.0, 0.1, logDifference) *
                      common::CTools::linearlyInterpolate(logThreshold, logThreshold + LOG_SIXTEEN,
                                                          1.0, 0.1, logDifference)};
        common::CBasicStatistics::count(values[outlier.second]) *= weight;
        result |= (weight < 1.0);
    }
    LOG_TRACE(<< "values - outliers = " << core::CContainerPrinter::print(values));

    return result;
}

double CSignal::meanNumberRepeatedValues(const TFloatMeanAccumulatorVec& values,
                                         const SSeasonalComponentSummary& period) {
    TDoubleVec counts(period.period(), 0.0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (common::CBasicStatistics::count(values[i]) > 0.0 && period.contains(i)) {
            counts[period.offset(i)] += 1.0;
        }
    }
    return common::CBasicStatistics::mean(
        std::accumulate(counts.begin(), counts.end(), TMeanAccumulator{},
                        [](TMeanAccumulator mean, double count) {
                            if (count > 0.0) {
                                mean.add(count);
                            }
                            return mean;
                        }));
}

CSignal::SVarianceStats
CSignal::residualVarianceStats(const TFloatMeanAccumulatorVec& values,
                               const TSeasonalComponentVec& periods,
                               const TMeanAccumulatorVecVec& components) {
    std::size_t numberValues{0};
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (common::CBasicStatistics::count(values[i]) > 0.0 &&
            std::any_of(periods.begin(), periods.end(),
                        [i](const auto& period) { return period.contains(i); })) {
            ++numberValues;
        }
    }
    std::size_t parameters{0};
    for (const auto& component : components) {
        parameters += countNotMissing(component);
    }
    return {residualVariance(values, periods, components,
                             [](const TFloatMeanAccumulator&) { return 1.0; }),
            residualVariance(values, periods, components),
            static_cast<double>(numberValues) - static_cast<double>(parameters), parameters};
}

double CSignal::residualVariance(const TFloatMeanAccumulatorVec& values,
                                 const TSeasonalComponentVec& periods,
                                 const TMeanAccumulatorVecVec& components,
                                 const TMomentWeightFunc& weight) {
    TMeanVarAccumulator moments;
    for (std::size_t i = 0; i < values.size(); ++i) {
        bool skip{true};
        double prediction{0.0};
        if (common::CBasicStatistics::count(values[i]) > 0.0) {
            for (std::size_t j = 0; j < periods.size(); ++j) {
                if (periods[j].contains(i)) {
                    skip = false;
                    prediction += common::CBasicStatistics::mean(
                        components[j][periods[j].offset(i)]);
                }
            }
        }
        if (skip == false) {
            double value{common::CBasicStatistics::mean(values[i])};
            moments.add(value - prediction, weight(values[i]));
        }
    }
    return common::CBasicStatistics::maximumLikelihoodVariance(moments);
}

double CSignal::nestedDecompositionPValue(const SVarianceStats& H0,
                                          const SVarianceStats& H1) {
    if (H1.s_DegreesFreedom <= 0.0 || // Insufficient data to test H1
        H1.s_NumberParameters <= H0.s_NumberParameters || // H1 is not nested
        H0.s_ResidualVariance == 0.0) { // The values were constant
        return 1.0;
    }

    double eps{std::numeric_limits<double>::epsilon()};
    double v0[]{H0.s_ResidualVariance, std::max(H0.s_TruncatedResidualVariance,
                                                eps * H0.s_ResidualVariance)};
    double v1[]{std::max(H1.s_ResidualVariance, eps * v0[0]),
                std::max(H1.s_TruncatedResidualVariance, eps * v0[1])};
    double df[]{static_cast<double>(H1.s_NumberParameters - H0.s_NumberParameters),
                H1.s_DegreesFreedom};

    // This assumes that H1 is nested in H0.
    double F[]{(df[1] * std::max(v0[0] - v1[0], 0.0)) / (df[0] * v1[0]),
               (df[1] * std::max(v0[1] - v1[1], 0.0)) / (df[0] * v1[1])};
    return std::min(common::CStatisticalTests::rightTailFTest(F[0], df[0], df[1]),
                    common::CStatisticalTests::rightTailFTest(F[1], df[0], df[1]));
}

std::size_t CSignal::selectComponentSize(const TFloatMeanAccumulatorVec& values,
                                         std::size_t period) {

    auto interpolate = [&](std::size_t i, const std::size_t* adjacent,
                           const TMeanAccumulatorVec& centres,
                           const TMeanAccumulatorVec& model) {
        double prediction{0.0};
        double adjacentCentres[]{common::CBasicStatistics::mean(centres[adjacent[0]]),
                                 common::CBasicStatistics::mean(centres[adjacent[1]]),
                                 common::CBasicStatistics::mean(centres[adjacent[2]])};
        double distances[]{std::fabs(adjacentCentres[0] - static_cast<double>(i)),
                           std::fabs(adjacentCentres[1] - static_cast<double>(i)),
                           std::fabs(adjacentCentres[2] - static_cast<double>(i))};
        double Z{0.0};
        for (std::size_t j = 0; j < 3; ++j) {
            distances[j] = std::min(distances[j],
                                    static_cast<double>(period) - distances[j]);
            double weight{std::exp(-distances[j])};
            prediction += weight * common::CBasicStatistics::mean(model[adjacent[j]]);
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
            moments.add(common::CBasicStatistics::mean(values[i]) - prediction,
                        common::CBasicStatistics::count(values[i]));
        }
        return common::CBasicStatistics::maximumLikelihoodVariance(moments);
    };

    TMeanAccumulatorVec centres(period);
    for (std::size_t i = 0; i < centres.size(); ++i) {
        centres[i].add(static_cast<double>(i % period));
    }

    TMeanAccumulatorVecVec component;
    fitSeasonalComponents({seasonalComponentSummary(period)}, values, component);
    TMeanAccumulatorVec compressedComponent{std::move(component[0])};

    std::size_t size{period};

    std::size_t H0{1};
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
                                    common::CBasicStatistics::count(values[j]));
                compressedComponent[bucket].add(
                    common::CBasicStatistics::mean(values[j]),
                    common::CBasicStatistics::count(values[j]));
            }
            LOG_TRACE(<< "centres = " << core::CContainerPrinter::print(centres));

            degreesFreedom[H0] = static_cast<double>(values.size() - period / i);
            variances[H0] = residualVariance(centres, compressedComponent);
            LOG_TRACE(<< "degrees freedom = "
                      << core::CContainerPrinter::print(degreesFreedom));
            LOG_TRACE(<< "variances = " << core::CContainerPrinter::print(variances));

            if ((variances[H0] > 0.0 && variances[1 - H0] == 0.0) ||
                common::CStatisticalTests::rightTailFTest(
                    variances[H0] == variances[1 - H0] ? 1.0 : variances[H0] / variances[1 - H0],
                    degreesFreedom[H0], degreesFreedom[1 - H0]) < 0.1) {
                break;
            }
            if ((variances[1 - H0] > 0.0 && variances[H0] == 0.0) ||
                common::CStatisticalTests::rightTailFTest(
                    variances[1 - H0] == variances[H0] ? 1.0 : variances[1 - H0] / variances[H0],
                    degreesFreedom[1 - H0], degreesFreedom[H0]) < 0.1) {
                H0 = 1 - H0;
            }
            size = compressedComponent.size();
        }
    }

    LOG_TRACE(<< "selected size = " << size);

    return size;
}

CSignal::TMeanAccumulatorVec
CSignal::smoothResample(std::size_t size, TMeanAccumulatorVec component) {

    if (size >= component.size()) {
        return component;
    }

    // Pass size of zero is really an error. We'll handle this as best we can
    // and log an error.
    if (size == 0) {
        LOG_ERROR(<< "Smooth resample passed component size of zero! Using one.");
        size = 1;
    }

    // Smooth by convolving with a triangle function.

    int n{static_cast<int>(component.size())};
    int width{static_cast<int>((n + size - 1) / size)};
    TMeanAccumulatorVec smooth(n);
    for (int i = 0; i < n; ++i) {
        if (common::CBasicStatistics::count(component[i]) > 0.0) {
            double Z{0.0};
            for (int j = i - width + 1; j < i + width; ++j) {
                const auto& value = component[(n + j) % n];
                if (common::CBasicStatistics::count(value) > 0.0) {
                    double weight{static_cast<double>(
                        width - (std::max(i, j) - std::min(i, j)))};
                    smooth[i].add(common::CBasicStatistics::mean(value),
                                  weight * common::CBasicStatistics::count(value));
                    Z += weight;
                }
            }
            common::CBasicStatistics::count(smooth[i]) /= Z;
        }
    }

    return smooth;
}

CSignal::TPredictor CSignal::bucketPredictor(const TPredictor& predictor,
                                             core_t::TTime bucketsStartTime,
                                             core_t::TTime bucketLength,
                                             core_t::TTime bucketAverageSampleTime,
                                             core_t::TTime sampleInterval) {

    // We assume that we have samples at some approximately fixed interval l and
    // a bucket length L >= l. We also know
    //
    //   bucketAverageSampleTime = l / L sum_{0<=i<n}{ t0 + i l }             (1)
    //
    // where n = L / l. This summation is n t0 + n(n - 1)/2 l and we can use this
    // to solve (1) for t0. The prediction of the bucket average for the bucket
    // starting at t is then
    //
    //   1 / n sum_{0<=i<n}{ p(t + t0 + i * l) }

    double n{static_cast<double>(bucketLength) / static_cast<double>(sampleInterval)};
    core_t::TTime offset{static_cast<core_t::TTime>(
        std::max(static_cast<double>(bucketAverageSampleTime) -
                     0.5 * (n - 1.0) * static_cast<double>(sampleInterval),
                 0.0))};

    return [=](core_t::TTime time) {
        TMeanAccumulator result;
        time += bucketsStartTime;
        for (core_t::TTime dt = offset; dt < bucketLength; dt += sampleInterval) {
            result.add(predictor(time + dt));
        }
        return common::CBasicStatistics::mean(result);
    };
}

void CSignal::removeComponents(const TSeasonalComponentVec& periods,
                               const TMeanAccumulatorVecVec& components,
                               TFloatMeanAccumulatorVec& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (common::CBasicStatistics::count(values[i]) > 0.0) {
            for (std::size_t j = 0; j < components.size(); ++j) {
                common::CBasicStatistics::moment<0>(values[i]) -=
                    periods[j].value(components[j], i);
            }
        }
    }
}

void CSignal::checkForSeasonalDecomposition(const TDoubleVec& correlations,
                                            std::size_t maxCorrelationPeriod,
                                            double cutoff,
                                            std::size_t maxComponents,
                                            TSizeVec& candidatePeriods) {
    double correlation{0.0};
    while (candidatePeriods.size() < maxComponents) {
        std::size_t nextPeriod{0};
        double nextCorrelation{0.0};
        for (std::size_t i = maxCorrelationPeriod / 2; i >= 2; --i) {
            std::size_t period{maxCorrelationPeriod / i};
            if (maxCorrelationPeriod % period == 0 &&
                std::find_if(candidatePeriods.begin(), candidatePeriods.end(),
                             [&](const auto& candidatePeriod) {
                                 return candidatePeriod % period == 0 ||
                                        period % candidatePeriod == 0;
                             }) == candidatePeriods.end() &&
                correlations[period - 1] > nextCorrelation) {
                nextPeriod = period;
                nextCorrelation = correlations[period - 1];
            }
        }

        if (nextCorrelation > 0.2 && correlation < cutoff) {
            candidatePeriods.push_back(nextPeriod);
            correlation += nextCorrelation;
        } else {
            break;
        }
    }
    if (candidatePeriods.empty() || correlation < cutoff) {
        candidatePeriods.assign(1, maxCorrelationPeriod);
    }
}

bool CSignal::checkForTradingDayDecomposition(TFloatMeanAccumulatorVec& values,
                                              double outlierFraction,
                                              std::size_t day,
                                              std::size_t week,
                                              TSeasonalComponentVec& decomposition,
                                              TMeanAccumulatorVecVec& components,
                                              TSizeVec& candidatePeriods,
                                              TSeasonalComponentVec& result,
                                              TOptionalSize startOfWeekOverride,
                                              double significantPValue) {

    if (std::find_if(candidatePeriods.begin(), candidatePeriods.end(),
                     [&](const auto& period) { return period == week; }) !=
        candidatePeriods.end()) {

        decomposition.clear();
        for (const auto& period : candidatePeriods) {
            if (period != day && period != week) {
                appendSeasonalComponentSummary(period, decomposition);
            }
        }
        if (decomposition.size() > 0) {
            fitSeasonalComponents(decomposition, values, components);
            removeComponents(decomposition, components, values);
        }

        decomposition = tradingDayDecomposition(
            values, outlierFraction, week, startOfWeekOverride, significantPValue);

        if (decomposition.size() > 0) {
            for (std::size_t i = 0; i + 1 < candidatePeriods.size(); ++i) {
                if (candidatePeriods[i] != day) {
                    appendSeasonalComponentSummary(candidatePeriods[i], result);
                } else {
                    result.insert(result.end(), decomposition.begin(),
                                  decomposition.end());
                    decomposition.clear();
                }
            }
            result.insert(result.end(), decomposition.begin(), decomposition.end());
        }
    }
    return candidatePeriods.back() == week;
}

template<typename VALUES, typename COMPONENT>
void CSignal::doFitSeasonalComponents(const TSeasonalComponentVec& periods,
                                      const VALUES& values,
                                      std::vector<COMPONENT>& components) {
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
                    common::CBasicStatistics::mean(values[i]) - predictor(i),
                    common::CBasicStatistics::count(values[i]));
            }
        }
    }
}
}
}
}
