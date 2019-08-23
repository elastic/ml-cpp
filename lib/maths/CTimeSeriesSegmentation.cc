/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesSegmentation.h>

#include <core/CTriple.h>

#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTools.h>

#include <numeric>

namespace ml {
namespace maths {
namespace {
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
using TRegression = CLeastSquaresOnlineRegression<1, double>;
using TDoubleDoubleFunc = std::function<double(double)>;
using TDoubleSizeFunc = std::function<double(std::size_t)>;

//! Returns 1.0.
double unit(std::size_t) {
    return 1.0;
}

//! Fit a linear model to the values in [\p begin, \p end).
template<typename ITR>
TRegression fitLinearModel(ITR begin, ITR end, double startTime, double dt) {
    TRegression result;
    for (double time = startTime + dt / 2.0; begin != end; ++begin) {
        double x{CBasicStatistics::mean(*begin)};
        double w{CBasicStatistics::count(*begin)};
        if (w > 0.0) {
            result.add(time, x, w);
        }
        time += dt;
    }
    return result;
}

//! Fit a periodic model of period \p period to the values [\p begin, \p end).
template<typename ITR>
TDoubleVec
fitPeriodicModel(ITR begin, ITR end, std::size_t period, const TDoubleSizeFunc& scale) {
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    TMeanAccumulatorVec levels(period);
    for (std::size_t i = 0; begin != end; ++i, ++begin) {
        double x{CBasicStatistics::mean(*begin) / scale(i)};
        double w{CBasicStatistics::count(*begin) * scale(i)};
        if (w > 0.0) {
            levels[i % period].add(x, w);
        }
    }
    TDoubleVec result(period);
    for (std::size_t i = 0; i < period; ++i) {
        result[i] = CBasicStatistics::mean(levels[i]);
    }
    return result;
}

//! Re-weight the outlying values in [\p begin, \p end) w.r.t. the
//! predictions from \p predict.
template<typename ITR>
bool reweightOutliers(ITR begin,
                      ITR end,
                      double startTime,
                      double dt,
                      const TDoubleDoubleFunc& predict,
                      double fraction,
                      double weight) {
    using TDoublePtrPr = std::pair<double, typename std::iterator_traits<ITR>::value_type*>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<TDoublePtrPr, std::greater<TDoublePtrPr>>;

    std::size_t n{static_cast<std::size_t>(
        fraction * static_cast<double>(std::distance(begin, end)))};
    if (n > 0) {
        LOG_TRACE(<< "  # outliers = " << n);
        TMaxAccumulator outliers{n};
        for (double time = startTime + dt / 2.0; begin != end; time += dt, ++begin) {
            double diff{CBasicStatistics::mean(*begin) - predict(time)};
            outliers.add({std::fabs(diff), &(*begin)});
        }
        LOG_TRACE(<< "  outliers = " << core::CContainerPrinter::print(outliers));
        for (auto outlier : outliers) {
            CBasicStatistics::count(*outlier.second) *= weight;
        }
        return true;
    }
    return false;
}

//! Re-weight the outlying values in [\p begin, \p end) w.r.t. the
//! predictions from \p predict.
template<typename ITR>
bool reweightOutliers(ITR begin, ITR end, const TDoubleSizeFunc& predict, double fraction, double weight) {
    using TDoublePtrPr = std::pair<double, typename std::iterator_traits<ITR>::value_type*>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<TDoublePtrPr, std::greater<TDoublePtrPr>>;

    std::size_t n{static_cast<std::size_t>(
        fraction * static_cast<double>(std::distance(begin, end)))};
    if (n > 0) {
        LOG_TRACE(<< "  # outliers = " << n);
        TMaxAccumulator outliers{n};
        for (std::size_t i = 0; begin != end; ++i, ++begin) {
            double diff{CBasicStatistics::mean(*begin) - predict(i)};
            outliers.add({std::fabs(diff), &(*begin)});
        }
        LOG_TRACE(<< "  outliers = " << core::CContainerPrinter::print(outliers));
        for (auto outlier : outliers) {
            CBasicStatistics::count(*outlier.second) *= weight;
        }
        return true;
    }
    return false;
}

//! Compute the moments of the values in [\p begin, \p end) after subtracting
//! the predictions of \p model.
template<typename ITR>
TMeanVarAccumulator
residualMoments(ITR begin, ITR end, double startTime, double dt, const TRegression& model) {
    TMeanVarAccumulator moments;
    TRegression::TArray params;
    model.parameters(params);
    for (double time = startTime + dt / 2.0; begin != end; ++begin, time += dt) {
        double w{CBasicStatistics::count(*begin)};
        if (w > 0.0) {
            double x{CBasicStatistics::mean(*begin) - TRegression::predict(params, time)};
            moments.add(x, w);
        }
    }
    return moments;
}

//! Compute the BIC of the linear model fit to [\p begin, \p end).
template<typename ITR>
TMeanVarAccumulator centredResidualMoments(ITR begin, ITR end, double startTime, double dt) {
    if (begin == end) {
        return CBasicStatistics::momentsAccumulator(0.0, 0.0, 0.0);
    }
    if (std::distance(begin, end) == 1) {
        return CBasicStatistics::momentsAccumulator(CBasicStatistics::count(*begin),
                                                    0.0, 0.0);
    }
    TRegression model{fitLinearModel(begin, end, startTime, dt)};
    TMeanVarAccumulator moments{residualMoments(begin, end, startTime, dt, model)};
    CBasicStatistics::moment<0>(moments) = 0.0;
    return moments;
}

//! Compute the BIC of the scaled periodic model for the values in
//! [\p begin, \p end).
template<typename ITR>
TMeanVarAccumulator
centredResidualMoments(ITR begin, ITR end, std::size_t offset, const TDoubleVec& model) {
    if (begin == end) {
        return CBasicStatistics::momentsAccumulator(0.0, 0.0, 0.0);
    }
    if (std::distance(begin, end) == 1) {
        return CBasicStatistics::momentsAccumulator(CBasicStatistics::count(*begin),
                                                    0.0, 0.0);
    }
    std::size_t period{model.size()};

    TMeanAccumulator projection;
    TMeanAccumulator norm2;
    for (ITR i = begin; i != end; ++i) {
        double x{CBasicStatistics::mean(*i)};
        double w{CBasicStatistics::count(*i)};
        double p{model[(offset + std::distance(begin, i)) % period]};
        if (w > 0.0) {
            projection.add(x * p, w);
            norm2.add(p * p, w);
        }
    }
    double scale{CBasicStatistics::mean(projection) == CBasicStatistics::mean(norm2)
                     ? 1.0
                     : CBasicStatistics::mean(projection) / CBasicStatistics::mean(norm2)};
    LOG_TRACE(<< "  scale = " << scale);

    TMeanVarAccumulator moments;
    for (ITR i = begin; i != end; ++i) {
        double x{CBasicStatistics::mean(*i)};
        double w{CBasicStatistics::count(*i)};
        double p{scale * model[(offset + std::distance(begin, i)) % period]};
        if (w > 0.0) {
            moments.add(x - p, w);
        }
    }
    CBasicStatistics::moment<0>(moments) = 0.0;
    LOG_TRACE(<< "  moments = " << moments);

    return moments;
}
}

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::piecewiseLinear(const TFloatMeanAccumulatorVec& values,
                                         double significanceToSegment,
                                         double outlierFraction,
                                         double outlierWeight) {
    TSizeVec segmentation{0, values.size()};
    double dt{10.0 / static_cast<double>(values.size())};
    fitTopDownPiecewiseLinear(values.cbegin(), values.cend(), 0, 0.0, dt, significanceToSegment,
                              outlierFraction, outlierWeight, segmentation);
    std::sort(segmentation.begin(), segmentation.end());
    return segmentation;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinear(TFloatMeanAccumulatorVec values,
                                               const TSizeVec& segmentation,
                                               double outlierFraction,
                                               double outlierWeight) {
    auto predict = fitPiecewiseLinear(values, segmentation, outlierFraction, outlierWeight);
    double dt{10.0 / static_cast<double>(values.size())};
    double time{dt / 2.0};
    for (auto i = values.begin(); i != values.end(); ++i, time += dt) {
        if (CBasicStatistics::count(*i) > 0.0) {
            CBasicStatistics::moment<0>(*i) -= predict(time);
        }
    }
    return values;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinearDiscontinuities(TFloatMeanAccumulatorVec values,
                                                              const TSizeVec& segmentation,
                                                              double outlierFraction,
                                                              double outlierWeight) {
    TFloatMeanAccumulatorVec reweighted(values);
    auto predict = fitPiecewiseLinear(reweighted, segmentation, outlierFraction, outlierWeight);
    double dt{10.0 / static_cast<double>(values.size())};
    TDoubleVec steps;
    steps.reserve(segmentation.size() - 1);
    for (std::size_t i = 1; i + 1 < segmentation.size(); ++i) {
        double time{static_cast<double>(2 * segmentation[i] + 1) * dt / 2.0};
        steps.push_back(predict(time - dt / 20) - predict(time + dt / 20));
    }
    steps.push_back(0.0);
    LOG_TRACE(<< "steps = " << core::CContainerPrinter::print(steps));
    std::partial_sum(steps.rbegin(), steps.rend(), steps.rbegin());
    for (std::size_t i = segmentation.size() - 1; i > 0; --i) {
        LOG_TRACE(<< "Shifting [" << segmentation[i - 1] << ","
                  << segmentation[i] << ") by " << steps[i - 1]);
        for (std::size_t j = segmentation[i - 1]; j < segmentation[i]; ++j) {
            CBasicStatistics::moment<0>(values[j]) -= steps[i - 1];
        }
    }
    return values;
}

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::piecewiseLinearScaledPeriodic(const TFloatMeanAccumulatorVec& values,
                                                       std::size_t period,
                                                       double significanceToSegment,
                                                       double outlierFraction,
                                                       double outlierWeight) {
    TFloatMeanAccumulatorVec values_(values);
    TDoubleVec model(fitPeriodicModel(values.begin(), values.end(), period, unit));
    reweightOutliers(values_.begin(), values_.end(),
                     [&model](std::size_t i) { return model[i % model.size()]; },
                     outlierFraction, outlierWeight);
    model = fitPeriodicModel(values_.begin(), values_.end(), period, unit);

    TSizeVec segmentation{0, values.size()};
    fitTopDownPiecewiseLinearScaledPeriodic(values_.cbegin(), values_.cend(), 0, model,
                                            significanceToSegment, segmentation);
    std::sort(segmentation.begin(), segmentation.end());

    return segmentation;
}

CTimeSeriesSegmentation::TDoubleVecDoubleVecPr
CTimeSeriesSegmentation::piecewiseLinearScaledPeriodic(const TFloatMeanAccumulatorVec& values,
                                                       std::size_t period,
                                                       const TSizeVec& segmentation,
                                                       double outlierFraction,
                                                       double outlierWeight) {
    TFloatMeanAccumulatorVec reweighted;
    TDoubleVec model;
    TDoubleVec scales;
    fitPiecewiseLinearScaledPeriodic(values, period, segmentation, outlierFraction,
                                     outlierWeight, reweighted, model, scales);
    return {model, scales};
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinearScaledPeriodic(const TFloatMeanAccumulatorVec& values,
                                                             std::size_t period,
                                                             const TSizeVec& segmentation,
                                                             double outlierFraction,
                                                             double outlierWeight) {
    TFloatMeanAccumulatorVec reweighted;
    TDoubleVec model;
    TDoubleVec scales;
    fitPiecewiseLinearScaledPeriodic(values, period, segmentation, outlierFraction,
                                     outlierWeight, reweighted, model, scales);

    auto predict = [&](std::size_t i) {
        auto right = std::upper_bound(segmentation.begin(), segmentation.end(), i);
        return scales[right - segmentation.begin() - 1] * model[i % period];
    };

    for (std::size_t i = 0; i != reweighted.size(); ++i) {
        if (CBasicStatistics::count(reweighted[i]) > 0.0) {
            CBasicStatistics::moment<0>(reweighted[i]) -= predict(i);
        }
    }

    return reweighted;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinearScaledPeriodic(TFloatMeanAccumulatorVec values,
                                                             const TSizeVec& segmentation,
                                                             const TDoubleVec& model,
                                                             const TDoubleVec& scales) {
    std::size_t period{model.size()};

    auto predict = [&](std::size_t i) {
        auto right = std::upper_bound(segmentation.begin(), segmentation.end(), i);
        return scales[right - segmentation.begin() - 1] * model[i % period];
    };

    for (std::size_t i = 0; i != values.size(); ++i) {
        if (CBasicStatistics::count(values[i]) > 0.0) {
            CBasicStatistics::moment<0>(values[i]) -= predict(i);
        }
    }

    return values;
}

template<typename ITR>
void CTimeSeriesSegmentation::fitTopDownPiecewiseLinear(ITR begin,
                                                        ITR end,
                                                        std::size_t offset,
                                                        double startTime,
                                                        double dt,
                                                        double significanceToSegment,
                                                        double outlierFraction,
                                                        double outlierWeight,
                                                        TSizeVec& segmentation) {
    // We want to find the partition of [begin, end) into linear models which
    // efficiently predicts the values. To this end we maximize R^2 whilst
    // maintaining a parsimonious model. We achieve the later objective by
    // doing model selection for each candidate segment using the significance
    // of the variance ratio.

    using TVec = std::vector<typename std::iterator_traits<ITR>::value_type>;

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};
    TMeanVarAccumulator moments;
    TVec values{begin, end};
    {
        TRegression model{fitLinearModel(values.cbegin(), values.cend(), startTime, dt)};
        auto predict = [&model](double time) { return model.predict(time); };
        reweightOutliers(values.begin(), values.end(), startTime, dt, predict,
                         outlierFraction, outlierWeight);
        moments = centredResidualMoments(values.begin(), values.end(), startTime, dt);
    }

    LOG_TRACE(<< "Considering splitting [" << offset << "," << offset + range << ")");

    if (range >= 6 && CBasicStatistics::variance(moments) > 0.0) {
        using TDoubleDoubleItrTr = core::CTriple<double, double, ITR>;
        using TMinAccumulator =
            CBasicStatistics::COrderStatisticsStack<TDoubleDoubleItrTr, 1>;

        auto variance = [](const TMeanVarAccumulator& l, const TMeanVarAccumulator& r) {
            double nl{CBasicStatistics::count(l)};
            double nr{CBasicStatistics::count(r)};
            return (nl * CBasicStatistics::variance(l) + nr * CBasicStatistics::variance(r)) /
                   (nl + nr);
        };

        // We iterate through every possible binary split of the data into
        // contiguous ranges looking for the split which minimizes the total
        // variance of the values minus linear model predictions. Outliers
        // are removed based on an initial best split.

        TMinAccumulator minResidualVariance;
        values.assign(begin, end);

        for (auto pass : {0, 1}) {
            LOG_TRACE(<< "  pass = " << pass);

            int step{3};
            ITR i = values.cbegin();
            minResidualVariance = TMinAccumulator{};
            double splitTime{startTime};
            TRegression leftModel;
            TRegression rightModel{fitLinearModel(i, values.cend(), splitTime, dt)};

            auto varianceAfterStep = [&](int s) {
                TRegression deltaModel{fitLinearModel(i, i + s, splitTime, dt)};
                return variance(residualMoments(values.cbegin(), i + s, startTime,
                                                dt, leftModel + deltaModel),
                                residualMoments(i + s, values.cend(), splitTime + s * dt,
                                                dt, rightModel - deltaModel));
            };

            for (/**/; i + step < values.cend(); i += step, splitTime += step * dt) {
                if (minResidualVariance.add({varianceAfterStep(step),
                                             splitTime + step * dt, i + step})) {
                    for (int s = 1; s < step; ++s) {
                        minResidualVariance.add(
                            {varianceAfterStep(s), splitTime + s * dt, i + s});
                    }
                }
                TRegression deltaModel{fitLinearModel(i, i + step, splitTime, dt)};
                leftModel += deltaModel;
                rightModel -= deltaModel;
            }

            if (outlierFraction < 0.0 || outlierFraction >= 1.0) {
                LOG_ERROR(<< "Ignoring bad value for outlier fraction " << outlierFraction
                          << ". This must be in the range [0,1).");
                break;
            } else if (pass == 1 || outlierFraction == 0.0) {
                break;
            } else {
                splitTime = minResidualVariance[0].second;
                ITR split{minResidualVariance[0].third};
                leftModel = fitLinearModel(values.cbegin(), split, startTime, dt);
                rightModel = fitLinearModel(split, values.cend(), splitTime, dt);
                auto predict = [splitTime, &leftModel, &rightModel](double time) {
                    double eps{100.0 * std::numeric_limits<double>::epsilon()};
                    return (time + eps < splitTime ? leftModel : rightModel).predict(time);
                };
                reweightOutliers(values.begin(), values.end(), startTime, dt,
                                 predict, outlierFraction, outlierWeight);
            }
        }

        double splitTime{minResidualVariance[0].second};
        ITR split{minResidualVariance[0].third};
        std::size_t splitOffset(std::distance(values.cbegin(), split));
        LOG_TRACE(<< "  candidate = " << offset + splitOffset
                  << ", variance = " << minResidualVariance[0].first);

        TMeanVarAccumulator leftMoments{
            centredResidualMoments(values.cbegin(), split, startTime, dt)};
        TMeanVarAccumulator rightMoments{
            centredResidualMoments(split, values.cend(), splitTime, dt)};
        TMeanAccumulator mean{
            std::accumulate(values.begin(), values.end(), TMeanAccumulator{})};
        LOG_TRACE(<< "  total moments = " << moments << ", left moments = " << leftMoments
                  << ", right moments = " << rightMoments << ", mean = " << mean);
        double f{CBasicStatistics::variance(moments) /
                 (CBasicStatistics::variance(leftMoments + rightMoments) +
                  MINIMUM_COEFFICIENT_OF_VARIATION * std::fabs(CBasicStatistics::mean(mean)))};
        double significance{CStatisticalTests::rightTailFTest(
            f, static_cast<double>(range - 3), static_cast<double>(range - 5))};
        LOG_TRACE(<< "  significance = " << significance);

        if (significance < significanceToSegment) {
            fitTopDownPiecewiseLinear(begin, begin + splitOffset, offset, startTime,
                                      dt, significanceToSegment, outlierFraction,
                                      outlierWeight, segmentation);
            fitTopDownPiecewiseLinear(begin + splitOffset, end, offset + splitOffset,
                                      splitTime, dt, significanceToSegment,
                                      outlierFraction, outlierWeight, segmentation);
            segmentation.push_back(offset + splitOffset);
        }
    }
}

CTimeSeriesSegmentation::TPredictor
CTimeSeriesSegmentation::fitPiecewiseLinear(TFloatMeanAccumulatorVec& values,
                                            const TSizeVec& segmentation,
                                            double outlierFraction,
                                            double outlierWeight) {
    using TRegressionVec = std::vector<TRegression>;

    double dt{10.0 / static_cast<double>(values.size())};

    TDoubleVec segmentEndTimes(segmentation.size() - 1,
                               static_cast<double>(values.size()) * dt);
    TRegressionVec models(segmentation.size() - 1);

    auto predict = [&segmentEndTimes, &models](double time) {
        const double eps{100.0 * std::numeric_limits<double>::epsilon()};
        auto index = std::upper_bound(segmentEndTimes.begin(),
                                      segmentEndTimes.end(), time + eps);
        return models[index - segmentEndTimes.begin()].predict(time);
    };

    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        double a{static_cast<double>(2 * segmentation[i - 1] + 1) / 2.0 * dt};
        double b{static_cast<double>(2 * segmentation[i] + 1) / 2.0 * dt};
        segmentEndTimes[i - 1] = b;
        models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                       values.begin() + segmentation[i], a, dt);
    }
    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));
    LOG_TRACE(<< " = " << core::CContainerPrinter::print(segmentEndTimes));

    if (reweightOutliers(values.begin(), values.end(), 0.0, dt, predict,
                         outlierFraction, outlierWeight)) {
        for (std::size_t i = 1; i < segmentation.size(); ++i) {
            double a{static_cast<double>(2 * segmentation[i - 1] + 1) / 2.0 * dt};
            models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                           values.begin() + segmentation[i], a, dt);
            LOG_TRACE(<< "model = " << models[i - 1].print());
        }
    }

    return [segmentEndTimes, models](double time) {
        const double eps{100.0 * std::numeric_limits<double>::epsilon()};
        auto index = std::upper_bound(segmentEndTimes.begin(),
                                      segmentEndTimes.end(), time + eps);
        return models[index - segmentEndTimes.begin()].predict(time);
    };
}

template<typename ITR>
void CTimeSeriesSegmentation::fitTopDownPiecewiseLinearScaledPeriodic(ITR begin,
                                                                      ITR end,
                                                                      std::size_t offset,
                                                                      const TDoubleVec& model,
                                                                      double significanceToSegment,
                                                                      TSizeVec& segmentation) {
    // We want to find the partition of [begin, end) into scaled periodic
    // models which efficiently predicts the values. To this end we maximize
    // R^2 whilst maintaining a parsimonious model. We achieve the later
    // objective by doing model selection for each candidate segment using
    // the significance of the variance ratio.

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};
    std::size_t period{model.size()};
    TMeanVarAccumulator moments{centredResidualMoments(begin, end, offset, model)};

    LOG_TRACE(<< "Considering splitting [" << offset << "," << offset + range << ")");

    if (range > period + 2 && CBasicStatistics::variance(moments) > 0.0) {
        using TDoubleItrPr = std::pair<double, ITR>;
        using TMinAccumulator = CBasicStatistics::COrderStatisticsStack<TDoubleItrPr, 1>;
        using TMeanAccumulatorAry = std::array<TMeanAccumulator, 3>;

        TMinAccumulator minResidualVariance;

        auto statistics = [offset, begin, period, &model](ITR begin_, ITR end_) {
            TMeanAccumulatorAry result;
            for (ITR i = begin_; i != end_; ++i) {
                double x{CBasicStatistics::mean(*i)};
                double w{CBasicStatistics::count(*i)};
                double p{model[(offset + std::distance(begin, i)) % period]};
                result[0].add(x * x, w);
                result[1].add(x * p, w);
                result[2].add(p * p, w);
            }
            return result;
        };
        auto variance = [](const TMeanAccumulatorAry& l, const TMeanAccumulatorAry& r) {
            double nl{CBasicStatistics::count(l[0])};
            double nr{CBasicStatistics::count(r[0])};
            double vl{CBasicStatistics::mean(l[0]) -
                      CTools::pow2(CBasicStatistics::mean(l[1])) /
                          CBasicStatistics::mean(l[2])};
            double vr{CBasicStatistics::mean(r[0]) -
                      CTools::pow2(CBasicStatistics::mean(r[1])) /
                          CBasicStatistics::mean(r[2])};
            return (nl * vl + nr * vr) / (nl + nr);
        };

        ITR i = begin + 1;

        TMeanAccumulatorAry leftStatistics{statistics(begin, i)};
        TMeanAccumulatorAry rightStatistics{statistics(i, end)};
        minResidualVariance.add({variance(leftStatistics, rightStatistics), i});

        for (/**/; i + 1 != end; ++i) {
            TMeanAccumulatorAry deltaStatistics{statistics(i, i + 1)};
            for (std::size_t j = 0; j < 3; ++j) {
                leftStatistics[j] += deltaStatistics[j];
                rightStatistics[j] -= deltaStatistics[j];
            }
            minResidualVariance.add({variance(leftStatistics, rightStatistics), i + 1});
        }

        ITR split{minResidualVariance[0].second};
        std::size_t splitOffset{offset + std::distance(begin, split)};
        LOG_TRACE(<< "  candidate = " << splitOffset
                  << ", variance = " << minResidualVariance[0].first);

        TMeanVarAccumulator leftMoments{centredResidualMoments(begin, split, offset, model)};
        TMeanVarAccumulator rightMoments{
            centredResidualMoments(split, end, splitOffset, model)};
        TMeanAccumulator mean{std::accumulate(begin, end, TMeanAccumulator{})};
        LOG_TRACE(<< "  total moments = " << moments << ", left moments = " << leftMoments
                  << ", right moments = " << rightMoments);
        double f{CBasicStatistics::variance(moments) /
                 (CBasicStatistics::variance(leftMoments + rightMoments) +
                  MINIMUM_COEFFICIENT_OF_VARIATION * std::fabs(CBasicStatistics::mean(mean)))};
        double significance{CStatisticalTests::rightTailFTest(
            f, static_cast<double>(range - 1), static_cast<double>(range - 3))};
        LOG_TRACE(<< "  significance = " << significance);

        if (significance < 0.01) {
            fitTopDownPiecewiseLinearScaledPeriodic(
                begin, split, offset, model, significanceToSegment, segmentation);
            fitTopDownPiecewiseLinearScaledPeriodic(
                split, end, splitOffset, model, significanceToSegment, segmentation);
            segmentation.push_back(splitOffset);
        }
    }
}

void CTimeSeriesSegmentation::fitPiecewiseLinearScaledPeriodic(
    const TFloatMeanAccumulatorVec& values,
    std::size_t period,
    const TSizeVec& segmentation,
    double outlierFraction,
    double outlierWeight,
    TFloatMeanAccumulatorVec& reweighted,
    TDoubleVec& model,
    TDoubleVec& scales) {
    scales.assign(segmentation.size() - 1, 1.0);

    auto computeScale = [&](std::size_t begin, std::size_t end) {
        TMeanAccumulator projection;
        TMeanAccumulator Z;
        for (std::size_t i = begin; i < end; ++i) {
            double x{CBasicStatistics::mean(reweighted[i])};
            double w{CBasicStatistics::count(reweighted[i])};
            double p{model[i % period]};
            if (w > 0.0) {
                projection.add(x * p, w);
                Z.add(p * p, w);
            }
        }
        return CBasicStatistics::mean(Z) == 0.0
                   ? 0.0
                   : CBasicStatistics::mean(projection) / CBasicStatistics::mean(Z);
    };
    auto scale = [&](std::size_t i) {
        auto right = std::upper_bound(segmentation.begin(), segmentation.end(), i);
        return scales[(right - segmentation.begin()) - 1];
    };
    auto predict = [&](std::size_t i) {
        auto right = std::upper_bound(segmentation.begin(), segmentation.end(), i);
        return scales[(right - segmentation.begin()) - 1] * model[i % period];
    };
    auto fitScaledPeriodicModel = [&]() {
        for (std::size_t pass = 0; pass < 2; ++pass) {
            model = fitPeriodicModel(reweighted.begin(), reweighted.end(), period, scale);
            for (std::size_t i = 1; i < segmentation.size(); ++i) {
                scales[i - 1] = computeScale(segmentation[i - 1], segmentation[i]);
            }
        }
    };

    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    // First pass to re-weight any large outliers.
    model = fitPeriodicModel(values.begin(), values.end(), period, unit);
    reweighted.assign(values.begin(), values.end());
    reweightOutliers(reweighted.begin(), reweighted.end(), predict,
                     outlierFraction, outlierWeight);

    // Fit the model adjusting for scales.
    fitScaledPeriodicModel();
    LOG_TRACE(<< "model = " << core::CContainerPrinter::print(model));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));

    // Re-weight outliers based on new model.
    reweighted.assign(values.begin(), values.end());
    if (reweightOutliers(reweighted.begin(), reweighted.end(), predict,
                         outlierFraction, outlierWeight)) {
        // If any re-weighting happened fine tune the model fit.
        fitScaledPeriodicModel();
        LOG_TRACE(<< "model = " << core::CContainerPrinter::print(model));
        LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));
    }
}
}
}
