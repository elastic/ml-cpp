/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesSegmentation.h>

#include <core/CTriple.h>

#include <maths/CRegression.h>
#include <maths/CRegressionDetail.h>
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
using TRegression = CRegression::CLeastSquaresOnline<1, double>;

//! Fit a linear model to the values in [\p begin, \p end).
template<typename ITR>
TRegression fitLinearModel(ITR begin, ITR end, double startTime, double dt) {
    TRegression result;
    for (double time = startTime + dt / 2.0; begin != end; ++begin) {
        result.add(time, CBasicStatistics::mean(*begin), CBasicStatistics::count(*begin));
        time += dt;
    }
    return result;
}

//! Fit a periodic model of period \p period to the values [\p begin, \p end).
template<typename ITR>
TDoubleVec fitPeriodicModel(ITR begin, ITR end, std::size_t period) {
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    TMeanAccumulatorVec levels(period);
    for (std::size_t i = 0; begin != end; ++i, ++begin) {
        levels[i % period].add(CBasicStatistics::mean(*begin),
                               CBasicStatistics::count(*begin));
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
void reweightOutliers(ITR begin,
                      ITR end,
                      double startTime,
                      double dt,
                      const std::function<double(double)>& predict,
                      double fraction,
                      double weight) {
    using TDoublePtrPr = std::pair<double, typename std::iterator_traits<ITR>::value_type*>;
    using TMaxAccumulator = typename CBasicStatistics::SMax<TDoublePtrPr>::TAccumulator;

    double n{static_cast<double>(std::distance(begin, end))};
    TMaxAccumulator outliers{static_cast<std::size_t>(fraction * n)};
    for (double time = startTime + dt / 2.0; begin != end; time += dt, ++begin) {
        double diff{CBasicStatistics::mean(*begin) - predict(time)};
        outliers.add({std::fabs(diff), &(*begin)});
    }

    for (auto outlier : outliers) {
        CBasicStatistics::count(*outlier.second) *= weight;
    }
}

//! Re-weight the outlying values in [\p begin, \p end) w.r.t. the
//! predictions from \p predict.
template<typename ITR>
void reweightOutliers(ITR begin,
                      ITR end,
                      const std::function<double(std::size_t)>& predict,
                      double fraction,
                      double weight) {
    using TDoublePtrPr = std::pair<double, typename std::iterator_traits<ITR>::value_type*>;
    using TMaxAccumulator = typename CBasicStatistics::SMax<TDoublePtrPr>::TAccumulator;

    double n{static_cast<double>(std::distance(begin, end))};
    TMaxAccumulator outliers{static_cast<std::size_t>(fraction * n)};
    for (std::size_t i = 0; begin != end; ++i, ++begin) {
        double diff{CBasicStatistics::mean(*begin) - predict(i)};
        outliers.add({std::fabs(diff), &(*begin)});
    }

    for (auto outlier : outliers) {
        CBasicStatistics::count(*outlier.second) *= weight;
    }
}

//! Compute the moments of the values in [\p begin, \p end) after subtracting
//! the predictions of \p model.
template<typename ITR>
TMeanVarAccumulator
residualMoments(ITR begin, ITR end, double startTime, double dt, const TRegression& model) {
    TMeanVarAccumulator moments;
    for (double time = startTime + dt / 2.0; begin != end; ++begin, time += dt) {
        if (CBasicStatistics::count(*begin) > 0.0) {
            moments.add(CBasicStatistics::mean(*begin) - model.predict(time),
                        CBasicStatistics::count(*begin));
        }
    }
    return moments;
}

//! Compute the BIC of the linear model fit to [\p begin, \p end).
template<typename ITR>
double logLikelihoodLinearModel(ITR begin,
                                ITR end,
                                double startTime,
                                double dt,
                                double outlierFraction,
                                double outlierWeight) {
    using T = typename std::iterator_traits<ITR>::value_type;
    using TVec = std::vector<T>;

    if (outlierFraction > 0.0 && outlierFraction < 1.0) {
        double oldCount{std::accumulate(begin, end, 0.0, [](double count, const T& value) {
            return count + CBasicStatistics::count(value);
        })};

        TRegression model{fitLinearModel(begin, end, startTime, dt)};

        TVec values{begin, end};
        reweightOutliers(values.begin(), values.end(), startTime, dt,
                         [&model](double time) { return model.predict(time); },
                         outlierFraction, outlierWeight);
        model = fitLinearModel(values.begin(), values.end(), startTime, dt);

        TMeanVarAccumulator moments{
            residualMoments(values.begin(), values.end(), startTime, dt, model)};
        double newCount{std::accumulate(begin, end, 0.0, [](double count, const T& value) {
            return count + CBasicStatistics::count(value);
        })};
        double n{static_cast<double>(values.size()) * newCount / oldCount};
        return n * CTools::fastLog(CBasicStatistics::variance(moments));
    }

    if (outlierFraction < 0.0 || outlierFraction >= 1.0) {
        LOG_ERROR(<< "Ignoring bad value for outlier fraction "
                  << outlierFraction << ". This must be in the range [0,1).");
    }

    TRegression model{fitLinearModel(begin, end, startTime, dt)};
    TMeanVarAccumulator moments{residualMoments(begin, end, startTime, dt, model)};
    double n{static_cast<double>(std::distance(begin, end))};
    return n * CTools::fastLog(CBasicStatistics::maximumLikelihoodVariance(moments));
}

//! Compute the BIC of the scaled periodic model for the values in
//! [\p begin, \p end).
template<typename ITR>
double logLikelihoodScaledPeriodicModel(ITR begin,
                                        ITR end,
                                        std::size_t offset,
                                        const TDoubleVec& model) {
    std::size_t period{model.size()};

    TMeanAccumulator projection;
    TMeanAccumulator norm2;
    for (ITR i = begin; i != end; ++i) {
        double x{CBasicStatistics::mean(*i)};
        double w{CBasicStatistics::count(*i)};
        double p{model[(offset + std::distance(begin, i)) % period]};
        projection.add(x * p, w);
        norm2.add(p * p, w);
    }
    double scale{CBasicStatistics::mean(projection) / CBasicStatistics::mean(norm2)};
    LOG_TRACE(<< "  scale = " << scale);

    TMeanVarAccumulator moments;
    for (ITR i = begin; i != end; ++i) {
        double x{CBasicStatistics::mean(*i)};
        double w{CBasicStatistics::count(*i)};
        double p{scale * model[(offset + std::distance(begin, i)) % period]};
        moments.add(x - p, w);
    }
    LOG_TRACE("  moments = " << moments);

    double n{static_cast<double>(std::distance(begin, end))};
    return n * CTools::fastLog(CBasicStatistics::maximumLikelihoodVariance(moments));
}

//! Remove the predictions of the piecewise linear model with breakpoints
//! defined by \p segmentation fit to [\p begin, \p end).
void removePiecewiseLinearModelPredictions(TFloatMeanAccumulatorVec& values,
                                           const TSizeVec& segmentation,
                                           double outlierFraction,
                                           double outlierWeight) {
    using TRegressionVec = std::vector<TRegression>;

    double dt{10.0 / static_cast<double>(values.size())};

    TDoubleVec times(segmentation.size(), static_cast<double>(values.size() * dt));
    TRegressionVec models(segmentation.size() - 1);

    auto predict = [&times, &models](double time) {
        std::ptrdiff_t index{std::upper_bound(times.begin(), times.end(), time) -
                             times.begin() - 1};
        return models[index].predict(time);
    };

    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        times[i - 1] = static_cast<double>(segmentation[i]) * dt;
        models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                       values.begin() + segmentation[i],
                                       times[i - 1], dt);
    }

    reweightOutliers(values.begin(), values.end(), 0.0, dt, predict,
                     outlierFraction, outlierWeight);

    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                       values.begin() + segmentation[i],
                                       times[i - 1], dt);
    }

    double time{dt / 2.0};
    for (auto i = values.begin(); i != values.end(); ++i, time += dt) {
        if (CBasicStatistics::count(*i) > 0.0) {
            CBasicStatistics::moment<0>(*i) -= predict(time);
        }
    }
}

//! Remove the predictions of the piecewise scaled periodic model with
//! breakpoints defined by \p segmentation fit to [\p begin, \p end).
void removePiecewiseLinearScalingModelPredictions(TFloatMeanAccumulatorVec& values,
                                                  std::size_t period,
                                                  const TSizeVec& segmentation,
                                                  double outlierFraction,
                                                  double outlierWeight) {
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;

    TDoubleVec model(fitPeriodicModel(values.begin(), values.end(), period));
    TDoubleVec scales(segmentation.size() - 1);

    auto computeScale = [&values, &model, period](std::size_t begin, std::size_t end) {
        TMeanAccumulator projection;
        TMeanAccumulator Z;
        for (std::size_t i = begin; i < end; ++i) {
            double x{CBasicStatistics::mean(values[i])};
            double w{CBasicStatistics::count(values[i])};
            double p{model[i % period]};
            projection.add(x * p, w);
            Z.add(x * p, w);
        }
        return CBasicStatistics::mean(projection) / std::sqrt(CBasicStatistics::mean(Z));
    };
    auto scale = [&segmentation, &scales](std::size_t i) {
        std::ptrdiff_t index{std::upper_bound(segmentation.begin(), segmentation.end(), i) -
                             segmentation.begin() - 1};
        return scales[index];
    };
    auto predict = [&scale, &model, period](std::size_t i) {
        return scale(i) * model[i % period];
    };

    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        scales[i - 1] = computeScale(segmentation[i - 1], segmentation[i]);
    }

    reweightOutliers(values.begin(), values.end(), predict, outlierFraction, outlierWeight);
    TMeanAccumulatorVec levels(period);

    for (std::size_t i = 0; i < values.size(); ++i) {
        levels[i % period].add(CBasicStatistics::mean(values[i]) / scale(i),
                               CBasicStatistics::count(values[i]));
    }
    for (std::size_t i = 0; i < period; ++i) {
        model[i] = CBasicStatistics::mean(levels[i]);
    }

    for (std::size_t i = 0; i != values.size(); ++i) {
        if (CBasicStatistics::count(values[i]) > 0.0) {
            CBasicStatistics::moment<0>(values[i]) -= predict(i);
        }
    }
}
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::topDownPiecewiseLinear(const TFloatMeanAccumulatorVec& values,
                                                double bicGainToSegment,
                                                double outlierFraction,
                                                double outlierWeight) {
    double dt{10.0 / static_cast<double>(values.size())};

    TSizeVec segmentation{0, values.size()};
    topDownPiecewiseLinear(values.begin(), values.end(), 0, 0.0, dt, bicGainToSegment,
                           outlierFraction, outlierWeight, segmentation);
    std::sort(segmentation.begin(), segmentation.end());

    TFloatMeanAccumulatorVec values_(values);
    removePiecewiseLinearModelPredictions(values_, segmentation, outlierFraction, outlierWeight);

    return values_;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::topDownPeriodicPiecewiseLinearScaling(const TFloatMeanAccumulatorVec& values,
                                                               std::size_t period,
                                                               double bicGainToSegment,
                                                               double outlierFraction,
                                                               double outlierWeight) {
    TDoubleVec model(fitPeriodicModel(values.begin(), values.end(), period));

    TFloatMeanAccumulatorVec values_(values);
    reweightOutliers(values_.begin(), values_.end(),
                     [&model](std::size_t i) { return model[i % model.size()]; },
                     outlierFraction, outlierWeight);
    model = fitPeriodicModel(values_.begin(), values_.end(), period);

    LOG_DEBUG(<< core::CContainerPrinter::print(model));

    TSizeVec segmentation{0, values.size()};
    topDownPeriodicPiecewiseLinearScaling(values_.begin(), values_.end(), 0,
                                          model, bicGainToSegment, segmentation);
    std::sort(segmentation.begin(), segmentation.end());

    values_ = values;
    removePiecewiseLinearScalingModelPredictions(values_, period, segmentation,
                                                 outlierFraction, outlierWeight);

    return values_;
}

template<typename ITR>
void CTimeSeriesSegmentation::topDownPiecewiseLinear(ITR begin,
                                                     ITR end,
                                                     std::size_t offset,
                                                     double startTime,
                                                     double dt,
                                                     double bicGainToSegment,
                                                     double outlierFraction,
                                                     double outlierWeight,
                                                     TSizeVec& segmentation) {

    // We want to find the partition of [begin, end) into linear models which
    // efficiently predicts the values. To this end we maximize R^2 whilst
    // maintaining a parsimonious model. We achieve the later objective by
    // doing model selection for each candidate segment using delta BIC.

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};

    if (range >= 6) {
        using TDoubleDoubleItrTr = core::CTriple<double, double, ITR>;
        using TMinAccumulator =
            CBasicStatistics::COrderStatisticsStack<TDoubleDoubleItrTr, 1>;

        auto variance = [](const TMeanVarAccumulator& l, const TMeanVarAccumulator& r) {
            double nl{CBasicStatistics::count(l)};
            double nr{CBasicStatistics::count(r)};
            return (nl * CBasicStatistics::maximumLikelihoodVariance(l) +
                    nr * CBasicStatistics::maximumLikelihoodVariance(r)) /
                   (nl + nr);
        };

        ITR i = begin + 3;
        double splitTime{startTime + static_cast<double>(std::distance(begin, i)) * dt};

        TMinAccumulator minResidualVariance;

        TRegression leftModel{fitLinearModel(begin, i, startTime, dt)};
        TRegression rightModel{fitLinearModel(i, end, splitTime, dt)};
        TMeanVarAccumulator leftResidualMoments{
            residualMoments(begin, i, startTime, dt, leftModel)};
        TMeanVarAccumulator rightResidualMoments{
            residualMoments(i, end, splitTime, dt, rightModel)};
        minResidualVariance.add(
            {variance(leftResidualMoments, rightResidualMoments), splitTime, i});

        // We iterate through every possible binary split of the data into
        // contiguous ranges looking for the split which minimizes the total
        // variance of the values minus linear model predictions.

        for (/**/; i + 3 != end; ++i, splitTime += dt) {
            TRegression deltaModel{fitLinearModel(i, i + 1, splitTime, dt)};
            leftModel += deltaModel;
            rightModel -= deltaModel;
            TMeanVarAccumulator deltaLeftResidualMoments{
                residualMoments(i, i + 1, splitTime, dt, leftModel)};
            TMeanVarAccumulator deltaRightResidualMoments{
                residualMoments(i, i + 1, splitTime, dt, rightModel)};
            leftResidualMoments += deltaLeftResidualMoments;
            rightResidualMoments -= deltaRightResidualMoments;
            minResidualVariance.add({variance(leftResidualMoments, rightResidualMoments),
                                     splitTime, i + 1});
        }

        splitTime = minResidualVariance[0].second;
        ITR split{minResidualVariance[0].third};
        double bicGain{
            logLikelihoodLinearModel(begin, end, startTime, dt, outlierFraction, outlierWeight) -
            logLikelihoodLinearModel(begin, split, startTime, dt, outlierFraction, outlierWeight) -
            logLikelihoodLinearModel(split, end, startTime, dt, outlierFraction, outlierWeight) +
            2.0 * CTools::fastLog(static_cast<double>(range))};

        if (bicGain > bicGainToSegment) {
            std::size_t splitOffset{offset + std::distance(begin, split)};
            topDownPiecewiseLinear(begin, split, offset, startTime, dt, bicGainToSegment,
                                   outlierFraction, outlierWeight, segmentation);
            topDownPiecewiseLinear(split, end, splitOffset, splitTime, dt, bicGainToSegment,
                                   outlierFraction, outlierWeight, segmentation);
            segmentation.push_back(splitOffset);
        }
    }
}

template<typename ITR>
void CTimeSeriesSegmentation::topDownPeriodicPiecewiseLinearScaling(ITR begin,
                                                                    ITR end,
                                                                    std::size_t offset,
                                                                    const TDoubleVec& model,
                                                                    double bicGainToSegment,
                                                                    TSizeVec& segmentation) {
    // We want to find the partition of

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};
    std::size_t period{model.size()};

    LOG_DEBUG(<< "Considering splitting [" << offset << "," << offset + range << ")");

    if (range >= period) {
        using TDoubleItrPr = std::pair<double, ITR>;
        using TMinAccumulator = CBasicStatistics::COrderStatisticsStack<TDoubleItrPr, 1>;
        using TMeanAccumulatorAry = boost::array<TMeanAccumulator, 3>;

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
                      CTools::pow2(CBasicStatistics::mean(l[1])) / CBasicStatistics::mean(l[2])};
            double vr{CBasicStatistics::mean(r[0]) -
                      CTools::pow2(CBasicStatistics::mean(r[1])) / CBasicStatistics::mean(r[2])};
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
        LOG_DEBUG(<< "  candidate = " << splitOffset << ", variance = " << minResidualVariance[0].first);

        double bicGain{logLikelihoodScaledPeriodicModel(begin, end, offset, model) -
                       logLikelihoodScaledPeriodicModel(begin, split, offset, model) -
                       logLikelihoodScaledPeriodicModel(split, end, splitOffset, model) +
                       2.0 * CTools::fastLog(static_cast<double>(range))};
        LOG_DEBUG(<< "  delta BIC = " << bicGain);

        if (bicGain > bicGainToSegment) {
            topDownPeriodicPiecewiseLinearScaling(begin, split, offset, model,
                                                  bicGainToSegment, segmentation);
            topDownPeriodicPiecewiseLinearScaling(split, end, splitOffset, model,
                                                  bicGainToSegment, segmentation);
            segmentation.push_back(splitOffset);
        }
    }
}
}
}
