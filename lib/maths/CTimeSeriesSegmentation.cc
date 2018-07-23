/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesSegmentation.h>

#include <core/CTriple.h>

#include <maths/CRegression.h>
#include <maths/CRegressionDetail.h>
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
TDoubleVec fitPeriodicModel(ITR begin,
                            ITR end,
                            std::size_t period,
                            const std::function<double(double)>& scale = [](std::size_t) { return 1.0; }) {
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    TMeanAccumulatorVec levels(period);
    for (std::size_t i = 0; begin != end; ++i, ++begin) {
        levels[i % period].add(CBasicStatistics::mean(*begin) / scale(i),
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
bool reweightOutliers(ITR begin,
                      ITR end,
                      double startTime,
                      double dt,
                      const std::function<double(double)>& predict,
                      double fraction,
                      double weight) {
    using TDoublePtrPr = std::pair<double, typename std::iterator_traits<ITR>::value_type*>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<TDoublePtrPr, std::greater<TDoublePtrPr>>;

    if (fraction > 0.0) {
        double n{static_cast<double>(std::distance(begin, end))};
        LOG_TRACE(<< "  # outliers = " << static_cast<std::size_t>(fraction * n));
        TMaxAccumulator outliers{static_cast<std::size_t>(fraction * n)};
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
bool reweightOutliers(ITR begin,
                      ITR end,
                      const std::function<double(std::size_t)>& predict,
                      double fraction,
                      double weight) {
    using TDoublePtrPr = std::pair<double, typename std::iterator_traits<ITR>::value_type*>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsHeap<TDoublePtrPr, std::greater<TDoublePtrPr>>;

    if (fraction > 0.0) {
        double n{static_cast<double>(std::distance(begin, end))};
        TMaxAccumulator outliers{static_cast<std::size_t>(fraction * n)};
        for (std::size_t i = 0; begin != end; ++i, ++begin) {
            double diff{CBasicStatistics::mean(*begin) - predict(i)};
            outliers.add({std::fabs(diff), &(*begin)});
        }

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
        if (CBasicStatistics::count(*begin) > 0.0) {
            moments.add(CBasicStatistics::mean(*begin) - CRegression::predict(params, time),
                        CBasicStatistics::count(*begin));
        }
    }
    return moments;
}

//! Compute the BIC of the linear model fit to [\p begin, \p end).
template<typename ITR>
TMeanVarAccumulator centredResidualMoments(ITR begin, ITR end, double startTime, double dt) {
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
    CBasicStatistics::moment<0>(moments) = 0.0;
    LOG_TRACE("  moments = " << moments);

    return moments;
}
}

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::topDownPiecewiseLinear(const TFloatMeanAccumulatorVec& values,
                                                double significanceToSegment,
                                                double outlierFraction,
                                                double outlierWeight) {
    TSizeVec segmentation{0, values.size()};
    double dt{10.0 / static_cast<double>(values.size())};
    topDownPiecewiseLinear(values.cbegin(), values.cend(), 0, 0.0, dt, significanceToSegment,
                           outlierFraction, outlierWeight, segmentation);
    std::sort(segmentation.begin(), segmentation.end());
    return segmentation;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePredictionsOfPiecewiseLinear(TFloatMeanAccumulatorVec values,
                                                            const TSizeVec& segmentation,
                                                            double outlierFraction,
                                                            double outlierWeight) {
    using TRegressionVec = std::vector<TRegression>;

    double dt{10.0 / static_cast<double>(values.size())};

    TDoubleVec times(segmentation.size() - 1, static_cast<double>(values.size() * dt));
    TRegressionVec models(segmentation.size() - 1);

    auto predict = [&times, &models](double time) {
        const double eps{100.0 * std::numeric_limits<double>::epsilon()};
        auto right = std::upper_bound(times.begin(), times.end(), time + eps);
        return models[right - times.begin()].predict(time);
    };

    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        double a{static_cast<double>(2 * segmentation[i - 1] + 1) / 2.0 * dt};
        double b{static_cast<double>(2 * segmentation[i] + 1) / 2.0 * dt};
        times[i - 1] = b;
        models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                       values.begin() + segmentation[i], a, dt);
    }
    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));
    LOG_TRACE(<< "times = " << core::CContainerPrinter::print(times));

    if (reweightOutliers(values.begin(), values.end(), 0.0, dt, predict,
                         outlierFraction, outlierWeight)) {
        for (std::size_t i = 1; i < segmentation.size(); ++i) {
            double a{static_cast<double>(2 * segmentation[i - 1] + 1) / 2.0 * dt};
            models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                           values.begin() + segmentation[i], a, dt);
            LOG_TRACE(<< "model = " << models[i - 1].print());
        }
    }

    double time{dt / 2.0};
    for (auto i = values.begin(); i != values.end(); ++i, time += dt) {
        if (CBasicStatistics::count(*i) > 0.0) {
            CBasicStatistics::moment<0>(*i) -= predict(time);
        }
    }
    return values;
}

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::topDownPeriodicPiecewiseLinearScaled(const TFloatMeanAccumulatorVec& values,
                                                              std::size_t period,
                                                              double significanceToSegment,
                                                              double outlierFraction,
                                                              double outlierWeight) {
    TFloatMeanAccumulatorVec values_(values);
    TDoubleVec model(fitPeriodicModel(values.begin(), values.end(), period));
    reweightOutliers(values_.begin(), values_.end(),
                     [&model](std::size_t i) { return model[i % model.size()]; },
                     outlierFraction, outlierWeight);
    model = fitPeriodicModel(values_.begin(), values_.end(), period);

    TSizeVec segmentation{0, values.size()};
    topDownPeriodicPiecewiseLinearScaled(values_.cbegin(), values_.cend(), 0, model,
                                         significanceToSegment, segmentation);
    std::sort(segmentation.begin(), segmentation.end());

    return segmentation;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePredictionsOfPiecewiseLinearScaled(const TFloatMeanAccumulatorVec& values,
                                                                  std::size_t period,
                                                                  const TSizeVec& segmentation,
                                                                  double outlierFraction,
                                                                  double outlierWeight) {
    TFloatMeanAccumulatorVec reweighted;
    TDoubleVec model;
    TDoubleVec scales(segmentation.size() - 1, 1.0);

    auto computeScale = [&](std::size_t begin, std::size_t end) {
        TMeanAccumulator projection;
        TMeanAccumulator Z;
        for (std::size_t i = begin; i < end; ++i) {
            double x{CBasicStatistics::mean(reweighted[i])};
            double w{CBasicStatistics::count(reweighted[i])};
            double p{model[i % period]};
            projection.add(x * p, w);
            Z.add(p * p, w);
        }
        return CBasicStatistics::mean(projection) / CBasicStatistics::mean(Z);
    };
    auto scale = [&](std::size_t i) {
        auto right = std::upper_bound(segmentation.begin(), segmentation.end(), i);
        return scales[right - segmentation.begin() - 1];
    };
    auto predict = [&](std::size_t i) {
        return scale(i) * model[i % period];
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
    model = fitPeriodicModel(values.begin(), values.end(), period);
    reweighted.assign(values.begin(), values.end());
    reweightOutliers(reweighted.begin(), reweighted.end(), predict, outlierFraction, outlierWeight);

    // Fit the model adjusting for scales.
    fitScaledPeriodicModel();
    LOG_TRACE(<< "model = " << core::CContainerPrinter::print(model));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));

    // Re-weight outliers based on new model.
    reweighted.assign(values.begin(), values.end());
    if (reweightOutliers(reweighted.begin(), reweighted.end(), predict, outlierFraction, outlierWeight)) {
        // If any re-weighting happened fine tune the model fit.
        fitScaledPeriodicModel();
        LOG_TRACE(<< "model = " << core::CContainerPrinter::print(model));
        LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));
    }

    for (std::size_t i = 0; i != reweighted.size(); ++i) {
        if (CBasicStatistics::count(reweighted[i]) > 0.0) {
            CBasicStatistics::moment<0>(reweighted[i]) -= predict(i);
        }
    }

    return reweighted;
}

template<typename ITR>
void CTimeSeriesSegmentation::topDownPiecewiseLinear(ITR begin,
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
            ITR i = values.cbegin() + 3;
            double splitTime{startTime +
                             static_cast<double>(std::distance(values.cbegin(), i)) * dt};
            minResidualVariance = TMinAccumulator{};

            TRegression leftModel{fitLinearModel(values.cbegin(), i, startTime, dt)};
            TRegression rightModel{fitLinearModel(i, values.cend(), splitTime, dt)};
            TMeanVarAccumulator leftResidualMoments{
                residualMoments(values.cbegin(), i, startTime, dt, leftModel)};
            TMeanVarAccumulator rightResidualMoments{
                residualMoments(i, values.cend(), splitTime, dt, rightModel)};
            minResidualVariance.add(
                {variance(leftResidualMoments, rightResidualMoments), splitTime, i});

            for (/**/; i + 3 != values.cend(); ++i, splitTime += dt) {
                TRegression deltaModel{fitLinearModel(i, i + 1, splitTime, dt)};
                leftModel += deltaModel;
                rightModel -= deltaModel;
                leftResidualMoments = residualMoments(values.cbegin(), i + 1,
                                                      startTime, dt, leftModel);
                rightResidualMoments = residualMoments(i + 1, values.cend(),
                                                       splitTime, dt, rightModel);
                minResidualVariance.add({variance(leftResidualMoments, rightResidualMoments),
                                         splitTime + dt, i + 1});
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
        LOG_TRACE(<< "  total moments = " << moments
                  << ", left moments = " << leftMoments
                  << ", right moments = " << rightMoments);
        double f{CBasicStatistics::variance(moments) /
                 CBasicStatistics::variance(leftMoments + rightMoments)};
        double significance{CStatisticalTests::rightTailFTest(f, range - 3, range - 5)};
        LOG_TRACE(<< "  significance = " << significance);

        if (significance < significanceToSegment) {
            topDownPiecewiseLinear(begin, begin + splitOffset, offset,
                                   startTime, dt, significanceToSegment,
                                   outlierFraction, outlierWeight, segmentation);
            topDownPiecewiseLinear(begin + splitOffset, end, offset + splitOffset,
                                   splitTime, dt, significanceToSegment,
                                   outlierFraction, outlierWeight, segmentation);
            segmentation.push_back(offset + splitOffset);
        }
    }
}

template<typename ITR>
void CTimeSeriesSegmentation::topDownPeriodicPiecewiseLinearScaled(ITR begin,
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
        LOG_TRACE(<< "  total moments = " << moments
                  << ", left moments = " << leftMoments
                  << ", right moments = " << rightMoments);
        double f{CBasicStatistics::variance(moments) /
                 CBasicStatistics::variance(leftMoments + rightMoments)};
        double significance{CStatisticalTests::rightTailFTest(f, range - 1, range - 3)};
        LOG_TRACE(<< "  significance = " << significance);

        if (significance < 0.01) {
            topDownPeriodicPiecewiseLinearScaled(begin, split, offset, model,
                                                 significanceToSegment, segmentation);
            topDownPeriodicPiecewiseLinearScaled(split, end, splitOffset, model,
                                                 significanceToSegment, segmentation);
            segmentation.push_back(splitOffset);
        }
    }
}
}
}
