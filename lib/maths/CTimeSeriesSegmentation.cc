/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CTimeSeriesSegmentation.h>

#include <core/CContainerPrinter.h>
#include <core/CTriple.h>
#include <core/CVectorRange.h>

#include <maths/CBasicStatistics.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CLeastSquaresOnlineRegressionDetail.h>
#include <maths/COrderings.h>
#include <maths/CSignal.h>
#include <maths/CStatisticalTests.h>
#include <maths/CTools.h>

#include <algorithm>
#include <iterator>
#include <numeric>

namespace ml {
namespace maths {

CTimeSeriesSegmentation::TSizeVec
CTimeSeriesSegmentation::piecewiseLinear(const TFloatMeanAccumulatorVec& values,
                                         double pValueToSegment,
                                         double outlierFraction,
                                         std::size_t maxSegments) {

    std::size_t maxDepth{static_cast<std::size_t>(
        std::ceil(std::log2(static_cast<double>(maxSegments))))};
    LOG_TRACE(<< "max depth = " << maxDepth);

    TFloatMeanAccumulatorVec reweighted;
    TSizeVec segmentation{0, values.size()};
    TDoubleDoublePrVec depthAndPValue;
    fitTopDownPiecewiseLinear(values.cbegin(), values.cend(), 0, // depth
                              0, // offset of first value in range
                              maxDepth, pValueToSegment, outlierFraction,
                              segmentation, depthAndPValue, reweighted);

    selectSegmentation(maxSegments, segmentation, depthAndPValue);

    return segmentation;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinear(TFloatMeanAccumulatorVec values,
                                               const TSizeVec& segmentation,
                                               double outlierFraction) {
    TFloatMeanAccumulatorVec reweighted{values};
    auto predict = fitPiecewiseLinear(segmentation, outlierFraction, reweighted);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (CBasicStatistics::count(values[i]) > 0.0) {
            CBasicStatistics::moment<0>(values[i]) -= predict(static_cast<double>(i));
        }
    }
    return values;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::removePiecewiseLinearDiscontinuities(TFloatMeanAccumulatorVec values,
                                                              const TSizeVec& segmentation,
                                                              double outlierFraction) {
    TFloatMeanAccumulatorVec reweighted(values);
    auto predict = fitPiecewiseLinear(segmentation, outlierFraction, reweighted);
    TDoubleVec steps;
    steps.reserve(segmentation.size() - 1);
    for (std::size_t i = 1; i + 1 < segmentation.size(); ++i) {
        double j{static_cast<double>(segmentation[i])};
        steps.push_back(predict(j - 0.05) - predict(j + 0.05));
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
CTimeSeriesSegmentation::piecewiseLinearScaledSeasonal(const TFloatMeanAccumulatorVec& values,
                                                       const TSeasonality& seasonality,
                                                       double pValueToSegment,
                                                       std::size_t maxSegments) {

    std::size_t maxDepth{static_cast<std::size_t>(
        std::ceil(std::log2(static_cast<double>(maxSegments))))};
    LOG_TRACE(<< "max depth = " << maxDepth);

    TSizeVec segmentation{0, values.size()};
    TDoubleDoublePrVec depthAndPValue;
    fitTopDownPiecewiseLinearScaledSeasonal(values.cbegin(), values.cend(),
                                            0, // depth
                                            0, // offset of first value in range
                                            seasonality, maxDepth, pValueToSegment,
                                            segmentation, depthAndPValue);

    selectSegmentation(maxSegments, segmentation, depthAndPValue);

    return segmentation;
}

CTimeSeriesSegmentation::TFloatMeanAccumulatorVec
CTimeSeriesSegmentation::meanScalePiecewiseLinearScaledSeasonal(
    const TFloatMeanAccumulatorVec& values,
    const TSeasonalComponentVec& periods,
    const TSizeVec& segmentation,
    const TIndexWeight& indexWeight,
    double outlierFraction,
    TDoubleVecVec& models,
    TDoubleVec& scales) {

    using TMinMaxAccumulator = CBasicStatistics::CMinMax<double>;

    TFloatMeanAccumulatorVec scaledValues;
    fitPiecewiseLinearScaledSeasonal(values, periods, segmentation, outlierFraction,
                                     scaledValues, models, scales);
    LOG_TRACE(<< "models = " << core::CContainerPrinter::print(models));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));

    double scale{meanScale(segmentation, scales, indexWeight)};
    LOG_TRACE(<< "mean scale = " << scale);

    scaledValues = values;
    for (std::size_t i = 0; i != values.size(); ++i) {
        for (std::size_t j = 0; j < periods.size(); ++j) {
            if (periods[j].contains(i) && CBasicStatistics::count(values[i]) > 0.0) {
                CBasicStatistics::moment<0>(scaledValues[i]) -=
                    scaleAt(i, segmentation, scales) * models[j][periods[j].offset(i)];
            }
        }
    }

    double noiseStandardDeviation{std::sqrt([&] {
        TMeanVarAccumulator moments;
        for (std::size_t i = 0; i < scaledValues.size(); ++i) {
            if (std::any_of(periods.begin(), periods.end(), [&](const auto& period) {
                    return period.contains(i);
                })) {
                moments.add(CBasicStatistics::mean(scaledValues[i]),
                            CBasicStatistics::count(scaledValues[i]));
            }
        }
        return CBasicStatistics::variance(moments);
    }())};

    double amplitude{[&] {
        TMinMaxAccumulator minmax;
        for (std::size_t i = 0; i < scaledValues.size(); ++i) {
            double prediction{0.0};
            for (std::size_t j = 0; j < periods.size(); ++j) {
                if (periods[j].contains(i)) {
                    prediction += models[j][periods[j].offset(i)];
                }
            }
            minmax.add(prediction);
        }
        return minmax.range();
    }()};

    LOG_TRACE(<< "amplitude = " << amplitude << ", sd noise = " << noiseStandardDeviation);

    if (*std::min_element(scales.begin(), scales.end()) < 0.0 ||
        2.0 * *std::max_element(scales.begin(), scales.end()) * amplitude <= noiseStandardDeviation) {
        return values;
    }

    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::any_of(periods.begin(), periods.end(),
                        [&](const auto& period) { return period.contains(i); })) {
            // If the component is "scaled away" we treat it as missing.
            double weight{std::min(2.0 * scaleAt(i, segmentation, scales) * amplitude - noiseStandardDeviation,
                                   1.0)};
            double prediction{0.0};
            for (std::size_t j = 0; j < periods.size(); ++j) {
                if (periods[j].contains(i)) {
                    prediction += models[j][periods[j].offset(i)];
                }
            }

            if (weight <= 0.0) {
                scaledValues[i] = TFloatMeanAccumulator{};
            } else {
                CBasicStatistics::count(scaledValues[i]) *= weight;
                CBasicStatistics::moment<0>(scaledValues[i]) += scale * prediction;
            }
        }
    }

    return scaledValues;
}

double CTimeSeriesSegmentation::meanScale(const TSizeVec& segmentation,
                                          const TDoubleVec& scales,
                                          const TIndexWeight& indexWeight) {
    TMeanAccumulator result;
    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        for (std::size_t j = segmentation[i - 1]; j < segmentation[i]; ++j) {
            result.add(scales[i - 1], indexWeight(j));
        }
    }
    return CBasicStatistics::mean(result);
}

double CTimeSeriesSegmentation::scaleAt(std::size_t index,
                                        const TSizeVec& segmentation,
                                        const TDoubleVec& scales) {
    return scales[std::upper_bound(segmentation.begin(), segmentation.end(), index) -
                  segmentation.begin() - 1];
}

void CTimeSeriesSegmentation::selectSegmentation(std::size_t maxSegments,
                                                 TSizeVec& segmentation,
                                                 TDoubleDoublePrVec& depthAndPValue) {

    if (segmentation.size() - 1 > maxSegments) {
        // Prune by depth breaking ties by descending p-value.
        auto splits = core::make_range(segmentation, 2, segmentation.size());
        COrderings::simultaneousSort(
            depthAndPValue, splits,
            [](const TDoubleDoublePr& lhs, const TDoubleDoublePr& rhs) {
                return COrderings::lexicographical_compare(lhs.first, -lhs.second,
                                                           rhs.first, -rhs.second);
            });
        LOG_TRACE(<< "depth and p-values = "
                  << core::CContainerPrinter::print(depthAndPValue));
        LOG_TRACE(<< "splits = " << core::CContainerPrinter::print(splits));
        segmentation.resize(maxSegments + 1);
    }
    std::sort(segmentation.begin(), segmentation.end());
}

template<typename ITR>
void CTimeSeriesSegmentation::fitTopDownPiecewiseLinear(ITR begin,
                                                        ITR end,
                                                        std::size_t depth,
                                                        std::size_t offset,
                                                        std::size_t maxDepth,
                                                        double pValueToSegment,
                                                        double outlierFraction,
                                                        TSizeVec& segmentation,
                                                        TDoubleDoublePrVec& depthAndPValue,
                                                        TFloatMeanAccumulatorVec& values) {

    // We want to find the partition of [begin, end) into linear models which
    // efficiently predicts the values. To this end we maximize R^2 whilst
    // maintaining a parsimonious model. We achieve the later objective by
    // doing model selection for each candidate segment using the significance
    // of the variance ratio.

    auto range = std::distance(begin, end);
    int step{3};
    if (range < 2 * step || depth > maxDepth) {
        return;
    }

    LOG_TRACE(<< "Considering splitting [" << offset << "," << offset + range << ")");

    double startTime{static_cast<double>(offset)};
    values.assign(begin, end);
    TMeanVarAccumulator moments{[&] {
        TRegression model{fitLinearModel(values.cbegin(), values.cend(), startTime)};
        auto predict = [&](std::size_t i) {
            double time{startTime + static_cast<double>(i)};
            return model.predict(time);
        };
        CSignal::reweightOutliers(predict, outlierFraction, values);
        return centredResidualMoments(values.begin(), values.end(), startTime);
    }()};

    if (CBasicStatistics::variance(moments) > 0.0) {
        using TDoubleItrPr = std::pair<double, ITR>;
        using TMinAccumulator = typename CBasicStatistics::SMin<TDoubleItrPr>::TAccumulator;

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

        for (auto reweights : {0, 1}) {
            LOG_TRACE(<< "  reweights = " << reweights);

            ITR i = values.cbegin();
            minResidualVariance = TMinAccumulator{};
            double splitTime{startTime};
            TRegression leftModel;
            TRegression rightModel{fitLinearModel(i, values.cend(), splitTime)};

            auto varianceAfterStep = [&](int s) {
                TRegression deltaModel{fitLinearModel(i, i + s, splitTime)};
                return variance(residualMoments(values.cbegin(), i + s, startTime,
                                                leftModel + deltaModel),
                                residualMoments(i + s, values.cend(), splitTime + s,
                                                rightModel - deltaModel));
            };

            for (/**/; i + step < values.cend(); i += step, splitTime += step) {
                if (minResidualVariance.add({varianceAfterStep(step), i + step})) {
                    for (int s = 1; s < step; ++s) {
                        minResidualVariance.add({varianceAfterStep(s), i + s});
                    }
                }
                TRegression deltaModel{fitLinearModel(i, i + step, splitTime)};
                leftModel += deltaModel;
                rightModel -= deltaModel;
            }

            if (outlierFraction < 0.0 || outlierFraction >= 1.0) {
                LOG_ERROR(<< "Ignoring bad value for outlier fraction " << outlierFraction
                          << ". This must be in the range [0,1).");
                break;
            } else if (reweights == 1 || outlierFraction == 0.0) {
                break;
            } else {
                ITR split{minResidualVariance[0].second};
                splitTime = static_cast<double>(std::distance(values.cbegin(), split));
                leftModel = fitLinearModel(values.cbegin(), split, startTime);
                rightModel = fitLinearModel(split, values.cend(), splitTime);
                auto predict = [&](std::size_t j) {
                    double time{startTime + static_cast<double>(j)};
                    return (time < splitTime ? leftModel : rightModel).predict(time);
                };
                values.assign(begin, end);
                CSignal::reweightOutliers(predict, outlierFraction, values);
            }
        }

        ITR split{minResidualVariance[0].second};
        std::size_t splitOffset(std::distance(values.cbegin(), split));
        double splitTime{static_cast<double>(splitOffset)};
        LOG_TRACE(<< "  candidate = " << offset + splitOffset
                  << ", variance = " << minResidualVariance[0].first);

        TMeanVarAccumulator leftMoments{
            centredResidualMoments(values.cbegin(), split, startTime)};
        TMeanVarAccumulator rightMoments{
            centredResidualMoments(split, values.cend(), splitTime)};
        TMeanAccumulator meanAbs{std::accumulate(
            begin, end, TMeanAccumulator{}, [](auto partialMean, const auto& value) {
                partialMean.add(std::fabs(CBasicStatistics::mean(value)),
                                CBasicStatistics::count(value));
                return partialMean;
            })};
        LOG_TRACE(<< "  total moments = " << moments << ", left moments = " << leftMoments
                  << ", right moments = " << rightMoments << ", mean abs = " << meanAbs);

        double v[]{CBasicStatistics::maximumLikelihoodVariance(moments),
                   CBasicStatistics::maximumLikelihoodVariance(leftMoments + rightMoments) +
                       MINIMUM_COEFFICIENT_OF_VARIATION * CBasicStatistics::mean(meanAbs)};
        double df[]{4.0 - 2.0, static_cast<double>(range - 4)};
        double F{(df[1] * std::max(v[0] - v[1], 0.0)) / (df[0] * v[1])};
        double pValue{CTools::oneMinusPowOneMinusX(
            CStatisticalTests::rightTailFTest(F, df[0], df[1]),
            static_cast<double>(values.size() / step))};
        LOG_TRACE(<< "  p-value = " << pValue);

        if (pValue < pValueToSegment) {
            fitTopDownPiecewiseLinear(begin, begin + splitOffset, depth + 1, offset,
                                      maxDepth, pValueToSegment, outlierFraction,
                                      segmentation, depthAndPValue, values);
            fitTopDownPiecewiseLinear(begin + splitOffset, end, depth + 1, offset + splitOffset,
                                      maxDepth, pValueToSegment, outlierFraction,
                                      segmentation, depthAndPValue, values);
            segmentation.push_back(offset + splitOffset);
            depthAndPValue.emplace_back(static_cast<double>(depth), pValue);
        }
    }
}

CTimeSeriesSegmentation::TPredictor
CTimeSeriesSegmentation::fitPiecewiseLinear(const TSizeVec& segmentation,
                                            double outlierFraction,
                                            TFloatMeanAccumulatorVec& values) {
    using TRegressionVec = std::vector<TRegression>;

    TDoubleVec segmentEndTimes(segmentation.size() - 1,
                               static_cast<double>(values.size()));
    TRegressionVec models(segmentation.size() - 1);

    for (std::size_t i = 1; i < segmentation.size(); ++i) {
        double a{static_cast<double>(segmentation[i - 1])};
        double b{static_cast<double>(segmentation[i])};
        segmentEndTimes[i - 1] = b;
        models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                       values.begin() + segmentation[i], a);
    }
    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    auto predict = [&](std::size_t i) {
        double time{static_cast<double>(i)};
        auto index = std::upper_bound(segmentEndTimes.begin(), segmentEndTimes.end(), time);
        return models[index - segmentEndTimes.begin()].predict(time);
    };

    if (CSignal::reweightOutliers(predict, outlierFraction, values)) {
        for (std::size_t i = 1; i < segmentation.size(); ++i) {
            double a{static_cast<double>(segmentation[i - 1])};
            models[i - 1] = fitLinearModel(values.begin() + segmentation[i - 1],
                                           values.begin() + segmentation[i], a);
            LOG_TRACE(<< "model = " << models[i - 1].print());
        }
    }

    return [=](double time) {
        auto index = std::upper_bound(segmentEndTimes.begin(), segmentEndTimes.end(), time);
        return models[index - segmentEndTimes.begin()].predict(time);
    };
}

template<typename ITR>
void CTimeSeriesSegmentation::fitTopDownPiecewiseLinearScaledSeasonal(
    ITR begin,
    ITR end,
    std::size_t depth,
    std::size_t offset,
    const TSeasonality& model,
    std::size_t maxDepth,
    double pValueToSegment,
    TSizeVec& segmentation,
    TDoubleDoublePrVec& depthAndPValue) {

    // We want to find the partition of [begin, end) into scaled seasonal
    // models which efficiently predicts the values. To this end we maximize
    // R^2 whilst maintaining a parsimonious model. We achieve the later
    // objective by doing model selection for each candidate segment using
    // the significance of the explained variance divided by the split model
    // residual variance.

    using TDoubleItrPr = std::pair<double, ITR>;
    using TMinAccumulator =
        CBasicStatistics::COrderStatisticsStack<TDoubleItrPr, 1, COrderings::SFirstLess>;
    using TMeanAccumulatorAry = std::array<TMeanAccumulator, 3>;

    std::size_t range{static_cast<std::size_t>(std::distance(begin, end))};
    if (range < 4 || depth > maxDepth) {
        return;
    }

    LOG_TRACE(<< "Considering splitting [" << offset << "," << offset + range << ")");

    TMeanVarAccumulator moments{centredResidualMoments(begin, end, offset, model)};

    if (CBasicStatistics::variance(moments) > 0.0) {
        TMinAccumulator minResidualVariance;

        auto statistics = [offset, begin, &model](ITR begin_, ITR end_) {
            TMeanAccumulatorAry result;
            for (ITR i = begin_; i != end_; ++i) {
                double x{CBasicStatistics::mean(*i)};
                double w{CBasicStatistics::count(*i)};
                double p{model(offset + std::distance(begin, i))};
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
                      (CBasicStatistics::mean(l[2]) == 0.0
                           ? 0.0
                           : CTools::pow2(CBasicStatistics::mean(l[1])) /
                                 CBasicStatistics::mean(l[2]))};
            double vr{CBasicStatistics::mean(r[0]) -
                      (CBasicStatistics::mean(l[2]) == 0.0
                           ? 0.0
                           : CTools::pow2(CBasicStatistics::mean(r[1])) /
                                 CBasicStatistics::mean(r[2]))};
            return (nl * vl + nr * vr) / (nl + nr);
        };

        ITR i{std::next(begin, 2)};

        TMeanAccumulatorAry leftStatistics{statistics(begin, i)};
        TMeanAccumulatorAry rightStatistics{statistics(i, end)};
        minResidualVariance.add({variance(leftStatistics, rightStatistics), i});

        for (ITR end_ = std::prev(end, 2); i != end_; ++i) {
            TMeanAccumulatorAry deltaStatistics{statistics(i, std::next(i))};
            for (std::size_t j = 0; j < 3; ++j) {
                leftStatistics[j] += deltaStatistics[j];
                rightStatistics[j] -= deltaStatistics[j];
            }
            minResidualVariance.add(
                {variance(leftStatistics, rightStatistics), std::next(i)});
        }

        ITR split{minResidualVariance[0].second};
        std::size_t splitOffset{offset + std::distance(begin, split)};
        LOG_TRACE(<< "  candidate = " << splitOffset
                  << ", variance = " << minResidualVariance[0].first);

        TMeanVarAccumulator leftMoments{centredResidualMoments(begin, split, offset, model)};
        TMeanVarAccumulator rightMoments{
            centredResidualMoments(split, end, splitOffset, model)};
        TMeanAccumulator meanAbs{std::accumulate(
            begin, end, TMeanAccumulator{}, [](auto partialMean, const auto& value) {
                partialMean.add(std::fabs(CBasicStatistics::mean(value)),
                                CBasicStatistics::count(value));
                return partialMean;
            })};
        LOG_TRACE(<< "  total moments = " << moments << ", left moments = " << leftMoments
                  << ", right moments = " << rightMoments << ", mean abs = " << meanAbs);

        double v[]{CBasicStatistics::maximumLikelihoodVariance(moments),
                   CBasicStatistics::maximumLikelihoodVariance(leftMoments + rightMoments) +
                       MINIMUM_COEFFICIENT_OF_VARIATION * CBasicStatistics::mean(meanAbs)};
        double df[]{2.0 - 1.0, static_cast<double>(range - 2)};
        double F{df[1] * std::max(v[0] - v[1], 0.0) / (df[0] * v[1])};
        double pValue{CTools::oneMinusPowOneMinusX(
            CStatisticalTests::rightTailFTest(F, df[0], df[1]),
            static_cast<double>(range - 4))};
        LOG_TRACE(<< "  p-value = " << pValue);

        if (pValue < pValueToSegment) {
            fitTopDownPiecewiseLinearScaledSeasonal(begin, split, depth + 1, offset,
                                                    model, maxDepth, pValueToSegment,
                                                    segmentation, depthAndPValue);
            fitTopDownPiecewiseLinearScaledSeasonal(split, end, depth + 1, splitOffset,
                                                    model, maxDepth, pValueToSegment,
                                                    segmentation, depthAndPValue);
            segmentation.push_back(splitOffset);
            depthAndPValue.emplace_back(static_cast<double>(depth), pValue);
        }
    }
}

void CTimeSeriesSegmentation::fitPiecewiseLinearScaledSeasonal(
    const TFloatMeanAccumulatorVec& values,
    const TSeasonalComponentVec& periods,
    const TSizeVec& segmentation,
    double outlierFraction,
    TFloatMeanAccumulatorVec& reweighted,
    TDoubleVecVec& models,
    TDoubleVec& scales) {

    models.assign(periods.size(), {});
    scales.assign(segmentation.size() - 1, 1.0);

    auto seasonality = [&](std::size_t i) {
        double result{0.0};
        for (std::size_t j = 0; j < periods.size(); ++j) {
            if (periods[j].contains(i)) {
                result += models[j][periods[j].offset(i)];
            }
        }
        return result;
    };

    auto scale = [&](std::size_t i) { return scaleAt(i, segmentation, scales); };

    auto scaledSeasonality = [&](std::size_t i) {
        return scale(i) * seasonality(i);
    };

    auto computeScale = [&](std::size_t begin, std::size_t end) {
        TMeanAccumulator projection;
        TMeanAccumulator Z;
        for (std::size_t i = begin; i < end; ++i) {
            double x{CBasicStatistics::mean(reweighted[i])};
            double w{CBasicStatistics::count(reweighted[i])};
            double p{seasonality(i)};
            if (w > 0.0) {
                projection.add(x * p, w);
                Z.add(p * p, w);
            }
        }
        return CBasicStatistics::mean(Z) == 0.0
                   ? 0.0
                   : CBasicStatistics::mean(projection) / CBasicStatistics::mean(Z);
    };

    auto fitSeasonalModels = [&](const TScale& scale_) {
        for (std::size_t i = 0; i < periods.size(); ++i) {
            models[i].assign(periods[i].period(), 0.0);
        }
        std::size_t iterations(periods.size() > 1 ? 2 : 1);
        for (std::size_t i = 0; i < iterations; ++i) {
            for (std::size_t j = 0; j < periods.size(); ++j) {
                models[j].assign(periods[j].period(), 0.0);
                fitSeasonalModel(reweighted.begin(), reweighted.end(),
                                 periods[j], seasonality, scale_, models[j]);
            }
        }
    };

    auto fitScaledSeasonalModels = [&](const TScale& scale_) {
        for (std::size_t i = 0; i < 2; ++i) {
            fitSeasonalModels(scale_);
            for (std::size_t j = 1; j < segmentation.size(); ++j) {
                scales[j - 1] = computeScale(segmentation[j - 1], segmentation[j]);
            }
        }
    };

    LOG_TRACE(<< "segmentation = " << core::CContainerPrinter::print(segmentation));

    // First pass to re-weight any large outliers.
    reweighted = values;
    fitSeasonalModels([](std::size_t) { return 1.0; });
    CSignal::reweightOutliers(scaledSeasonality, outlierFraction, reweighted);

    // Fit the model adjusting for scales.
    fitScaledSeasonalModels(scale);
    LOG_TRACE(<< "models = " << core::CContainerPrinter::print(models));
    LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));

    // Re-weight outliers based on the new model.
    reweighted = values;
    if (CSignal::reweightOutliers(scaledSeasonality, outlierFraction, reweighted)) {
        // If any re-weighting happened fine tune.
        fitScaledSeasonalModels(scale);
        LOG_TRACE(<< "model = " << core::CContainerPrinter::print(models));
        LOG_TRACE(<< "scales = " << core::CContainerPrinter::print(scales));
    }
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulator
CTimeSeriesSegmentation::centredResidualMoments(ITR begin, ITR end, double startTime) {
    if (begin == end) {
        return CBasicStatistics::momentsAccumulator(0.0, 0.0, 0.0);
    }
    if (std::distance(begin, end) == 1) {
        return CBasicStatistics::momentsAccumulator(CBasicStatistics::count(*begin),
                                                    0.0, 0.0);
    }
    TRegression model{fitLinearModel(begin, end, startTime)};
    TMeanVarAccumulator moments{residualMoments(begin, end, startTime, model)};
    CBasicStatistics::moment<0>(moments) = 0.0;
    return moments;
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulator
CTimeSeriesSegmentation::centredResidualMoments(ITR begin,
                                                ITR end,
                                                std::size_t offset,
                                                const TSeasonality& model) {
    if (begin == end) {
        return CBasicStatistics::momentsAccumulator(0.0, 0.0, 0.0);
    }
    if (std::distance(begin, end) == 1) {
        return CBasicStatistics::momentsAccumulator(CBasicStatistics::count(*begin),
                                                    0.0, 0.0);
    }

    TMeanAccumulator projection;
    TMeanAccumulator norm2;
    for (ITR i = begin; i != end; ++i) {
        double x{CBasicStatistics::mean(*i)};
        double w{CBasicStatistics::count(*i)};
        double p{model(offset + std::distance(begin, i))};
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
        double p{scale * model(offset + std::distance(begin, i))};
        if (w > 0.0) {
            moments.add(x - p, w);
        }
    }
    CBasicStatistics::moment<0>(moments) = 0.0;
    LOG_TRACE(<< "  moments = " << moments);

    return moments;
}

template<typename ITR>
CTimeSeriesSegmentation::TMeanVarAccumulator
CTimeSeriesSegmentation::residualMoments(ITR begin, ITR end, double startTime, const TRegression& model) {
    TMeanVarAccumulator moments;
    TRegression::TArray params;
    model.parameters(params);
    for (double time = startTime; begin != end; ++begin, time += 1.0) {
        double w{CBasicStatistics::count(*begin)};
        if (w > 0.0) {
            double x{CBasicStatistics::mean(*begin) - TRegression::predict(params, time)};
            moments.add(x, w);
        }
    }
    return moments;
}

template<typename ITR>
CTimeSeriesSegmentation::TRegression
CTimeSeriesSegmentation::fitLinearModel(ITR begin, ITR end, double startTime) {
    TRegression result;
    for (double time = startTime; begin != end; ++begin) {
        double x{CBasicStatistics::mean(*begin)};
        double w{CBasicStatistics::count(*begin)};
        if (w > 0.0) {
            result.add(time, x, w);
        }
        time += 1.0;
    }
    return result;
}

template<typename ITR>
void CTimeSeriesSegmentation::fitSeasonalModel(ITR begin,
                                               ITR end,
                                               const TSeasonalComponent& period,
                                               const TPredictor& predictor,
                                               const TScale& scale,
                                               TDoubleVec& result) {
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;

    TMeanAccumulatorVec model(period.period());
    for (std::size_t i = 0; begin != end; ++i, ++begin) {
        if (period.contains(i)) {
            double x{CBasicStatistics::mean(*begin) / scale(i)};
            double w{CBasicStatistics::count(*begin) * scale(i)};
            if (w > 0.0) {
                model[period.offset(i)].add(x - predictor(static_cast<double>(i)), w);
            }
        }
    }

    for (std::size_t i = 0; i < model.size(); ++i) {
        result[i] = CBasicStatistics::mean(model[i]);
    }
}
}
}
